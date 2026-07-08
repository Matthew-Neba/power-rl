"""Numeric gradient check for the Nethack Mixer/Patch CUDA encoder (src/ocean.cu).

Same method as test_nethack_encoder.py: builds tests/test_nethack_mixer_cuda.cu
as a float shared lib and verifies analytic gradients from backward() against
central finite differences of L = sum(out * g_out), with ReLU-kink detection.
tok_w2/ch_w2 are zero-initialized (identity residual blocks), so the test
randomizes them first — otherwise the mixer hidden layers would receive
trivially zero gradient and go unchecked.

Run: python tests/test_nethack_mixer.py
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LIB = os.path.join(HERE, "nethack_mixer_test.so")

VP = ctypes.c_void_p

PARAMS = ["embed_w", "stem_w", "stem_b", "tok_w1", "tok_b1", "tok_w2", "tok_b2",
          "ch_w1", "ch_b1", "ch_w2", "ch_b2", "pool_w",
          "bl_w", "bl_b", "proj_w", "proj_b"]


def build():
    cu = os.path.join(HERE, "test_nethack_mixer_cuda.cu")
    cmd = [
        "nvcc", "-shared", "-o", LIB, cu, "-I", SRC,
        "-lcublas", "-lcublasLt", "-lcudnn", "-lcurand",
        "--compiler-options", "-fPIC", "-Xcompiler", "-O2", "-arch=native",
    ]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load():
    lib = ctypes.CDLL(LIB)
    for p in PARAMS:
        for pre in ["nhm_get_", "nhm_set_", "nhm_grad_"]:
            getattr(lib, pre + p).argtypes = [VP]
            getattr(lib, pre + p).restype = None
        getattr(lib, "nhm_numel_" + p).restype = ctypes.c_int
    lib.nhm_init.argtypes = [ctypes.c_int] * 5
    for fn in ["nhm_obs_size", "nhm_grid", "nhm_vocab", "nhm_embed_dim"]:
        getattr(lib, fn).restype = ctypes.c_int
    lib.nhm_forward.argtypes = [VP, VP, ctypes.c_int]
    lib.nhm_backward.argtypes = [VP, ctypes.c_int]
    return lib


def make_obs(B, obs_size, grid, max_glyph_used):
    obs = np.zeros((B, obs_size), dtype=np.float32)
    rng = np.random.default_rng(0)
    glyphs = rng.integers(0, max_glyph_used, size=(B, grid)).astype(np.int32)
    obs[:, 0:2 * grid:2] = (glyphs & 0xFF).astype(np.float32)
    obs[:, 1:2 * grid:2] = ((glyphs >> 8) & 0xFF).astype(np.float32)
    bl_off = 2 * grid
    vals = rng.integers(-5, 500, size=(B, 27)).astype(np.int64)
    vals[:, 25] = rng.integers(0, 8192, size=B)
    u = vals.astype(np.uint32)
    for k in range(4):
        obs[:, bl_off + k::4][:, :27] = ((u >> (8 * k)) & 0xFF).astype(np.float32)
    # extra stats: prayer cooldown, prev action (-1..23), 18 class counts
    ex = np.concatenate([
        rng.integers(0, 1000, size=(B, 1)),
        rng.integers(-1, 24, size=(B, 1)),
        rng.integers(0, 6, size=(B, 18)),
    ], axis=1).astype(np.int64).astype(np.uint32)
    for k in range(4):
        obs[:, bl_off + k::4][:, 27:47] = ((ex >> (8 * k)) & 0xFF).astype(np.float32)
    return obs, glyphs


_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaMalloc.argtypes = [VP, ctypes.c_size_t]
_cudart.cudaMemcpy.argtypes = [VP, VP, ctypes.c_size_t, ctypes.c_int]


def dev(nbytes):
    p = VP()
    _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes))
    return p


def h2d(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    p = dev(arr.nbytes)
    _cudart.cudaMemcpy(p, arr.ctypes.data_as(VP), ctypes.c_size_t(arr.nbytes), 1)
    return p, arr.nbytes


def d2h(p, n):
    out = np.empty(n, dtype=np.float32)
    _cudart.cudaMemcpy(out.ctypes.data_as(VP), p, ctypes.c_size_t(n * 4), 2)
    return out


def run(lib, variant="mixer", C=64, D=16):
    B, hidden = 4, 24
    mode = {"patch": 0, "mixer": 1, "minpatch": 2, "cellflat": 3}[variant]
    lib.nhm_init(B, hidden, mode, C, D)
    params = PARAMS
    if variant != "mixer":
        params = [p for p in params if not p.startswith(("tok_", "ch_"))]
    if variant != "minpatch":
        params = [p for p in params if p != "pool_w"]
    if variant == "cellflat":
        params = [p for p in params if not p.startswith("stem_")]
    obs_size, grid = lib.nhm_obs_size(), lib.nhm_grid()
    print(f"variant: {variant} C={C} D={D}")
    print(f"obs_size={obs_size} grid={grid} vocab={lib.nhm_vocab()} embed_dim={lib.nhm_embed_dim()}")

    # Residual output layers init at zero; randomize so hidden layers get grads.
    rng = np.random.default_rng(42)
    for name in (["tok_w2", "ch_w2"] if variant == "mixer" else []):
        n = getattr(lib, "nhm_numel_" + name)()
        w = (rng.standard_normal(n) * 0.1).astype(np.float32)
        getattr(lib, "nhm_set_" + name)(w.ctypes.data_as(VP))

    max_glyph_used = 40
    obs, glyphs = make_obs(B, obs_size, grid, max_glyph_used)
    obs_d, _ = h2d(obs)
    out_d = dev(B * hidden * 4)

    rng = np.random.default_rng(7)
    g_out = rng.standard_normal((B, hidden)).astype(np.float32)

    def forward_loss():
        lib.nhm_forward(out_d, obs_d, B)
        out = d2h(out_d, B * hidden).reshape(B, hidden)
        return float((out * g_out).sum())

    L0 = forward_loss()
    grad_d, _ = h2d(g_out)
    lib.nhm_backward(grad_d, B)

    eps = 1e-3
    kink_tol = 2e-4
    rel_tol = 1.5e-2
    rng = np.random.default_rng(123)
    all_ok = True
    for name in params:
        get = getattr(lib, "nhm_get_" + name)
        setw = getattr(lib, "nhm_set_" + name)
        gradf = getattr(lib, "nhm_grad_" + name)
        n = getattr(lib, "nhm_numel_" + name)()
        w0 = np.empty(n, dtype=np.float32); get(w0.ctypes.data_as(VP))
        ga = np.empty(n, dtype=np.float32); gradf(ga.ctypes.data_as(VP))

        if name == "embed_w":
            used = np.unique(glyphs)
            D = lib.nhm_embed_dim()
            cand = np.array([g * D + d for g in used for d in range(D)], dtype=np.int64)
        else:
            cand = np.arange(n, dtype=np.int64)
        cand = cand[np.abs(ga[cand]) > 1e-3]
        if len(cand) == 0:
            print(f"  [SKIP] {name:8s} n={n:8d} (no entry with |grad|>1e-3)")
            continue
        rng.shuffle(cand)

        rels, skipped = [], 0
        for i in cand:
            if len(rels) >= 12:
                break
            i = int(i)
            wp = w0.copy(); wp[i] += eps; setw(wp.ctypes.data_as(VP)); Lp = forward_loss()
            wm = w0.copy(); wm[i] -= eps; setw(wm.ctypes.data_as(VP)); Lm = forward_loss()
            setw(w0.ctypes.data_as(VP))
            if abs(Lp + Lm - 2 * L0) > kink_tol:
                skipped += 1
                continue
            gnum = (Lp - Lm) / (2 * eps)
            rels.append(abs(gnum - ga[i]) / max(1.0, abs(gnum), abs(ga[i])))
        # The mixer stacks 3 ReLUs after the early params, so FD has a
        # near-kink noise tail even below kink_tol; the median is robust to
        # it while still catching systematic errors (a real bug gives O(1)).
        med = float(np.median(rels)) if rels else 1.0
        mx = max(rels) if rels else 1.0
        ok = len(rels) >= 3 and med < 1e-2 and mx < 0.15
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:8s} n={n:8d} "
              f"checked={len(rels)} kink_skipped={skipped} "
              f"max|analytic|={np.abs(ga).max():.4g} med_rel={med:.2e} max_rel={mx:.2e}")
    return all_ok


if __name__ == "__main__":
    if "--no-build" not in sys.argv:
        build()
    lib = load()
    ok = run(lib, variant="mixer")
    ok = run(lib, variant="patch") and ok
    ok = run(lib, variant="minpatch") and ok
    ok = run(lib, variant="cellflat") and ok
    ok = run(lib, variant="patch", C=96, D=48) and ok
    ok = run(lib, variant="patch", C=128, D=32) and ok
    ok = run(lib, variant="cellflat", D=48) and ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
