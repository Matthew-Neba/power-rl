"""Numeric gradient check for the Nethack CUDA encoder (src/nethack.cu).

Builds tests/test_nethack_cuda.cu as a float shared lib and verifies the
analytic gradients from encoder backward() against central finite differences
of a scalar loss L = sum(out * g_out). This exercises the whole hand-written
backward chain: projection -> concat split -> conv2/conv1 (incl. conv1 input
grad) -> glyph-embedding scatter-add, plus the blstats Linear branch.

Run: python tests/test_nethack_encoder.py
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LIB = os.path.join(HERE, "nethack_test.so")

VP = ctypes.c_void_p


def build():
    cu = os.path.join(HERE, "test_nethack_cuda.cu")
    cmd = [
        "nvcc", "-shared", "-o", LIB, cu, "-I", SRC,
        "-lcublas", "-lcublasLt", "-lcudnn", "-lcurand",
        "--compiler-options", "-fPIC", "-Xcompiler", "-O2", "-arch=native",
    ]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load():
    lib = ctypes.CDLL(LIB)
    for fn in ["nh_forward", "nh_backward", "nh_get_embed_w", "nh_set_embed_w",
               "nh_get_bl_w", "nh_set_bl_w", "nh_get_bl_b", "nh_set_bl_b",
               "nh_get_proj_w", "nh_set_proj_w", "nh_get_proj_b", "nh_set_proj_b",
               "nh_get_conv1_w", "nh_set_conv1_w", "nh_get_conv2_w", "nh_set_conv2_w",
               "nh_grad_embed_w", "nh_grad_bl_w", "nh_grad_bl_b", "nh_grad_proj_w",
               "nh_grad_proj_b", "nh_grad_conv1_w", "nh_grad_conv2_w"]:
        getattr(lib, fn).argtypes = [VP]
        getattr(lib, fn).restype = None
    lib.nh_init.argtypes = [ctypes.c_int, ctypes.c_int]
    for fn in ["nh_obs_size", "nh_bl_feat", "nh_glyph_vocab", "nh_embed_dim",
               "nh_concat", "nh_grid", "nh_numel_embed_w", "nh_numel_bl_w",
               "nh_numel_bl_b", "nh_numel_proj_w", "nh_numel_proj_b",
               "nh_numel_conv1_w", "nh_numel_conv2_w"]:
        getattr(lib, fn).restype = ctypes.c_int
    lib.nh_forward.argtypes = [VP, VP, ctypes.c_int]
    lib.nh_backward.argtypes = [VP, ctypes.c_int]
    return lib


def make_obs(B, obs_size, grid, max_glyph_used):
    """Build a valid packed obs: glyphs int16 LE, then blstats int32 LE, as
    byte-valued float32 (matching cast_dispatch's per-byte float cast)."""
    obs = np.zeros((B, obs_size), dtype=np.float32)
    rng = np.random.default_rng(0)
    # glyphs @0: grid cells, 2 bytes each (restrict to a small glyph set so many
    # embedding rows receive gradient and are individually checkable).
    glyphs = rng.integers(0, max_glyph_used, size=(B, grid)).astype(np.int32)
    lo = (glyphs & 0xFF).astype(np.float32)
    hi = ((glyphs >> 8) & 0xFF).astype(np.float32)
    obs[:, 0:2 * grid:2] = lo
    obs[:, 1:2 * grid:2] = hi
    # blstats @ 2*grid: 27 int32, mixed magnitudes incl. negatives (AC/align).
    bl_off = 2 * grid
    vals = rng.integers(-5, 500, size=(B, 27)).astype(np.int64)
    vals[:, 25] = rng.integers(0, 8192, size=B)  # CONDITION bitmask
    u = vals.astype(np.uint32)
    for k in range(4):
        obs[:, bl_off + k::4][:, :27] = ((u >> (8 * k)) & 0xFF).astype(np.float32)
    # extra stats @ +27*4: prayer cooldown, prev action (-1..23), 18 class counts
    ex = np.concatenate([
        rng.integers(0, 1000, size=(B, 1)),
        rng.integers(-1, 24, size=(B, 1)),
        rng.integers(0, 6, size=(B, 18)),
    ], axis=1).astype(np.int64).astype(np.uint32)
    for k in range(4):
        obs[:, bl_off + k::4][:, 27:47] = ((ex >> (8 * k)) & 0xFF).astype(np.float32)
    return obs, glyphs


def dev(nbytes):
    import ctypes
    p = VP()
    _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes))
    return p


def h2d(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    p = dev(arr.nbytes)
    _cudart.cudaMemcpy(p, arr.ctypes.data_as(VP), ctypes.c_size_t(arr.nbytes), 1)  # H2D
    return p, arr.nbytes


def d2h(p, n):
    out = np.empty(n, dtype=np.float32)
    _cudart.cudaMemcpy(out.ctypes.data_as(VP), p, ctypes.c_size_t(n * 4), 2)  # D2H
    return out


_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaMalloc.argtypes = [VP, ctypes.c_size_t]
_cudart.cudaMemcpy.argtypes = [VP, VP, ctypes.c_size_t, ctypes.c_int]


def run(lib):
    B, hidden = 4, 24
    lib.nh_init(B, hidden)
    obs_size = lib.nh_obs_size()
    grid = lib.nh_grid()
    vocab = lib.nh_glyph_vocab()
    print(f"obs_size={obs_size} grid={grid} vocab={vocab} "
          f"bl_feat={lib.nh_bl_feat()} concat={lib.nh_concat()}")

    max_glyph_used = 40  # keep embedding usage dense & checkable
    obs, glyphs = make_obs(B, obs_size, grid, max_glyph_used)
    obs_d, _ = h2d(obs)
    out_d = dev(B * hidden * 4)

    # Fixed upstream grad g_out; loss L = sum(out * g_out).
    rng = np.random.default_rng(7)
    g_out = rng.standard_normal((B, hidden)).astype(np.float32)

    def forward_loss():
        lib.nh_forward(out_d, obs_d, B)
        out = d2h(out_d, B * hidden).reshape(B, hidden)
        return float((out * g_out).sum())

    # Analytic grads: forward then backward with dL/dout = g_out.
    L0 = forward_loss()
    grad_d, _ = h2d(g_out)  # backward mutates grad in place
    lib.nh_backward(grad_d, B)

    specs = [
        ("proj_w",  lib.nh_get_proj_w,  lib.nh_set_proj_w,  lib.nh_grad_proj_w,  lib.nh_numel_proj_w),
        ("proj_b",  lib.nh_get_proj_b,  lib.nh_set_proj_b,  lib.nh_grad_proj_b,  lib.nh_numel_proj_b),
        ("bl_w",    lib.nh_get_bl_w,    lib.nh_set_bl_w,    lib.nh_grad_bl_w,    lib.nh_numel_bl_w),
        ("bl_b",    lib.nh_get_bl_b,    lib.nh_set_bl_b,    lib.nh_grad_bl_b,    lib.nh_numel_bl_b),
        ("conv2_w", lib.nh_get_conv2_w, lib.nh_set_conv2_w, lib.nh_grad_conv2_w, lib.nh_numel_conv2_w),
        ("conv1_w", lib.nh_get_conv1_w, lib.nh_set_conv1_w, lib.nh_grad_conv1_w, lib.nh_numel_conv1_w),
        ("embed_w", lib.nh_get_embed_w, lib.nh_set_embed_w, lib.nh_grad_embed_w, lib.nh_numel_embed_w),
    ]

    # Central finite differences of L = sum(out*g_out). The encoder ends in a
    # ReLU (and the blstats branch has its own), so a perturbation that flips a
    # unit's sign makes the FD non-smooth and disagree with the (correct)
    # subgradient. We detect such kink crossings via the second difference
    # (|L+ + L- - 2*L0| is O(eps) at a kink vs O(eps^2) on a smooth region)
    # and skip those entries. eps is small so smooth curvature stays negligible.
    eps = 1e-3
    kink_tol = 2e-4      # |Lp+Lm-2*L0| above this ⇒ a ReLU flipped; skip entry.
    rel_tol = 1.5e-2
    rng = np.random.default_rng(123)
    all_ok = True
    for name, get, setw, gradf, numelf in specs:
        n = numelf()
        w0 = np.empty(n, dtype=np.float32); get(w0.ctypes.data_as(VP))
        ga = np.empty(n, dtype=np.float32); gradf(ga.ctypes.data_as(VP))

        if name == "embed_w":
            used = np.unique(glyphs)
            D = lib.nh_embed_dim()
            cand = np.array([g * D + d for g in used for d in range(D)], dtype=np.int64)
        else:
            cand = np.arange(n, dtype=np.int64)
        cand = cand[np.abs(ga[cand]) > 1e-3]        # need signal for a meaningful ratio
        if len(cand) == 0:
            print(f"  [SKIP] {name:8s} n={n:8d} (no entry with |grad|>1e-3)")
            continue
        rng.shuffle(cand)

        max_rel, checked, skipped = 0.0, 0, 0
        for i in cand:
            if checked >= 10:
                break
            i = int(i)
            wp = w0.copy(); wp[i] += eps; setw(wp.ctypes.data_as(VP)); Lp = forward_loss()
            wm = w0.copy(); wm[i] -= eps; setw(wm.ctypes.data_as(VP)); Lm = forward_loss()
            setw(w0.ctypes.data_as(VP))  # restore
            if abs(Lp + Lm - 2 * L0) > kink_tol:     # ReLU kink crossing → FD invalid
                skipped += 1
                continue
            gnum = (Lp - Lm) / (2 * eps)
            rel = abs(gnum - ga[i]) / max(1.0, abs(gnum), abs(ga[i]))
            max_rel = max(max_rel, rel)
            checked += 1
        ok = checked >= 3 and max_rel < rel_tol
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:8s} n={n:8d} "
              f"checked={checked} kink_skipped={skipped} "
              f"max|analytic|={np.abs(ga).max():.4g} max_rel_err={max_rel:.2e}")
    return all_ok


if __name__ == "__main__":
    if "--no-build" not in sys.argv:
        build()
    ok = run(load())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
