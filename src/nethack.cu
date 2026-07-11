// NetHack CUDA encoder: one shared glyph embedding feeding two views of the
// full 79x21 map — an egocentric 9x9 crop at per-cell detail (flatten-linear)
// and a global 8x4-patch view (embed sums per patch, 10x6 = 60 tokens) — plus
// the blstats MLP. Included by ocean.cu — requires kernels.cu, models.cu.
// Bit-deterministic backward: fixed-point integer atomics and split-K GEMMs
// with atomic reduction schemes masked out.

#include <cublasLt.h>

// cublasGemmExDense through cublasLt's heuristic + workspace. GemmEx picks
// near-serial kernels for skinny or odd shapes (tiny-output/huge-K weight
// grads, muon's Newton-Schulz on wide encoder projections); the Lt heuristic
// split-Ks them properly. Same math, fp32 accumulation either way.
static inline void cublasLtGemmDense(
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    static thread_local cublasLtHandle_t lt = nullptr;
    static thread_local void* lt_ws = nullptr;
    static const size_t LT_WS_BYTES = 64 * 1024 * 1024;
    if (!lt) {
        cublasLtCreate(&lt);
        cudaMalloc(&lt_ws, LT_WS_BYTES);
    }
    // Row-major C(M,N) = op(A) @ op(B) is column-major C'(N,M) = op(B') @ op(A')
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasLtMatmulDesc_t op;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_PRECISION, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &op_b, sizeof(op_b));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &op_a, sizeof(op_a));
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUBLAS_PRECISION,
        op_b == CUBLAS_OP_N ? N : K, op_b == CUBLAS_OP_N ? K : N, ldb);
    cublasLtMatrixLayoutCreate(&lb, CUBLAS_PRECISION,
        op_a == CUBLAS_OP_N ? K : M, op_a == CUBLAS_OP_N ? M : K, lda);
    cublasLtMatrixLayoutCreate(&lc, CUBLAS_PRECISION, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &LT_WS_BYTES, sizeof(LT_WS_BYTES));
    // exclude INPLACE split-K (atomics): nondeterministic accumulation order
    uint32_t red_mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
        &red_mask, sizeof(red_mask));
    cublasLtMatmulHeuristicResult_t heur;
    int nheur = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &nheur);
    if (nheur > 0) {
        cublasLtMatmul(lt, op, &alpha, B, la, A, lb, &beta,
            C, lc, C, lc, &heur.algo, lt_ws, LT_WS_BYTES, stream);
    } else {
        cublasGemmExDense(op_a, op_b, M, N, K, A, B, C, stream, alpha, beta);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
}

// puf_mm_tn for tiny-output/huge-K weight grads (e.g. M,N < 256, K > 100k).
void puf_mm_tn_splitk(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasLtGemmDense(CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, a->data, b->data, out->data, stream);
}

__global__ void nh_bias_relu_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[idx % dim])));
}

// ---- Nethack constants ----
// Obs layout (must match ocean/nethack/nethack.h):
//   [0, 2*NH_MGRID)  full 79x21 glyph grid, int16 LE (map memory included)
//   [2*NH_MGRID, +4*NH_BL_RAW)  blstats, int32 LE (x,y first)
//   [+4*NH_BL_RAW, +4*NH_EX_RAW)  extra stats, int32 LE: prayer cooldown,
//                                 previous action, 18 inventory class counts

static constexpr int NH_MAPW = 79, NH_MAPH = 21;
static constexpr int NH_MGRID = NH_MAPW * NH_MAPH;
static constexpr int NH_GLYPH_VOCAB = 5977;        // MAX_GLYPH + 1 (NetHack 3.6.6)
static constexpr int NH_PAD_GLYPH = NH_GLYPH_VOCAB - 1;   // NO_GLYPH: off-map crop cells
static constexpr int NH_EMBED_DIM = 32;
static constexpr int NH_CROP = 9, NH_CHALF = 4;    // NETHACK_CROP, egocentric
static constexpr int NH_CGRID = NH_CROP * NH_CROP;
static constexpr int NH_PW = 5, NH_PH = 5;         // patch size (cells)
static constexpr int NH_PX = 16, NH_PY = 5;        // patch grid (ceil 79/5, 21/5)
static constexpr int NH_TOK = NH_PX * NH_PY;       // 80 global tokens
static constexpr int NH_PCELLS = NH_PW * NH_PH;    // cells per patch (off-map -> pad glyph)
static constexpr int NH_LOC_IN = NH_CGRID * NH_EMBED_DIM;
static constexpr int NH_LOC_HID = 256;
// Global branch: per patch, embed+flatten (25 cells x 32 dims) + normalized
// (dx,dy) patch-center offset from the hero -> 16 -> 128, then elementwise
// MAX over the 80 tokens. The 16-dim bottleneck keeps the fused per-glyph
// gather table (embed+flatten+first layer, NH_TROW cols) L2-resident, and
// fusing the max means the (B, 80, 128) token activations never exist in
// memory. The (dx,dy) slice lives in its own 16x2 weight tensor — same math
// as concatenating onto the flatten.
static constexpr int NH_P1 = 16;
static constexpr int NH_GLB_HID = 128;
static constexpr int NH_TROW = NH_PCELLS * NH_P1;  // fused-table row: per-pos 16-dim
static constexpr int NH_PAD_PER_SAMPLE = NH_TOK * NH_PCELLS - NH_MGRID;
static constexpr int NH_HOT_G = 10;                // hot-glyph dT smem slots (10x400 int64 = 32KB)
static constexpr int NH_BL_RAW = 27;               // NLE_BLSTATS_SIZE
static constexpr int NH_BL_HUNGER = 21, NH_BL_CONDITION = 25;
static constexpr int NH_ACTIONS = 12;              // NETHACK_NUM_ACTIONS
static constexpr int NH_OCLASSES = 18;             // MAXOCLASSES
static constexpr int NH_EX_RAW = 2 + NH_OCLASSES;  // NETHACK_EXTRA_INTS
// scalars + hunger onehot + condition bits + cooldown + prev-action onehot + counts
static constexpr int NH_BL_FEAT = 25 + 7 + 13 + 1 + NH_ACTIONS + NH_OCLASSES;
static constexpr int NH_BL_HID = 64;
// Inventory entity branch: 55 slot glyphs (the item action head indexes these
// positions), each embed -> shared 32->16 linear -> relu, flattened in slot
// order (position must survive: the policy has to name a slot index). Fused
// per-glyph table T_inv = E @ inv1_w^T (5977x16) rebuilt per forward.
static constexpr int NH_INV = 55;                  // NETHACK_INV_SLOTS
static constexpr int NH_INV_HID = 32;
static constexpr int NH_INV_FLAT = NH_INV * NH_INV_HID;
static constexpr int NH_CONCAT = NH_LOC_HID + NH_GLB_HID + NH_INV_FLAT + NH_BL_HID + NH_BL_FEAT;
static constexpr int NH_BL_OFF = 2 * NH_MGRID;     // blstats offset, obs elements
static constexpr int NH_INV_OFF = NH_BL_OFF + (NH_BL_RAW + NH_EX_RAW) * 4;
static constexpr int NH_OBS_SIZE = NH_INV_OFF + NH_INV * 2;
static constexpr int NH_SORT_BLOCKS = 256;         // hist grid (smem histograms)
static constexpr int NH_HOT_T = 16;                // hot-glyph smem rows (16x32 int64 = 4KB)

// 2^24 fixed-point gradient accumulators: integer atomics are associative, so
// scatter/bias sums are bit-identical run to run (float atomicAdd ordering is
// not). Quantization (6e-8) is below fp32 accumulation error at these counts.
static constexpr float NH_FXP = 16777216.0f;
__device__ __forceinline__ void nh_fxp_atomic_add(long long* addr, float v) {
    atomicAdd((unsigned long long*)addr, (unsigned long long)(long long)__float2ll_rn(v * NH_FXP));
}
__device__ __forceinline__ float nh_fxp_to_float(long long v) {
    return (float)((double)v * (1.0 / 16777216.0));
}
__global__ void nh_fxp_to_precision_kernel(
    precision_t* __restrict__ dst, const long long* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(nh_fxp_to_float(src[idx]));
}

// Row-sparse fixed-point -> precision cast for the embed-table grad: a
// minibatch touches only a few hundred of the 5977 rows (counts>0 or hot).
// Untouched rows skip the int64 read; touched rows are re-zeroed in place,
// replacing a full-table memset. Invariant: src is all-zero between
// iterations (alloc_create zeroes it once).
__global__ void nh_fxp_to_precision_rows_kernel(
    precision_t* __restrict__ dst, long long* __restrict__ src,
    const int* __restrict__ counts, const int* __restrict__ hot_map,
    int trow, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = idx / trow;
    if (counts[row] == 0 && hot_map[row] < 0) { dst[idx] = from_float(0.0f); return; }
    dst[idx] = from_float(nh_fxp_to_float(src[idx]));
    src[idx] = 0;
}

// Per-blstat normalization: log1p fields get log1p(max(v,0))*scale, the rest
// v*scale. Hunger (21) and condition (25) are expanded, not scaled.
__constant__ float NH_BL_SCALE[NH_BL_RAW] = {
    1.f/79, 1.f/21,                                  // x, y
    1.f/25, 1.f/125, 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,  // str25 str125 dex con int wis cha
    0.1f,                                            // score (log)
    1.f/200, 1.f/200, 1.f/50,                        // hp, hpmax, depth
    0.1f,                                            // gold (log)
    1.f/100, 1.f/100, 1.f/10, 1.f/10, 1.f/30,        // ene, enemax, ac, hd, xp level
    0.1f, 0.1f,                                      // exp points, time (log)
    0.f,                                             // hunger (expanded)
    1.f/4, 1.f/10, 1.f/50,                           // cap, dnum, dlevel
    0.f,                                             // condition (expanded)
    1.f,                                             // align
};
__constant__ int NH_BL_ISLOG[NH_BL_RAW] = {
    0,0,0,0,0,0,0,0,0, 1, 0,0,0, 1, 0,0,0,0,0, 1,1, 0, 0,0,0, 0, 0,
};

// ---- Nethack kernels ----

__device__ __forceinline__ int nh_bl_read_i32(const precision_t* p) {
    return (int)((unsigned int)(int)to_float(p[0])
               | ((unsigned int)(int)to_float(p[1]) << 8)
               | ((unsigned int)(int)to_float(p[2]) << 16)
               | ((unsigned int)(int)to_float(p[3]) << 24));
}

// Decode int16 LE glyph ids into an fp32 index buffer (full grid).
__global__ void nh_decode_kernel(
    float* __restrict__ idx, const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_MGRID) return;
    int b = t / NH_MGRID, cell = t % NH_MGRID;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + 2 * cell;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));
}

// Egocentric crop glyph ids: window centered on the hero (blstats x,y),
// off-map cells get the pad glyph.
__global__ void nh_crop_kernel(
    float* __restrict__ crop, const float* __restrict__ idx,
    const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_CGRID) return;
    int b = t / NH_CGRID, p = t % NH_CGRID;
    const precision_t* bl = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
    int r = nh_bl_read_i32(bl + 4) - NH_CHALF + p / NH_CROP;   // blstats[1] = y
    int c = nh_bl_read_i32(bl)     - NH_CHALF + p % NH_CROP;   // blstats[0] = x
    crop[t] = (r < 0 || r >= NH_MAPH || c < 0 || c >= NH_MAPW)
        ? (float)NH_PAD_GLYPH : idx[b * NH_MGRID + r * NH_MAPW + c];
}

// Local view: per-cell embedding gather, flattened (B, NH_CGRID*NH_EMBED_DIM).
__global__ void nh_local_gather_kernel(
    precision_t* __restrict__ x, const precision_t* __restrict__ E,
    const float* __restrict__ crop, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LOC_IN) return;
    int d = t % NH_EMBED_DIM;
    int64_t cell = t / NH_EMBED_DIM;   // b*NH_CGRID + p
    x[t] = E[(int64_t)(int)crop[cell] * NH_EMBED_DIM + d];
}

// glb1.w (P1, PCELLS*D) -> W' (PCELLS*P1, D), so T = E @ W'^T lands as
// T[g, pos*P1+k]: the fused embed+flatten+layer1 lookup table, rebuilt with
// one small GEMM whenever the weights change.
__global__ void nh_permute_g1_kernel(
    precision_t* __restrict__ wp, const precision_t* __restrict__ w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int pos = r / NH_P1, k = r % NH_P1;
    wp[i] = w[k * (NH_PCELLS * NH_EMBED_DIM) + pos * NH_EMBED_DIM + d];
}

// Inverse for the weight grad.
__global__ void nh_unpermute_g1_kernel(
    precision_t* __restrict__ wg, const precision_t* __restrict__ wpg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int pos = r / NH_P1, k = r % NH_P1;
    wg[k * (NH_PCELLS * NH_EMBED_DIM) + pos * NH_EMBED_DIM + d] = wpg[i];
}

// Fused global view, one block per sample: stage 1 builds the 80 relu'd
// 16-dim tokens in smem from the fused table (off-map positions read the pad
// glyph's row) plus the (dx,dy) hero-offset slice; stage 2 expands each token
// 16->128 in registers and folds the elementwise max on the fly (argmax saved
// for the sparse backward). Fixed token order + strict > keep argmax
// deterministic. dxy is saved for the w_xy weight grad.
__global__ void nh_patch_max_kernel(
    precision_t* __restrict__ glb_out, precision_t* __restrict__ t16_save,
    precision_t* __restrict__ dxy_save, int* __restrict__ argmax,
    const precision_t* __restrict__ T, const precision_t* __restrict__ b1,
    const precision_t* __restrict__ w_xy, const precision_t* __restrict__ w2,
    const precision_t* __restrict__ b2, const float* __restrict__ idx,
    const precision_t* __restrict__ obs, int B) {
    __shared__ float w2s[NH_GLB_HID * NH_P1];
    __shared__ float t16s[NH_TOK * NH_P1];
    __shared__ float hero[2];
    int b = blockIdx.x;
    if (b >= B) return;
    if (threadIdx.x == 0) {
        const precision_t* bl = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
        hero[0] = (float)nh_bl_read_i32(bl);       // x
        hero[1] = (float)nh_bl_read_i32(bl + 4);   // y
    }
    for (int i = threadIdx.x; i < NH_GLB_HID * NH_P1; i += blockDim.x)
        w2s[i] = to_float(w2[i]);
    __syncthreads();
    const float* gi = idx + (int64_t)b * NH_MGRID;
    for (int i = threadIdx.x; i < NH_TOK * NH_P1; i += blockDim.x) {
        int tk = i / NH_P1, k = i % NH_P1;
        int r0 = (tk / NH_PX) * NH_PH, c0 = (tk % NH_PX) * NH_PW;
        float dx = (c0 + 0.5f * (NH_PW - 1) - hero[0]) * (1.0f / NH_MAPW);
        float dy = (r0 + 0.5f * (NH_PH - 1) - hero[1]) * (1.0f / NH_MAPH);
        float acc = to_float(b1[k])
                  + dx * to_float(w_xy[k * 2]) + dy * to_float(w_xy[k * 2 + 1]);
        #pragma unroll
        for (int pos = 0; pos < NH_PCELLS; pos++) {
            int r = r0 + pos / NH_PW, c = c0 + pos % NH_PW;
            int g = (r < NH_MAPH && c < NH_MAPW) ? (int)gi[r * NH_MAPW + c] : NH_PAD_GLYPH;
            acc += to_float(T[(int64_t)g * NH_TROW + pos * NH_P1 + k]);
        }
        acc = fmaxf(acc, 0.0f);
        t16s[i] = acc;
        t16_save[(int64_t)b * (NH_TOK * NH_P1) + i] = from_float(acc);
        if (k < 2)
            dxy_save[((int64_t)b * NH_TOK + tk) * 2 + k] = from_float(k == 0 ? dx : dy);
    }
    __syncthreads();
    for (int o = threadIdx.x; o < NH_GLB_HID; o += blockDim.x) {
        float best = -1e30f;
        int bm = 0;
        for (int tk = 0; tk < NH_TOK; tk++) {
            float v = 0.0f;
            for (int k = 0; k < NH_P1; k++)
                v += w2s[o * NH_P1 + k] * t16s[tk * NH_P1 + k];
            if (v > best) { best = v; bm = tk; }
        }
        glb_out[(int64_t)b * NH_GLB_HID + o] = from_float(fmaxf(best + to_float(b2[o]), 0.0f));
        argmax[(int64_t)b * NH_GLB_HID + o] = bm;
    }
}

// Backward through max + layer 2, one block per sample. dglb is already
// relu-masked (and b2's grad accumulated) by nh_relu_bias_bwd. dW2 and dt16
// accumulate in fixed-point smem (deterministic), dt16 is relu-masked against
// the saved t16 and written back over it.
__global__ void nh_patch_max_bwd_kernel(
    precision_t* __restrict__ t16_io, long long* __restrict__ dw2_acc,
    const precision_t* __restrict__ dglb, const precision_t* __restrict__ w2,
    const int* __restrict__ argmax, int B) {
    __shared__ float w2s[NH_GLB_HID * NH_P1];
    __shared__ float t16s[NH_TOK * NH_P1];
    __shared__ long long dt16s[NH_TOK * NH_P1];
    __shared__ long long dw2s[NH_GLB_HID * NH_P1];
    int b = blockIdx.x;
    if (b >= B) return;
    for (int i = threadIdx.x; i < NH_GLB_HID * NH_P1; i += blockDim.x) {
        w2s[i] = to_float(w2[i]);
        dw2s[i] = 0;
    }
    for (int i = threadIdx.x; i < NH_TOK * NH_P1; i += blockDim.x) {
        t16s[i] = to_float(t16_io[(int64_t)b * (NH_TOK * NH_P1) + i]);
        dt16s[i] = 0;
    }
    __syncthreads();
    for (int o = threadIdx.x; o < NH_GLB_HID; o += blockDim.x) {
        float g = to_float(dglb[(int64_t)b * NH_GLB_HID + o]);
        if (g == 0.0f) continue;
        int m = argmax[(int64_t)b * NH_GLB_HID + o];
        for (int k = 0; k < NH_P1; k++) {
            float dt = g * w2s[o * NH_P1 + k];
            if (dt != 0.0f)
                atomicAdd((unsigned long long*)&dt16s[m * NH_P1 + k],
                          (unsigned long long)(long long)__float2ll_rn(dt * NH_FXP));
            float dw = g * t16s[m * NH_P1 + k];
            if (dw != 0.0f)
                atomicAdd((unsigned long long*)&dw2s[o * NH_P1 + k],
                          (unsigned long long)(long long)__float2ll_rn(dw * NH_FXP));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NH_GLB_HID * NH_P1; i += blockDim.x)
        if (dw2s[i] != 0)
            atomicAdd((unsigned long long*)&dw2_acc[i], (unsigned long long)dw2s[i]);
    for (int i = threadIdx.x; i < NH_TOK * NH_P1; i += blockDim.x) {
        float v = t16s[i] > 0.0f ? nh_fxp_to_float(dt16s[i]) : 0.0f;
        t16_io[(int64_t)b * (NH_TOK * NH_P1) + i] = from_float(v);
    }
}

// Decode int32 LE blstats + extra stats and expand to NH_BL_FEAT normalized
// features. Warp per sample: one serial thread per sample is a 200-op latency
// chain. Layout: 25 scalars | 7 hunger | 13 cond | cooldown | 24 prev-act |
// 18 inv counts.
__global__ void nh_blstats_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs, int B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / 32, lane = tid % 32;
    if (b >= B) return;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
    const precision_t* ex = src + 4 * NH_BL_RAW;
    precision_t* dst = out + (int64_t)b * NH_BL_FEAT;
    for (int j = lane; j < NH_BL_FEAT; j += 32) {
        float f;
        if (j < 25) {
            int i = j + (j >= 21) + (j >= 24);   // skip hunger(21), condition(25)
            int v = nh_bl_read_i32(src + 4*i);
            f = NH_BL_ISLOG[i] ? log1pf(fmaxf((float)v, 0.0f)) * NH_BL_SCALE[i]
                               : (float)v * NH_BL_SCALE[i];
        } else if (j < 32) {
            int v = nh_bl_read_i32(src + 4*NH_BL_HUNGER);
            f = (j - 25 == max(0, min(v, 6))) ? 1.0f : 0.0f;
        } else if (j < 45) {
            unsigned int cond = (unsigned int)nh_bl_read_i32(src + 4*NH_BL_CONDITION);
            f = (float)((cond >> (j - 32)) & 1u);
        } else if (j == 45) {
            f = log1pf(fmaxf((float)nh_bl_read_i32(ex), 0.0f)) * 0.1f;
        } else if (j < 46 + NH_ACTIONS) {
            f = (j - 46 == nh_bl_read_i32(ex + 4)) ? 1.0f : 0.0f;
        } else {
            f = (float)nh_bl_read_i32(ex + 4*(j - 44 - NH_ACTIONS)) * 0.125f;
        }
        dst[j] = from_float(f);
    }
}

// concat = [local hid | global hid | bl hid | bl raw feats]
__global__ void nh_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ loc,
    const precision_t* __restrict__ glb, const precision_t* __restrict__ inv,
    const precision_t* __restrict__ bl_out, const precision_t* __restrict__ bl_feats, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_CONCAT) return;
    int b = idx / NH_CONCAT, c = idx % NH_CONCAT;
    precision_t val;
    if (c < NH_LOC_HID)
        val = loc[(int64_t)b * NH_LOC_HID + c];
    else if (c < NH_LOC_HID + NH_GLB_HID)
        val = glb[(int64_t)b * NH_GLB_HID + (c - NH_LOC_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INV_FLAT)
        val = inv[(int64_t)b * NH_INV_FLAT + (c - NH_LOC_HID - NH_GLB_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INV_FLAT + NH_BL_HID)
        val = bl_out[(int64_t)b * NH_BL_HID + (c - NH_LOC_HID - NH_GLB_HID - NH_INV_FLAT)];
    else
        val = bl_feats[(int64_t)b * NH_BL_FEAT + (c - NH_LOC_HID - NH_GLB_HID - NH_INV_FLAT - NH_BL_HID)];
    out[idx] = val;
}

// Copy a per-sample slice [offset, offset+n) of a (B, stride) tensor into (B, n).
__global__ void nh_slice_kernel(
    precision_t* __restrict__ dst, const precision_t* __restrict__ src,
    int B, int stride, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n) return;
    dst[idx] = src[(idx / n) * stride + offset + idx % n];
}

// ---- inventory entity branch ----

__global__ void nh_inv_decode_kernel(
    float* __restrict__ idx, const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV) return;
    int b = t / NH_INV, s = t % NH_INV;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + NH_INV_OFF + 2 * s;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));
}

// T_inv[g,k] = dot(E[g,:], inv1_w[k,:]) — 5977x16, sequential inner loop
__global__ void nh_inv_table_kernel(precision_t* __restrict__ T,
    const precision_t* __restrict__ E, const precision_t* __restrict__ w1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_GLYPH_VOCAB * NH_INV_HID) return;
    int g = i / NH_INV_HID, k = i % NH_INV_HID;
    float acc = 0.0f;
    for (int d = 0; d < NH_EMBED_DIM; d++)
        acc += to_float(E[g * NH_EMBED_DIM + d]) * to_float(w1[k * NH_EMBED_DIM + d]);
    T[i] = from_float(acc);
}

__global__ void nh_inv_gather_kernel(precision_t* __restrict__ out,
    const precision_t* __restrict__ T, const precision_t* __restrict__ b1,
    const float* __restrict__ idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV_FLAT) return;
    int b = t / NH_INV_FLAT, r = t % NH_INV_FLAT;
    int s = r / NH_INV_HID, k = r % NH_INV_HID;
    int g = (int)idx[b * NH_INV + s];
    float v = to_float(T[g * NH_INV_HID + k]) + to_float(b1[k]);
    out[t] = from_float(v > 0.0f ? v : 0.0f);
}

// dT_inv scatter, plain global fxp atomics: 55x16 per sample is too small
// for the hot-row machinery to pay
__global__ void nh_dTinv_scatter_kernel(long long* __restrict__ dT,
    const precision_t* __restrict__ dflat, const float* __restrict__ idx, int64_t n) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float v = to_float(dflat[t]);
    if (v == 0.0f) return;
    int g = (int)idx[t / NH_INV_HID];
    nh_fxp_atomic_add(&dT[(int64_t)g * NH_INV_HID + t % NH_INV_HID], v);
}

__global__ void nh_add_inplace_kernel(precision_t* __restrict__ dst,
    const precision_t* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_float(to_float(dst[i]) + to_float(src[i]));
}

// Fused relu backward + bias grad: masks grad in place against out and
// accumulates the per-column sum into fixed-point bias_acc. Launch via
// nh_colsum_grid so (gridDim*blockDim) % dim == 0: each thread's column is
// then fixed across its grid-stride, so the sum lives in one register (fixed
// order -> deterministic) and costs one quantize + global atomic.
__global__ void nh_relu_bias_bwd_kernel(
    precision_t* __restrict__ grad, const precision_t* __restrict__ out,
    long long* __restrict__ bias_acc, int64_t total, int dim) {
    extern __shared__ long long sdata[];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) sdata[j] = 0;
    __syncthreads();
    int64_t i0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int64_t i = i0; i < total; i += stride) {
        // branch-free: a divergent store-vs-load branch here runs ~3x slower
        float g = to_float(out[i]) > 0.0f ? to_float(grad[i]) : 0.0f;
        grad[i] = from_float(g);
        acc += g;
    }
    if (acc != 0.0f) nh_fxp_atomic_add(&sdata[(int)(i0 % dim)], acc);
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        if (sdata[j] != 0) atomicAdd((unsigned long long*)&bias_acc[j], (unsigned long long)sdata[j]);
}

static inline int nh_colsum_grid(int64_t total, int dim) {
    int64_t g = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (g > 1024) g = 1024;
    while ((g * BLOCK_SIZE) % dim) g++;
    return (int)g;
}

// Cast the packed fixed-point bias accumulators to their grad tensors in one launch.
__global__ void nh_bias_flush_kernel(
    const long long* __restrict__ acc,
    precision_t* __restrict__ d0, int n0, precision_t* __restrict__ d1, int n1,
    precision_t* __restrict__ d2, int n2, precision_t* __restrict__ d3, int n3,
    precision_t* __restrict__ d4, int n4, precision_t* __restrict__ d5, int n5) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n0 + n1 + n2 + n3 + n4 + n5 ? nh_fxp_to_float(acc[i]) : 0.0f;
    if (i < n0) d0[i] = from_float(v);
    else if ((i -= n0) < n1) d1[i] = from_float(v);
    else if ((i -= n1) < n2) d2[i] = from_float(v);
    else if ((i -= n2) < n3) d3[i] = from_float(v);
    else if ((i -= n3) < n4) d4[i] = from_float(v);
    else if ((i -= n4) < n5) d5[i] = from_float(v);
}

// ---- embedding backward ----
// Both views are linear in the embeddings, so dE is a scatter-add of per-cell
// 32-dim grad vectors into glyph rows. The dominant glyphs (unexplored stone,
// floor, walls cover ~80% of cells) contend on the same rows; per-block smem
// accumulators for the top-NH_HOT_T glyphs absorb that, the cold tail goes
// straight to global atomics. Every element quantizes exactly once.

// Per-minibatch glyph histogram (per-block smem: hot counters would otherwise
// serialize global atomics).
__global__ void nh_hist_kernel(int* __restrict__ counts, const float* __restrict__ idx, int N) {
    __shared__ int hist[NH_GLYPH_VOCAB];
    for (int i = threadIdx.x; i < NH_GLYPH_VOCAB; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    int chunk = (N + gridDim.x - 1) / gridDim.x;
    int start = blockIdx.x * chunk, end = min(start + chunk, N);
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
        atomicAdd(&hist[(int)idx[i]], 1);
    __syncthreads();
    for (int g = threadIdx.x; g < NH_GLYPH_VOCAB; g += blockDim.x)
        if (hist[g]) atomicAdd(&counts[g], hist[g]);
}

// Top-K glyphs by count (single block; counts are consumed).
// hot_map must be pre-set to -1.
__global__ void nh_hot_select_kernel(
    int* __restrict__ hot_map, int* __restrict__ hot_list, int* __restrict__ hot_n,
    int* __restrict__ counts, int K) {
    __shared__ int best_v[1024], best_g[1024];
    int tid = threadIdx.x;
    for (int k = 0; k < K; k++) {
        int bv = 0, bg = -1;
        for (int g = tid; g < NH_GLYPH_VOCAB; g += blockDim.x)
            if (counts[g] > bv) { bv = counts[g]; bg = g; }
        best_v[tid] = bv; best_g[tid] = bg;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off && best_v[tid + off] > best_v[tid]) {
                best_v[tid] = best_v[tid + off]; best_g[tid] = best_g[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0 && best_g[0] >= 0) {
            hot_map[best_g[0]] = k;
            hot_list[k] = best_g[0];
            counts[best_g[0]] = 0;
            *hot_n = k + 1;
        }
        __syncthreads();
    }
}

// Local-view dE scatter: add each crop cell's 32-dim grad vector (contiguous
// in dvec) into its glyph's embed-grad row.
__global__ void nh_dE_scatter_kernel(
    long long* __restrict__ dE_i, const precision_t* __restrict__ dvec,
    const float* __restrict__ gidx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n,
    int64_t ncell) {
    extern __shared__ long long acc_s[];   // NH_HOT_T x NH_EMBED_DIM
    for (int i = threadIdx.x; i < NH_HOT_T * NH_EMBED_DIM; i += blockDim.x)
        acc_s[i] = 0;
    __syncthreads();
    int64_t total = ncell * NH_EMBED_DIM;
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (int64_t)gridDim.x * blockDim.x) {
        float v = to_float(dvec[t]);
        if (v == 0.0f) continue;
        unsigned long long q = (unsigned long long)(long long)__float2ll_rn(v * NH_FXP);
        if (q == 0) continue;
        int d = t % NH_EMBED_DIM;
        int g = (int)gidx[t / NH_EMBED_DIM];
        int slot = hot_map[g];
        if (slot >= 0) atomicAdd((unsigned long long*)&acc_s[slot * NH_EMBED_DIM + d], q);
        else atomicAdd((unsigned long long*)&dE_i[(int64_t)g * NH_EMBED_DIM + d], q);
    }
    __syncthreads();
    int n = *hot_n;
    for (int i = threadIdx.x; i < n * NH_EMBED_DIM; i += blockDim.x) {
        long long v = acc_s[i];
        if (v != 0)
            atomicAdd((unsigned long long*)&dE_i[(int64_t)hot_list[i / NH_EMBED_DIM] * NH_EMBED_DIM + i % NH_EMBED_DIM],
                      (unsigned long long)v);
    }
}

// Global-view dT scatter: dT[g, pos*P1+k] += dt16[b, tk*P1+k] for every
// (token, position) occurrence of glyph g. One thread per (b, tk, k) element,
// quantized once; the 32 positions then add the same integer (hot glyphs via
// smem, cold straight to global — same scheme as the old conv1 dT scatter).
__global__ void nh_dT_patch_scatter_kernel(
    long long* __restrict__ dT_i, const precision_t* __restrict__ dt16,
    const float* __restrict__ idx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n, int B) {
    extern __shared__ long long acc_s[];   // NH_HOT_G x NH_TROW
    for (int i = threadIdx.x; i < NH_HOT_G * NH_TROW; i += blockDim.x)
        acc_s[i] = 0;
    __syncthreads();
    int64_t total = (int64_t)B * NH_TOK * NH_P1;
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (int64_t)gridDim.x * blockDim.x) {
        float g = to_float(dt16[t]);
        if (g == 0.0f) continue;
        unsigned long long q = (unsigned long long)(long long)__float2ll_rn(g * NH_FXP);
        if (q == 0) continue;
        int k = t % NH_P1;
        int tk = (t / NH_P1) % NH_TOK;
        int64_t b = t / (NH_P1 * NH_TOK);
        int r0 = (tk / NH_PX) * NH_PH, c0 = (tk % NH_PX) * NH_PW;
        const float* gi = idx + b * NH_MGRID;
        #pragma unroll
        for (int pos = 0; pos < NH_PCELLS; pos++) {
            int r = r0 + pos / NH_PW, c = c0 + pos % NH_PW;
            int gl = (r < NH_MAPH && c < NH_MAPW) ? (int)gi[r * NH_MAPW + c] : NH_PAD_GLYPH;
            int slot = hot_map[gl];
            if (slot >= 0) atomicAdd((unsigned long long*)&acc_s[slot * NH_TROW + pos * NH_P1 + k], q);
            else atomicAdd((unsigned long long*)&dT_i[(int64_t)gl * NH_TROW + pos * NH_P1 + k], q);
        }
    }
    __syncthreads();
    int n = *hot_n;
    for (int i = threadIdx.x; i < n * NH_TROW; i += blockDim.x) {
        long long v = acc_s[i];
        if (v != 0)
            atomicAdd((unsigned long long*)&dT_i[(int64_t)hot_list[i / NH_TROW] * NH_TROW + i % NH_TROW],
                      (unsigned long long)v);
    }
}

// Plain per-column sum (b1's grad from dt16 rows); same launch contract as
// nh_relu_bias_bwd (dt16 is already relu-masked there, so no second mask).
__global__ void nh_col_sum_kernel(
    long long* __restrict__ bias_acc, const precision_t* __restrict__ src,
    int64_t total, int dim) {
    extern __shared__ long long sdata[];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) sdata[j] = 0;
    __syncthreads();
    int64_t i0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int64_t i = i0; i < total; i += stride) acc += to_float(src[i]);
    if (acc != 0.0f) nh_fxp_atomic_add(&sdata[(int)(i0 % dim)], acc);
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        if (sdata[j] != 0) atomicAdd((unsigned long long*)&bias_acc[j], (unsigned long long)sdata[j]);
}

// += variant of the row-sparse cast: adds the local view's embed grads on top
// of the global view's GEMM-produced dE (same guard + re-zero contract).
__global__ void nh_fxp_add_rows_kernel(
    precision_t* __restrict__ dst, long long* __restrict__ src,
    const int* __restrict__ counts, const int* __restrict__ hot_map,
    int trow, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = idx / trow;
    if (counts[row] == 0 && hot_map[row] < 0) return;
    dst[idx] = from_float(to_float(dst[idx]) + nh_fxp_to_float(src[idx]));
    src[idx] = 0;
}

// Seed the grid-glyph histogram with the static pad-glyph count (edge-patch
// positions past the map read the pad row; the hist over glyph_idx can't see
// them, but the rows-cast guard and hot selection must).
__global__ void nh_count_pad_kernel(int* __restrict__ counts, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        counts[NH_PAD_GLYPH] += NH_PAD_PER_SAMPLE * B;
}

// ---- Nethack encoder structs ----

struct NethackEncoderWeights {
    PrecisionTensor embed_w, loc_w, loc_b;
    PrecisionTensor glb1_w, glb1_xy, glb1_b, glb2_w, glb2_b;
    PrecisionTensor inv1_w, inv1_b;
    PrecisionTensor bl_w, bl_b, proj_w, proj_b;
    int obs_size, hidden;
};

struct NethackEncoderActivations {
    FloatTensor glyph_idx, crop_glyph;     // decoded grid + crop glyph ids
    PrecisionTensor x_local;               // crop embeds (grad aliases it)
    PrecisionTensor w_perm, glyph_T;       // fused embed+flatten+glb1 table
    PrecisionTensor t16;                   // relu'd patch tokens (dt16 overwrites)
    PrecisionTensor dxy;                   // per-token hero offsets (w_xy wgrad)
    IntTensor tok_argmax;                  // winning token per (sample, out dim)
    FloatTensor inv_idx;                   // inventory slot glyph ids
    PrecisionTensor inv_T, inv_out;        // fused inv table + relu'd flat slots
    PrecisionTensor loc_out, glb_out;
    PrecisionTensor bl_feats, bl_out;
    PrecisionTensor concat, out;
    PrecisionTensor loc_grad, glb_grad, inv_grad, bl_grad;   // contiguous concat slices
    PrecisionTensor dT, dw_perm;           // dT table + permuted glb1 wgrad
    PrecisionTensor dTinv, dE_tmp;         // inv-table grad + its dE staging
    LongTensor dT_i, dTinv_i;              // fixed-point dT scatter staging
    LongTensor dE_i;                       // fixed-point local embed-grad staging
    LongTensor dw2_acc;                    // fixed-point glb2 wgrad staging
    LongTensor bias_acc;                   // fixed-point bias grads: proj | loc | glb2 | bl | glb1 | inv1
    IntTensor sort_local, sort_grid;       // counts | hot_map | hot_list | hot_n
    PrecisionTensor embed_wgrad, loc_wgrad, loc_bgrad;
    PrecisionTensor glb1_wgrad, glb1_xygrad, glb1_bgrad, glb2_wgrad, glb2_bgrad;
    PrecisionTensor inv1_wgrad, inv1_bgrad;
    PrecisionTensor bl_wgrad, bl_bgrad, proj_wgrad, proj_bgrad;
};

static NethackEncoderWeights* nethack_encoder_create(int obs_size, int hidden) {
    if (obs_size != NH_OBS_SIZE) {
        fprintf(stderr, "nethack encoder: obs size %d != expected %d "
            "(env obs layout out of sync with src/nethack.cu?)\n",
            obs_size, NH_OBS_SIZE);
        exit(1);
    }
    NethackEncoderWeights* ew = (NethackEncoderWeights*)calloc(1, sizeof(NethackEncoderWeights));
    ew->obs_size = obs_size; ew->hidden = hidden;
    return ew;
}

// ---- Nethack encoder interface ----

static PrecisionTensor nethack_encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = input.shape[0];

    nh_decode_kernel<<<grid_size(B * NH_MGRID), BLOCK_SIZE, 0, stream>>>(
        a->glyph_idx.data, input.data, B);
    nh_crop_kernel<<<grid_size(B * NH_CGRID), BLOCK_SIZE, 0, stream>>>(
        a->crop_glyph.data, a->glyph_idx.data, input.data, B);

    nh_local_gather_kernel<<<grid_size(B * NH_LOC_IN), BLOCK_SIZE, 0, stream>>>(
        a->x_local.data, ew->embed_w.data, a->crop_glyph.data, B);
    puf_mm(&a->x_local, &ew->loc_w, &a->loc_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LOC_HID), BLOCK_SIZE, 0, stream>>>(
        a->loc_out.data, ew->loc_b.data, B * NH_LOC_HID, NH_LOC_HID);

    nh_permute_g1_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->w_perm.data, ew->glb1_w.data);
    puf_mm(&ew->embed_w, &a->w_perm, &a->glyph_T, stream);
    nh_patch_max_kernel<<<B, 128, 0, stream>>>(
        a->glb_out.data, a->t16.data, a->dxy.data, a->tok_argmax.data,
        a->glyph_T.data, ew->glb1_b.data, ew->glb1_xy.data, ew->glb2_w.data,
        ew->glb2_b.data, a->glyph_idx.data, input.data, B);

    nh_inv_decode_kernel<<<grid_size(B * NH_INV), BLOCK_SIZE, 0, stream>>>(
        a->inv_idx.data, input.data, B);
    nh_inv_table_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->inv_T.data, ew->embed_w.data, ew->inv1_w.data);
    nh_inv_gather_kernel<<<grid_size(B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->inv_out.data, a->inv_T.data, ew->inv1_b.data, a->inv_idx.data, B);

    nh_blstats_kernel<<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(
        a->bl_feats.data, input.data, B);
    puf_mm(&a->bl_feats, &ew->bl_w, &a->bl_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_out.data, ew->bl_b.data, B * NH_BL_HID, NH_BL_HID);

    nh_concat_kernel<<<grid_size(B * NH_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->loc_out.data, a->glb_out.data, a->inv_out.data,
        a->bl_out.data, a->bl_feats.data, B);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    nh_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nethack_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    // fixed-point bias-grad accumulators: [proj H | loc 256 | glb2 128 | bl 64 | glb1 16 | inv1 16]
    long long* bacc = (long long*)a->bias_acc.data;
    cudaMemsetAsync(bacc, 0, (H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID) * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * H, H), BLOCK_SIZE, H * sizeof(long long), stream>>>(
        grad.data, a->out.data, bacc, (int64_t)B * H, H);
    puf_mm_tn_splitk(&grad, &a->concat, &a->proj_wgrad, stream);  // tall-K, tiny output

    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, NH_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    // Local view: wgrad against saved x_local, then the input grad overwrites
    // x_local in place before scattering into the embed table.
    nh_slice_kernel<<<grid_size(B * NH_LOC_HID), BLOCK_SIZE, 0, stream>>>(
        a->loc_grad.data, grad_concat.data, B, NH_CONCAT, 0, NH_LOC_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LOC_HID, NH_LOC_HID), BLOCK_SIZE, NH_LOC_HID * sizeof(long long), stream>>>(
        a->loc_grad.data, a->loc_out.data, bacc + H, (int64_t)B * NH_LOC_HID, NH_LOC_HID);
    PrecisionTensor locg = {.data = a->loc_grad.data, .shape = {B, NH_LOC_HID}};
    puf_mm_tn(&locg, &a->x_local, &a->loc_wgrad, stream);
    PrecisionTensor dx_local = {.data = a->x_local.data, .shape = {B, NH_LOC_IN}};
    puf_mm_nn(&locg, &ew->loc_w, &dx_local, stream);

    // Global view: relu mask + b2 grad, then the fused max backward (dW2 via
    // fixed-point staging, dt16 overwrites t16), then b1's column sum.
    nh_slice_kernel<<<grid_size(B * NH_GLB_HID), BLOCK_SIZE, 0, stream>>>(
        a->glb_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID, NH_GLB_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_GLB_HID, NH_GLB_HID), BLOCK_SIZE, NH_GLB_HID * sizeof(long long), stream>>>(
        a->glb_grad.data, a->glb_out.data, bacc + H + NH_LOC_HID, (int64_t)B * NH_GLB_HID, NH_GLB_HID);
    cudaMemsetAsync(a->dw2_acc.data, 0, NH_GLB_HID * NH_P1 * sizeof(long long), stream);
    nh_patch_max_bwd_kernel<<<B, 128, 0, stream>>>(
        a->t16.data, (long long*)a->dw2_acc.data, a->glb_grad.data,
        ew->glb2_w.data, a->tok_argmax.data, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLB_HID * NH_P1), BLOCK_SIZE, 0, stream>>>(
        a->glb2_wgrad.data, (long long*)a->dw2_acc.data, NH_GLB_HID * NH_P1);
    nh_col_sum_kernel<<<nh_colsum_grid((int64_t)B * NH_TOK * NH_P1, NH_P1), BLOCK_SIZE, NH_P1 * sizeof(long long), stream>>>(
        bacc + H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID, a->t16.data, (int64_t)B * NH_TOK * NH_P1, NH_P1);
    // (dx,dy) weight slice: dW_xy = dt16^T @ dxy (tall-K, 16x2 output)
    PrecisionTensor dt16v = {.data = a->t16.data, .shape = {B * NH_TOK, NH_P1}};
    PrecisionTensor dxyv  = {.data = a->dxy.data, .shape = {B * NH_TOK, 2}};
    puf_mm_tn_splitk(&dt16v, &dxyv, &a->glb1_xygrad, stream);

    // Inventory branch: relu mask + inv1 bias, dT_inv scatter by slot glyph,
    // then dinv1_w = dT_inv^T @ E; the dE contribution is added at the end.
    nh_slice_kernel<<<grid_size(B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->inv_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID + NH_GLB_HID, NH_INV_FLAT);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_INV_FLAT, NH_INV_HID), BLOCK_SIZE, NH_INV_HID * sizeof(long long), stream>>>(
        a->inv_grad.data, a->inv_out.data, bacc + H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1,
        (int64_t)B * NH_INV_FLAT, NH_INV_HID);
    cudaMemsetAsync(a->dTinv_i.data, 0, (size_t)NH_GLYPH_VOCAB * NH_INV_HID * sizeof(long long), stream);
    nh_dTinv_scatter_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dTinv_i.data, a->inv_grad.data, a->inv_idx.data, (int64_t)B * NH_INV_FLAT);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->dTinv.data, (long long*)a->dTinv_i.data, NH_GLYPH_VOCAB * NH_INV_HID);
    puf_mm_tn_splitk(&a->dTinv, &ew->embed_w, &a->inv1_wgrad, stream);

    // Blstats branch (raw-feature slice of concat has no upstream params)
    nh_slice_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID + NH_GLB_HID + NH_INV_FLAT, NH_BL_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_BL_HID, NH_BL_HID), BLOCK_SIZE, NH_BL_HID * sizeof(long long), stream>>>(
        a->bl_grad.data, a->bl_out.data, bacc + H + NH_LOC_HID + NH_GLB_HID, (int64_t)B * NH_BL_HID, NH_BL_HID);
    PrecisionTensor blg = {.data = a->bl_grad.data, .shape = {B, NH_BL_HID}};
    puf_mm_tn(&blg, &a->bl_feats, &a->bl_wgrad, stream);

    // Global branch to the embed table + glb1: scatter dt16 occurrences into
    // dT, then dE = dT @ W' and dW' = dT^T @ E (the fused-table backward).
    int* counts_g = a->sort_grid.data;
    int* hot_map_g = counts_g + NH_GLYPH_VOCAB;
    int* hot_list_g = hot_map_g + NH_GLYPH_VOCAB;
    int* hot_n_g = hot_list_g + NH_HOT_G;
    cudaMemsetAsync(counts_g, 0, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_map_g, 0xFF, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_n_g, 0, sizeof(int), stream);
    nh_hist_kernel<<<NH_SORT_BLOCKS, 256, 0, stream>>>(counts_g, a->glyph_idx.data, B * NH_MGRID);
    nh_count_pad_kernel<<<1, 1, 0, stream>>>(counts_g, B);
    nh_hot_select_kernel<<<1, 1024, 0, stream>>>(hot_map_g, hot_list_g, hot_n_g, counts_g, NH_HOT_G);
    nh_dT_patch_scatter_kernel<<<128, 1024, NH_HOT_G * NH_TROW * sizeof(long long), stream>>>(
        (long long*)a->dT_i.data, a->t16.data, a->glyph_idx.data,
        hot_map_g, hot_list_g, hot_n_g, B);
    int dT_n = NH_GLYPH_VOCAB * NH_TROW;
    nh_fxp_to_precision_rows_kernel<<<grid_size(dT_n), BLOCK_SIZE, 0, stream>>>(
        a->dT.data, (long long*)a->dT_i.data, counts_g, hot_map_g, NH_TROW, dT_n);
    puf_mm_nn(&a->dT, &a->w_perm, &a->embed_wgrad, stream);   // dE  = dT @ W'
    puf_mm_tn(&a->dT, &ew->embed_w, &a->dw_perm, stream);     // dW' = dT^T @ E
    nh_unpermute_g1_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->glb1_wgrad.data, a->dw_perm.data);

    // Local branch adds its embed grads on top of the GEMM-produced dE.
    int* counts_l = a->sort_local.data;
    int* hot_map_l = counts_l + NH_GLYPH_VOCAB;
    int* hot_list_l = hot_map_l + NH_GLYPH_VOCAB;
    int* hot_n_l = hot_list_l + NH_HOT_T;
    cudaMemsetAsync(counts_l, 0, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_map_l, 0xFF, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_n_l, 0, sizeof(int), stream);
    nh_hist_kernel<<<NH_SORT_BLOCKS, 256, 0, stream>>>(counts_l, a->crop_glyph.data, B * NH_CGRID);
    nh_hot_select_kernel<<<1, 1024, 0, stream>>>(hot_map_l, hot_list_l, hot_n_l, counts_l, NH_HOT_T);
    nh_dE_scatter_kernel<<<1024, BLOCK_SIZE, NH_HOT_T * NH_EMBED_DIM * sizeof(long long), stream>>>(
        (long long*)a->dE_i.data, dx_local.data, a->crop_glyph.data,
        hot_map_l, hot_list_l, hot_n_l, (int64_t)B * NH_CGRID);
    int dE_n = NH_GLYPH_VOCAB * NH_EMBED_DIM;
    nh_fxp_add_rows_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, (long long*)a->dE_i.data, counts_l, hot_map_l, NH_EMBED_DIM, dE_n);

    // Inventory branch adds its embed grads last: dE += dT_inv @ inv1_w.
    puf_mm_nn(&a->dTinv, &ew->inv1_w, &a->dE_tmp, stream);
    nh_add_inplace_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->dE_tmp.data, dE_n);

    nh_bias_flush_kernel<<<grid_size(H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        bacc, a->proj_bgrad.data, H, a->loc_bgrad.data, NH_LOC_HID,
        a->glb2_bgrad.data, NH_GLB_HID, a->bl_bgrad.data, NH_BL_HID,
        a->glb1_bgrad.data, NH_P1, a->inv1_bgrad.data, NH_INV_HID);
}

static void nethack_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->loc_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->loc_b.data, 0, numel(ew->loc_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->glb1_w, 1.0f, (*seed)++, stream);
    // zero: kaiming's fan_in=2 would run 20x hotter than the glyph slice
    cudaMemsetAsync(ew->glb1_xy.data, 0, numel(ew->glb1_xy.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->glb1_b.data, 0, numel(ew->glb1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->glb2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->glb2_b.data, 0, numel(ew->glb2_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->inv1_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->inv1_b.data, 0, numel(ew->inv1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->bl_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->bl_b.data, 0, numel(ew->bl_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->proj_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->proj_b.data, 0, numel(ew->proj_b.shape) * sizeof(precision_t), stream);
}

// Param and grad registration orders must match pairwise (muon walks both flat).
static void nethack_encoder_reg_params(void* w, Allocator* alloc) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    ew->embed_w = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    ew->loc_w   = {.shape = {NH_LOC_HID, NH_LOC_IN}};
    ew->loc_b   = {.shape = {NH_LOC_HID}};
    ew->glb1_w  = {.shape = {NH_P1, NH_PCELLS * NH_EMBED_DIM}};
    ew->glb1_xy = {.shape = {NH_P1, 2}};
    ew->glb1_b  = {.shape = {NH_P1}};
    ew->glb2_w  = {.shape = {NH_GLB_HID, NH_P1}};
    ew->glb2_b  = {.shape = {NH_GLB_HID}};
    ew->inv1_w  = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    ew->inv1_b  = {.shape = {NH_INV_HID}};
    ew->bl_w    = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    ew->bl_b    = {.shape = {NH_BL_HID}};
    ew->proj_w  = {.shape = {ew->hidden, NH_CONCAT}};
    ew->proj_b  = {.shape = {ew->hidden}};
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->loc_w);   alloc_register(alloc,&ew->loc_b);
    alloc_register(alloc,&ew->glb1_w);  alloc_register(alloc,&ew->glb1_xy);
    alloc_register(alloc,&ew->glb1_b);
    alloc_register(alloc,&ew->glb2_w);  alloc_register(alloc,&ew->glb2_b);
    alloc_register(alloc,&ew->inv1_w);  alloc_register(alloc,&ew->inv1_b);
    alloc_register(alloc,&ew->bl_w);    alloc_register(alloc,&ew->bl_b);
    alloc_register(alloc,&ew->proj_w);  alloc_register(alloc,&ew->proj_b);
}

static void nethack_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    *a = {};
    a->glyph_idx  = {.shape = {B_TT, NH_MGRID}};
    a->crop_glyph = {.shape = {B_TT, NH_CGRID}};
    a->x_local    = {.shape = {B_TT, NH_LOC_IN}};
    a->w_perm     = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->glyph_T    = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->t16        = {.shape = {B_TT, NH_TOK * NH_P1}};
    a->dxy        = {.shape = {B_TT, NH_TOK * 2}};
    a->tok_argmax = {.shape = {B_TT, NH_GLB_HID}};
    a->inv_idx    = {.shape = {B_TT, NH_INV}};
    a->inv_T      = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->inv_out    = {.shape = {B_TT, NH_INV_FLAT}};
    a->loc_out    = {.shape = {B_TT, NH_LOC_HID}};
    a->glb_out    = {.shape = {B_TT, NH_GLB_HID}};
    a->bl_feats   = {.shape = {B_TT, NH_BL_FEAT}};
    a->bl_out     = {.shape = {B_TT, NH_BL_HID}};
    a->concat     = {.shape = {B_TT, NH_CONCAT}};
    a->out        = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts,&a->glyph_idx); alloc_register(acts,&a->crop_glyph);
    alloc_register(acts,&a->x_local);
    alloc_register(acts,&a->w_perm);    alloc_register(acts,&a->glyph_T);
    alloc_register(acts,&a->t16);       alloc_register(acts,&a->dxy);
    alloc_register(acts,&a->tok_argmax);
    alloc_register(acts,&a->inv_idx);   alloc_register(acts,&a->inv_T);
    alloc_register(acts,&a->inv_out);
    alloc_register(acts,&a->loc_out);   alloc_register(acts,&a->glb_out);
    alloc_register(acts,&a->bl_feats);  alloc_register(acts,&a->bl_out);
    alloc_register(acts,&a->concat);    alloc_register(acts,&a->out);
    a->loc_grad   = {.shape = {B_TT, NH_LOC_HID}};
    a->glb_grad   = {.shape = {B_TT, NH_GLB_HID}};
    a->inv_grad   = {.shape = {B_TT, NH_INV_FLAT}};
    a->bl_grad    = {.shape = {B_TT, NH_BL_HID}};
    a->dT         = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dT_i       = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dTinv      = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->dTinv_i    = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->dE_tmp     = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->dw_perm    = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->dE_i       = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->dw2_acc    = {.shape = {NH_GLB_HID * NH_P1}};
    a->sort_local = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_T + 1}};
    a->sort_grid  = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_G + 1}};
    a->bias_acc   = {.shape = {ew->hidden + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID}};
    alloc_register(acts,&a->loc_grad);  alloc_register(acts,&a->glb_grad);
    alloc_register(acts,&a->inv_grad);  alloc_register(acts,&a->bl_grad);
    alloc_register(acts,&a->dT);        alloc_register(acts,&a->dT_i);
    alloc_register(acts,&a->dTinv);     alloc_register(acts,&a->dTinv_i);
    alloc_register(acts,&a->dE_tmp);
    alloc_register(acts,&a->dw_perm);   alloc_register(acts,&a->dE_i);
    alloc_register(acts,&a->dw2_acc);
    alloc_register(acts,&a->sort_local); alloc_register(acts,&a->sort_grid);
    alloc_register(acts,&a->bias_acc);
    a->embed_wgrad = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->loc_wgrad   = {.shape = {NH_LOC_HID, NH_LOC_IN}};
    a->loc_bgrad   = {.shape = {NH_LOC_HID}};
    a->glb1_wgrad  = {.shape = {NH_P1, NH_PCELLS * NH_EMBED_DIM}};
    a->glb1_xygrad = {.shape = {NH_P1, 2}};
    a->glb1_bgrad  = {.shape = {NH_P1}};
    a->glb2_wgrad  = {.shape = {NH_GLB_HID, NH_P1}};
    a->glb2_bgrad  = {.shape = {NH_GLB_HID}};
    a->inv1_wgrad  = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    a->inv1_bgrad  = {.shape = {NH_INV_HID}};
    a->bl_wgrad    = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    a->bl_bgrad    = {.shape = {NH_BL_HID}};
    a->proj_wgrad  = {.shape = {ew->hidden, NH_CONCAT}};
    a->proj_bgrad  = {.shape = {ew->hidden}};
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(grads,&a->loc_wgrad);   alloc_register(grads,&a->loc_bgrad);
    alloc_register(grads,&a->glb1_wgrad);  alloc_register(grads,&a->glb1_xygrad);
    alloc_register(grads,&a->glb1_bgrad);
    alloc_register(grads,&a->glb2_wgrad);  alloc_register(grads,&a->glb2_bgrad);
    alloc_register(grads,&a->inv1_wgrad);  alloc_register(grads,&a->inv1_bgrad);
    alloc_register(grads,&a->bl_wgrad);    alloc_register(grads,&a->bl_bgrad);
    alloc_register(grads,&a->proj_wgrad);  alloc_register(grads,&a->proj_bgrad);
}

static void nethack_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    a->glyph_idx  = {.shape = {B, NH_MGRID}};
    a->crop_glyph = {.shape = {B, NH_CGRID}};
    a->x_local    = {.shape = {B, NH_LOC_IN}};
    a->w_perm     = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->glyph_T    = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->t16        = {.shape = {B, NH_TOK * NH_P1}};
    a->dxy        = {.shape = {B, NH_TOK * 2}};
    a->tok_argmax = {.shape = {B, NH_GLB_HID}};
    a->inv_idx    = {.shape = {B, NH_INV}};
    a->inv_T      = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->inv_out    = {.shape = {B, NH_INV_FLAT}};
    a->loc_out    = {.shape = {B, NH_LOC_HID}};
    a->glb_out    = {.shape = {B, NH_GLB_HID}};
    a->bl_feats   = {.shape = {B, NH_BL_FEAT}};
    a->bl_out     = {.shape = {B, NH_BL_HID}};
    a->concat     = {.shape = {B, NH_CONCAT}};
    a->out        = {.shape = {B, ew->hidden}};
    alloc_register(alloc,&a->glyph_idx); alloc_register(alloc,&a->crop_glyph);
    alloc_register(alloc,&a->x_local);
    alloc_register(alloc,&a->w_perm);    alloc_register(alloc,&a->glyph_T);
    alloc_register(alloc,&a->t16);       alloc_register(alloc,&a->dxy);
    alloc_register(alloc,&a->tok_argmax);
    alloc_register(alloc,&a->inv_idx);   alloc_register(alloc,&a->inv_T);
    alloc_register(alloc,&a->inv_out);
    alloc_register(alloc,&a->loc_out);   alloc_register(alloc,&a->glb_out);
    alloc_register(alloc,&a->bl_feats);  alloc_register(alloc,&a->bl_out);
    alloc_register(alloc,&a->concat);    alloc_register(alloc,&a->out);
}

static void* nethack_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return nethack_encoder_create(e->in_dim, e->out_dim);
}
static void nethack_encoder_free_weights(void* weights) { free(weights); }
static void nethack_encoder_free_activations(void* activations) { free(activations); }

static void create_nethack_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = nethack_encoder_forward,
        .backward = nethack_encoder_backward,
        .init_weights = nethack_encoder_init_weights,
        .reg_params = nethack_encoder_reg_params,
        .reg_train = nethack_encoder_reg_train,
        .reg_rollout = nethack_encoder_reg_rollout,
        .create_weights = nethack_encoder_create_weights,
        .free_weights = nethack_encoder_free_weights,
        .free_activations = nethack_encoder_free_activations,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(NethackEncoderActivations),
    };
}
