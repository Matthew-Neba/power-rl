// NetHack CUDA encoder: fused glyph-table conv1, cuDNN conv2, blstats MLP.
// Included by ocean.cu — requires kernels.cu, models.cu, cudnn_conv2d.cu.
// Bit-deterministic backward: fixed-point integer atomics, deterministic
// cuDNN algos, and split-K GEMMs with atomic reduction schemes masked out.

#include <cublasLt.h>

static cudnnDataType_t nh_cudnn_dtype() {
    return (PRECISION_SIZE == 2) ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
}

// thread_local like cublas_get_handle: buffer threads capture rollout graphs
// concurrently, and cudnnSetStream on a shared handle would race.
static cudnnHandle_t nh_cudnn_handle() {
    static thread_local cudnnHandle_t h = nullptr;
    if (!h) CHECK_CUDNN(cudnnCreate(&h));
    return h;
}

// Algo selection via v7 heuristics (no timing benchmark — same choice every
// run) filtered to CUDNN_DETERMINISTIC execution. cudnnFind* picks by
// wall-clock and can select atomics-based algos: both break run-to-run
// bit reproducibility.
static cudnnConvolutionFwdAlgo_t cudnn_det_fwd_algo(cudnnHandle_t h,
        cudnnTensorDescriptor_t in, cudnnFilterDescriptor_t filt,
        cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t out) {
    cudnnConvolutionFwdAlgoPerf_t p[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int n = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(h, in, filt, conv, out,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &n, p));
    for (int i = 0; i < n; i++)
        if (p[i].status == CUDNN_STATUS_SUCCESS && p[i].determinism == CUDNN_DETERMINISTIC)
            return p[i].algo;
    fprintf(stderr, "cuDNN: no deterministic fwd conv algo\n"); exit(1);
}
static cudnnConvolutionBwdFilterAlgo_t cudnn_det_wgrad_algo(cudnnHandle_t h,
        cudnnTensorDescriptor_t in, cudnnTensorDescriptor_t out,
        cudnnConvolutionDescriptor_t conv, cudnnFilterDescriptor_t filt) {
    cudnnConvolutionBwdFilterAlgoPerf_t p[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    int n = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(h, in, out, conv, filt,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &n, p));
    for (int i = 0; i < n; i++)
        if (p[i].status == CUDNN_STATUS_SUCCESS && p[i].determinism == CUDNN_DETERMINISTIC)
            return p[i].algo;
    fprintf(stderr, "cuDNN: no deterministic wgrad conv algo\n"); exit(1);
}
static cudnnConvolutionBwdDataAlgo_t cudnn_det_dgrad_algo(cudnnHandle_t h,
        cudnnFilterDescriptor_t filt, cudnnTensorDescriptor_t out,
        cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t in) {
    cudnnConvolutionBwdDataAlgoPerf_t p[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
    int n = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(h, filt, out, conv, in,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &n, p));
    for (int i = 0; i < n; i++)
        if (p[i].status == CUDNN_STATUS_SUCCESS && p[i].determinism == CUDNN_DETERMINISTIC)
            return p[i].algo;
    fprintf(stderr, "cuDNN: no deterministic dgrad conv algo\n"); exit(1);
}

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
// Obs layout (must match ocean/nethack/nethack.h defaults):
//   [0, 2*NH_GRID)  glyphs, int16 LE, NETHACK_GLYPH_CROP^2 egocentric window
//   [2*NH_GRID, +4*NH_BL_RAW)  blstats, int32 LE
//   [+4*NH_BL_RAW, +4*NH_EX_RAW)  extra stats, int32 LE: prayer cooldown,
//                                 previous action, 18 inventory class counts

static constexpr int NH_MAP = 21;                  // NETHACK_GLYPH_CROP
static constexpr int NH_GRID = NH_MAP * NH_MAP;
static constexpr int NH_GLYPH_VOCAB = 5977;        // MAX_GLYPH + 1 (NetHack 3.6.6)
static constexpr int NH_EMBED_DIM = 32;
static constexpr int NH_BL_RAW = 27;               // NLE_BLSTATS_SIZE
static constexpr int NH_BL_HUNGER = 21, NH_BL_CONDITION = 25;
static constexpr int NH_ACTIONS = 24;              // NETHACK_NUM_ACTIONS
static constexpr int NH_OCLASSES = 18;             // MAXOCLASSES
static constexpr int NH_EX_RAW = 2 + NH_OCLASSES;  // NETHACK_EXTRA_INTS
// scalars + hunger onehot + condition bits + cooldown + prev-action onehot + counts
static constexpr int NH_BL_FEAT = 25 + 7 + 13 + 1 + NH_ACTIONS + NH_OCLASSES;
static constexpr int NH_BL_HID = 64;
static constexpr int NH_OBS_SIZE = NH_GRID * 2 + (NH_BL_RAW + NH_EX_RAW) * 4;
static constexpr int NH_C1_IC = NH_EMBED_DIM, NH_C1_OC = 64, NH_C1_K = 5, NH_C1_S = 3;
static constexpr int NH_C1_OH = 6, NH_C1_OW = 6;
static constexpr int NH_C2_IC = 64, NH_C2_OC = 64, NH_C2_K = 3, NH_C2_S = 1;
static constexpr int NH_C2_OH = 4, NH_C2_OW = 4;
static constexpr int NH_CONV_FLAT = NH_C2_OC * NH_C2_OH * NH_C2_OW;
static constexpr int NH_CONCAT = NH_CONV_FLAT + NH_BL_HID + NH_BL_FEAT;
// Fused embed+conv1 table: T[g, tap*OC+oc] = sum_d E[g,d] * W[oc, d, tap].
// Embedding and conv1 are both linear, so their composition is a per-glyph
// lookup table rebuilt with one small GEMM whenever the weights change.
// The 19MB table stays L2-resident; glyph ids become leaves (no input grad).
static constexpr int NH_TAPS = NH_C1_K * NH_C1_K;
static constexpr int NH_TROW = NH_TAPS * NH_C1_OC;
static constexpr int NH_C1_SP = NH_C1_OH * NH_C1_OW;
static constexpr int NH_C2_SP = NH_C2_OH * NH_C2_OW;
static constexpr int NH_C2_KK = NH_C2_IC * NH_C2_K * NH_C2_K;
static constexpr int NH_SORT_BLOCKS = 256;               // hist grid (smem histograms)
static constexpr int NH_HOT_T = 7;                       // hot-glyph dT smem slots (7x1600 int64 = 89.6KB dynamic smem, opt-in)

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

// Row-sparse variant for glyph-table grads: a minibatch touches only a few
// hundred of the 5977 rows (counts>0 or hot). Untouched rows skip the int64
// read; touched rows are re-zeroed in place, replacing a full-table memset.
// Invariant: src is all-zero between iterations (alloc_create zeroes it once).
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

// Decode int16 LE glyph ids into an fp32 index buffer.
__global__ void nh_decode_kernel(
    float* __restrict__ idx, const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_GRID) return;
    int b = t / NH_GRID, cell = t % NH_GRID;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + 2 * cell;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));
}

// conv1.w (OC, D*TAPS) -> W' (TAPS*OC, D), so T = E @ W'^T lands as
// T[g, tap*OC+oc] and the gather reads coalesce over oc.
__global__ void nh_permute_w_kernel(
    precision_t* __restrict__ wp, const precision_t* __restrict__ w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int t = r / NH_C1_OC, oc = r % NH_C1_OC;
    wp[i] = w[oc * (NH_EMBED_DIM * NH_TAPS) + d * NH_TAPS + t];
}

// Fused embed+conv1: out rows layout (B*36, OC),
// out[(b*36+p), oc] = relu(bias[oc] + sum_taps T[g_tap, tap*OC+oc])
__global__ void nh_fused_conv1_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ T,
    const precision_t* __restrict__ bias, const float* __restrict__ idx, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * NH_C1_SP * NH_C1_OC) return;
    int oc = i % NH_C1_OC;
    int p  = (i / NH_C1_OC) % NH_C1_SP;
    int b  = i / (NH_C1_OC * NH_C1_SP);
    int oh = p / NH_C1_OW, ow = p % NH_C1_OW;
    const float* gi = idx + (int64_t)b * NH_GRID;
    float acc = to_float(bias[oc]);
    #pragma unroll
    for (int t = 0; t < NH_TAPS; t++) {
        int cell = (NH_C1_S * oh + t / NH_C1_K) * NH_MAP + NH_C1_S * ow + t % NH_C1_K;
        acc += to_float(T[(int64_t)(int)gi[cell] * NH_TROW + t * NH_C1_OC + oc]);
    }
    out[((int64_t)b * NH_C1_SP + p) * NH_C1_OC + oc] = from_float(fmaxf(acc, 0.0f));
}

// conv2 via cuDNN NHWC implicit GEMM. The rows layout IS NHWC, so the tensors
// feed cuDNN directly; only the filter is repacked (nh_w2_tmajor's t-major
// (64, 9*64) [oc][t*IC+ic] is exactly cuDNN's NHWC/KRSC filter layout). The
// previous im2col+cuBLAS path materialized 151MB col buffers per direction at
// train minibatch and its backward-data GEMM ran at ~15% of memory bandwidth.
// Bias is added in concat.

// dout rows (B*16, 64) from grad_concat's conv slice (c = oc*16+p).
__global__ void nh_dout_rows_kernel(
    precision_t* __restrict__ dst, const precision_t* __restrict__ grad_concat, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * NH_C2_SP * NH_C2_OC) return;
    int oc = i % NH_C2_OC;
    int p = (i / NH_C2_OC) % NH_C2_SP;
    int b = i / (NH_C2_OC * NH_C2_SP);
    dst[i] = grad_concat[(int64_t)b * NH_CONCAT + oc * NH_C2_SP + p];
}

// conv2.w (OC, IC*3*3) [oc][ic*9+t] -> KRSC (OC,KH,KW,IC) [oc][t*IC+ic],
// cuDNN's NHWC filter layout.
__global__ void nh_w2_tmajor_kernel(
    precision_t* __restrict__ wp, const precision_t* __restrict__ w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_C2_OC * NH_C2_KK) return;
    int oc = i / NH_C2_KK, c = i % NH_C2_KK;
    int t = c / NH_C2_IC, ic = c % NH_C2_IC;
    wp[i] = w[oc * NH_C2_KK + ic * (NH_C2_K * NH_C2_K) + t];
}

// Inverse: cuDNN's KRSC filter grad back to the (OC, IC*3*3) param layout.
__global__ void nh_wgrad_from_krsc_kernel(
    precision_t* __restrict__ wg, const precision_t* __restrict__ krsc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_C2_OC * NH_C2_KK) return;
    int oc = i / NH_C2_KK, c = i % NH_C2_KK;
    int ic = c / (NH_C2_K * NH_C2_K), t = c % (NH_C2_K * NH_C2_K);
    wg[i] = krsc[oc * NH_C2_KK + t * NH_C2_IC + ic];
}

// Fused relu backward + bias grad: masks grad in place against out and
// accumulates the per-column sum into fixed-point bias_acc. Launch via
// nh_colsum_grid so (gridDim*blockDim) % dim == 0: each thread's column is
// then fixed across its grid-stride, so the sum lives in one register (fixed
// order -> deterministic) and costs one quantize + global atomic. Replaces a
// strided-column bias_grad_kernel pass (~10x slower at these row counts)
// plus a separate relu_backward pass over the same tensor.
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

// Plain per-column sum (conv2 bias grad from dout2 rows); same launch contract.
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
    precision_t* __restrict__ d2, int n2, precision_t* __restrict__ d3, int n3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n0 + n1 + n2 + n3 ? nh_fxp_to_float(acc[i]) : 0.0f;
    if (i < n0) d0[i] = from_float(v);
    else if ((i -= n0) < n1) d1[i] = from_float(v);
    else if ((i -= n1) < n2) d2[i] = from_float(v);
    else if ((i -= n2) < n3) d3[i] = from_float(v);
}

// Decode int32 LE blstats + extra stats and expand to NH_BL_FEAT normalized
// features: 25 scaled scalars, 7 hunger one-hot, 13 condition bits, prayer
// cooldown (log), prev-action one-hot (-1 at episode start = all zeros),
// 18 inventory class counts.
__global__ void nh_blstats_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + 2 * NH_GRID;
    precision_t* dst = out + (int64_t)b * NH_BL_FEAT;
    int j = 0, hunger = 0;
    unsigned int cond = 0;
    for (int i = 0; i < NH_BL_RAW; i++) {
        unsigned int u = (unsigned int)(int)to_float(src[4*i])
                       | ((unsigned int)(int)to_float(src[4*i + 1]) << 8)
                       | ((unsigned int)(int)to_float(src[4*i + 2]) << 16)
                       | ((unsigned int)(int)to_float(src[4*i + 3]) << 24);
        int v = (int)u;
        if (i == NH_BL_HUNGER) { hunger = max(0, min(v, 6)); continue; }
        if (i == NH_BL_CONDITION) { cond = u; continue; }
        float f = NH_BL_ISLOG[i] ? log1pf(fmaxf((float)v, 0.0f)) * NH_BL_SCALE[i]
                                 : (float)v * NH_BL_SCALE[i];
        dst[j++] = from_float(f);
    }
    for (int h = 0; h < 7; h++) dst[j++] = from_float(h == hunger ? 1.0f : 0.0f);
    for (int k = 0; k < 13; k++) dst[j++] = from_float((float)((cond >> k) & 1u));
    const precision_t* ex = src + 4 * NH_BL_RAW;
    for (int i = 0; i < NH_EX_RAW; i++) {
        unsigned int u = (unsigned int)(int)to_float(ex[4*i])
                       | ((unsigned int)(int)to_float(ex[4*i + 1]) << 8)
                       | ((unsigned int)(int)to_float(ex[4*i + 2]) << 16)
                       | ((unsigned int)(int)to_float(ex[4*i + 3]) << 24);
        int v = (int)u;
        if (i == 0) dst[j++] = from_float(log1pf(fmaxf((float)v, 0.0f)) * 0.1f);
        else if (i == 1)
            for (int h = 0; h < NH_ACTIONS; h++) dst[j++] = from_float(h == v ? 1.0f : 0.0f);
        else dst[j++] = from_float((float)v * 0.125f);
    }
}

// concat = [conv2 flat + bias | bl hidden | bl raw feats]. conv2.out is rows
// (B*16, 64), a raw GEMM result; the bias lands here. The concat keeps the
// original NCHW feature order (c = oc*16+p) so existing checkpoints' proj_w
// columns stay valid.
__global__ void nh_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ conv_rows,
    const precision_t* __restrict__ conv_bias,
    const precision_t* __restrict__ bl_out, const precision_t* __restrict__ bl_feats, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_CONCAT) return;
    int b = idx / NH_CONCAT, c = idx % NH_CONCAT;
    precision_t val;
    if (c < NH_CONV_FLAT)
        val = from_float(
            to_float(conv_rows[b * NH_CONV_FLAT + (c % NH_C2_SP) * NH_C2_OC + c / NH_C2_SP])
            + to_float(conv_bias[c / NH_C2_SP]));
    else if (c < NH_CONV_FLAT + NH_BL_HID)
        val = bl_out[b * NH_BL_HID + (c - NH_CONV_FLAT)];
    else
        val = bl_feats[b * NH_BL_FEAT + (c - NH_CONV_FLAT - NH_BL_HID)];
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

// ---- conv1+embedding backward via the fused dT table ----
// dT[g, t*OC+oc] accumulates conv1's output grad for every (window, tap)
// occurrence of glyph g; dE and dW1 then follow as two tiny GEMMs on dT
// (dE = dT @ W', dW' = dT^T @ E). The scatter's atomic contention on the few
// dominant glyphs (pad/floor/wall cover ~80% of cells) is absorbed by
// per-block smem accumulators for the top-NH_HOT_T glyphs of the minibatch.

__global__ void nh_unpermute_w_kernel(
    precision_t* __restrict__ wg, const precision_t* __restrict__ wpg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int t = r / NH_C1_OC, oc = r % NH_C1_OC;
    wg[oc * (NH_EMBED_DIM * NH_TAPS) + d * NH_TAPS + t] = wpg[i];
}

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

// Top-NH_HOT_T glyphs by count (single block; counts are consumed).
// hot_map must be pre-set to -1.
__global__ void nh_hot_select_kernel(
    int* __restrict__ hot_map, int* __restrict__ hot_list, int* __restrict__ hot_n,
    int* __restrict__ counts) {
    __shared__ int best_v[1024], best_g[1024];
    int tid = threadIdx.x;
    for (int k = 0; k < NH_HOT_T; k++) {
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

// Scatter conv1's output grad (rows layout) into fixed-point dT. Hot glyphs
// accumulate in smem (consecutive oc lanes -> conflict-free) and flush once
// per block; the cold tail goes straight to global atomics. Each grad element
// is quantized once; the taps then add the same integer everywhere.
__global__ void nh_dT_scatter_kernel(
    long long* __restrict__ dT_i, const precision_t* __restrict__ grad,
    const float* __restrict__ idx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n, int B) {
    extern __shared__ long long acc_s[];   // NH_HOT_T x NH_TROW
    for (int i = threadIdx.x; i < NH_HOT_T * NH_TROW; i += blockDim.x)
        acc_s[i] = 0;
    __syncthreads();
    int64_t total = (int64_t)B * NH_C1_SP * NH_C1_OC;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        int oc = i % NH_C1_OC;
        int p  = (i / NH_C1_OC) % NH_C1_SP;
        int64_t b = i / (NH_C1_OC * NH_C1_SP);
        float g = to_float(grad[i]);
        if (g == 0.0f) continue;  // relu-masked
        unsigned long long q = (unsigned long long)(long long)__float2ll_rn(g * NH_FXP);
        if (q == 0) continue;
        int oh = p / NH_C1_OW, ow = p % NH_C1_OW;
        const float* gi = idx + b * NH_GRID;
        #pragma unroll
        for (int t = 0; t < NH_TAPS; t++) {
            int cell = (NH_C1_S * oh + t / NH_C1_K) * NH_MAP + NH_C1_S * ow + t % NH_C1_K;
            int gl = (int)gi[cell];
            int slot = hot_map[gl];
            if (slot >= 0) atomicAdd((unsigned long long*)&acc_s[slot * NH_TROW + t * NH_C1_OC + oc], q);
            else atomicAdd((unsigned long long*)&dT_i[(int64_t)gl * NH_TROW + t * NH_C1_OC + oc], q);
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

// ---- Nethack encoder structs ----

struct NethackEncoderWeights {
    ConvWeights conv1, conv2;
    PrecisionTensor embed_w, bl_w, bl_b, proj_w, proj_b;
    int obs_size, hidden;
    // Batch-independent cuDNN state for conv2 (filter is KRSC = w2p)
    cudnnFilterDescriptor_t c2_filt;
    cudnnConvolutionDescriptor_t c2_conv;
    bool c2_cudnn_ready;
};

struct NethackEncoderActivations {
    ConvActivations conv1, conv2;
    PrecisionTensor w_perm, glyph_T;       // fused embed+conv1 table (T = E @ W'^T)
    PrecisionTensor dout2, w2p;            // conv2 dout rows + KRSC filter
    PrecisionTensor wgrad_krsc;            // cuDNN filter grad before repack
    PrecisionTensor dT, dw_perm;           // dT table + permuted conv1 wgrad
    LongTensor dT_i;                       // fixed-point dT scatter staging
    LongTensor bias_acc;                   // fixed-point bias grads: proj | conv1 | bl | conv2
    IntTensor sort_buf;                    // counts | hot_map | hot_list | hot_n
    FloatTensor glyph_idx;                 // decoded glyph ids
    PrecisionTensor bl_feats, bl_out, bl_grad, concat, out;
    PrecisionTensor embed_wgrad, bl_wgrad, bl_bgrad, proj_wgrad, proj_bgrad;
    // Per-batch-size cuDNN state for conv2 (NHWC == rows layout)
    cudnnTensorDescriptor_t c2_in, c2_out;
    cudnnConvolutionFwdAlgo_t c2_fwd_algo;
    cudnnConvolutionBwdDataAlgo_t c2_dgrad_algo;
    cudnnConvolutionBwdFilterAlgo_t c2_wgrad_algo;
    void *c2_fwd_ws, *c2_dgrad_ws, *c2_wgrad_ws;
    size_t c2_fwd_ws_n, c2_dgrad_ws_n, c2_wgrad_ws_n;
};

// Descriptors + algo selection + workspace for conv2 at batch size B.
// Called from reg_train/reg_rollout (outside graph capture); the algo find
// runs real kernels, so fixed algos + preallocated workspaces keep the actual
// forward/backward capture-safe.
static void nh_conv2_cudnn_setup(NethackEncoderWeights* ew, NethackEncoderActivations* a, int B, bool bwd) {
    cudnnDataType_t dt = nh_cudnn_dtype();
    cudnnHandle_t h = nh_cudnn_handle();
    if (!ew->c2_cudnn_ready) {
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&ew->c2_filt));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(ew->c2_filt, dt, CUDNN_TENSOR_NHWC,
            NH_C2_OC, NH_C2_IC, NH_C2_K, NH_C2_K));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&ew->c2_conv));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(ew->c2_conv, 0, 0, NH_C2_S, NH_C2_S, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        // bf16: tensor cores. fp32 (gradient-check builds): FMA only — the
        // default would silently use TF32 and fail finite-difference checks.
        CHECK_CUDNN(cudnnSetConvolutionMathType(ew->c2_conv,
            PRECISION_SIZE == 2 ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH));
        ew->c2_cudnn_ready = true;
    }
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&a->c2_in));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(a->c2_in, CUDNN_TENSOR_NHWC, dt, B, NH_C2_IC, NH_C1_OH, NH_C1_OW));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&a->c2_out));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(a->c2_out, CUDNN_TENSOR_NHWC, dt, B, NH_C2_OC, NH_C2_OH, NH_C2_OW));

    a->c2_fwd_algo = cudnn_det_fwd_algo(h, a->c2_in, ew->c2_filt, ew->c2_conv, a->c2_out);
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(h, a->c2_in, ew->c2_filt, ew->c2_conv, a->c2_out,
        a->c2_fwd_algo, &a->c2_fwd_ws_n));
    if (a->c2_fwd_ws_n) cudaMalloc(&a->c2_fwd_ws, a->c2_fwd_ws_n);
    if (!bwd) return;

    a->c2_wgrad_algo = cudnn_det_wgrad_algo(h, a->c2_in, a->c2_out, ew->c2_conv, ew->c2_filt);
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(h, a->c2_in, a->c2_out, ew->c2_conv, ew->c2_filt,
        a->c2_wgrad_algo, &a->c2_wgrad_ws_n));
    if (a->c2_wgrad_ws_n) cudaMalloc(&a->c2_wgrad_ws, a->c2_wgrad_ws_n);

    a->c2_dgrad_algo = cudnn_det_dgrad_algo(h, ew->c2_filt, a->c2_out, ew->c2_conv, a->c2_in);
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(h, ew->c2_filt, a->c2_out, ew->c2_conv, a->c2_in,
        a->c2_dgrad_algo, &a->c2_dgrad_ws_n));
    if (a->c2_dgrad_ws_n) cudaMalloc(&a->c2_dgrad_ws, a->c2_dgrad_ws_n);
}

static NethackEncoderWeights* nethack_encoder_create(int obs_size, int hidden) {
    if (obs_size != NH_OBS_SIZE) {
        fprintf(stderr, "nethack encoder: obs size %d != expected %d "
            "(env built with different NETHACK_USE_*/NETHACK_GLYPH_CROP flags?)\n",
            obs_size, NH_OBS_SIZE);
        exit(1);
    }
    NethackEncoderWeights* ew = (NethackEncoderWeights*)calloc(1, sizeof(NethackEncoderWeights));
    ew->obs_size = obs_size; ew->hidden = hidden;
    conv_init(&ew->conv1, NH_C1_IC, NH_C1_OC, NH_C1_K, NH_C1_S, NH_MAP, NH_MAP, true);
    conv_init(&ew->conv2, NH_C2_IC, NH_C2_OC, NH_C2_K, NH_C2_S, NH_C1_OH, NH_C1_OW, false);
    return ew;
}

// ---- Nethack encoder interface ----

static PrecisionTensor nethack_encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = input.shape[0];

    nh_decode_kernel<<<grid_size(B * NH_GRID), BLOCK_SIZE, 0, stream>>>(
        a->glyph_idx.data, input.data, B);
    nh_permute_w_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->w_perm.data, ew->conv1.w.data);
    puf_mm(&ew->embed_w, &a->w_perm, &a->glyph_T, stream);
    nh_fused_conv1_kernel<<<grid_size(B * NH_C1_SP * NH_C1_OC), BLOCK_SIZE, 0, stream>>>(
        a->conv1.out.data, a->glyph_T.data, ew->conv1.b.data, a->glyph_idx.data, B);
    nh_w2_tmajor_kernel<<<grid_size(NH_C2_OC * NH_C2_KK), BLOCK_SIZE, 0, stream>>>(
        a->w2p.data, ew->conv2.w.data);
    cudnnHandle_t cudnn = nh_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(cudnn, stream));
    float c_alpha = 1.0f, c_beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn,          // bias added in concat
        &c_alpha, a->c2_in, a->conv1.out.data, ew->c2_filt, a->w2p.data,
        ew->c2_conv, a->c2_fwd_algo, a->c2_fwd_ws, a->c2_fwd_ws_n,
        &c_beta, a->c2_out, a->conv2.out.data));

    nh_blstats_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->bl_feats.data, input.data, B);
    puf_mm(&a->bl_feats, &ew->bl_w, &a->bl_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_out.data, ew->bl_b.data, B * NH_BL_HID, NH_BL_HID);

    nh_concat_kernel<<<grid_size(B * NH_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2.out.data, ew->conv2.b.data, a->bl_out.data, a->bl_feats.data, B);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    nh_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nethack_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    // fixed-point bias-grad accumulators: [proj H | conv1 64 | bl 64 | conv2 64]
    long long* bacc = (long long*)a->bias_acc.data;
    cudaMemsetAsync(bacc, 0, (H + 3 * NH_C1_OC) * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * H, H), BLOCK_SIZE, H * sizeof(long long), stream>>>(
        grad.data, a->out.data, bacc, (int64_t)B * H, H);
    puf_mm_tn_splitk(&grad, &a->concat, &a->proj_wgrad, stream);  // tall-K, tiny output

    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, NH_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    // Conv branch. conv2 grads via cuDNN NHWC: forward's conv1.out is the
    // saved input for wgrad; forward's w2p (KRSC) is the filter for dgrad.
    nh_dout_rows_kernel<<<grid_size(B * NH_C2_SP * NH_C2_OC), BLOCK_SIZE, 0, stream>>>(
        a->dout2.data, grad_concat.data, B);
    nh_col_sum_kernel<<<nh_colsum_grid((int64_t)B * NH_C2_SP * NH_C2_OC, NH_C2_OC), BLOCK_SIZE, NH_C2_OC * sizeof(long long), stream>>>(
        bacc + H + 2 * NH_C1_OC, a->dout2.data, (int64_t)B * NH_C2_SP * NH_C2_OC, NH_C2_OC);
    cudnnHandle_t cudnn = nh_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(cudnn, stream));
    float c_alpha = 1.0f, c_beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn,
        &c_alpha, a->c2_in, a->conv1.out.data, a->c2_out, a->dout2.data,
        ew->c2_conv, a->c2_wgrad_algo, a->c2_wgrad_ws, a->c2_wgrad_ws_n,
        &c_beta, ew->c2_filt, a->wgrad_krsc.data));
    nh_wgrad_from_krsc_kernel<<<grid_size(NH_C2_OC * NH_C2_KK), BLOCK_SIZE, 0, stream>>>(
        a->conv2.wgrad.data, a->wgrad_krsc.data);
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn,
        &c_alpha, ew->c2_filt, a->w2p.data, a->c2_out, a->dout2.data,
        ew->c2_conv, a->c2_dgrad_algo, a->c2_dgrad_ws, a->c2_dgrad_ws_n,
        &c_beta, a->c2_in, a->conv1.grad.data));
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_C1_SP * NH_C1_OC, NH_C1_OC), BLOCK_SIZE, NH_C1_OC * sizeof(long long), stream>>>(
        a->conv1.grad.data, a->conv1.out.data, bacc + H,
        (int64_t)B * NH_C1_SP * NH_C1_OC, NH_C1_OC);

    // conv1+embedding backward via dT: see kernel block comment.
    int N = B * NH_GRID;
    int* counts = a->sort_buf.data;
    int* hot_map = counts + NH_GLYPH_VOCAB;
    int* hot_list = hot_map + NH_GLYPH_VOCAB;
    int* hot_n = hot_list + NH_HOT_T;
    cudaMemsetAsync(counts, 0, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_map, 0xFF, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_n, 0, sizeof(int), stream);
    nh_hist_kernel<<<NH_SORT_BLOCKS, 256, 0, stream>>>(counts, a->glyph_idx.data, N);
    nh_hot_select_kernel<<<1, 1024, 0, stream>>>(hot_map, hot_list, hot_n, counts);
    int dT_n = NH_GLYPH_VOCAB * NH_TROW;
    nh_dT_scatter_kernel<<<1024, 256, NH_HOT_T * NH_TROW * sizeof(long long), stream>>>(
        (long long*)a->dT_i.data, a->conv1.grad.data, a->glyph_idx.data, hot_map, hot_list, hot_n, B);
    nh_fxp_to_precision_rows_kernel<<<grid_size(dT_n), BLOCK_SIZE, 0, stream>>>(
        a->dT.data, (long long*)a->dT_i.data, counts, hot_map, NH_TROW, dT_n);
    puf_mm_nn(&a->dT, &a->w_perm, &a->embed_wgrad, stream);   // dE  = dT @ W'
    puf_mm_tn(&a->dT, &ew->embed_w, &a->dw_perm, stream);     // dW' = dT^T @ E
    nh_unpermute_w_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->conv1.wgrad.data, a->dw_perm.data);

    // Blstats branch (raw-feature slice of concat has no upstream params)
    nh_slice_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, grad_concat.data, B, NH_CONCAT, NH_CONV_FLAT, NH_BL_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_BL_HID, NH_BL_HID), BLOCK_SIZE, NH_BL_HID * sizeof(float), stream>>>(
        a->bl_grad.data, a->bl_out.data, bacc + H + NH_C1_OC, (int64_t)B * NH_BL_HID, NH_BL_HID);
    PrecisionTensor blg = {.data = a->bl_grad.data, .shape = {B, NH_BL_HID}};
    puf_mm_tn(&blg, &a->bl_feats, &a->bl_wgrad, stream);

    nh_bias_flush_kernel<<<grid_size(H + 3 * NH_C1_OC), BLOCK_SIZE, 0, stream>>>(
        bacc, a->proj_bgrad.data, H, a->conv1.bgrad.data, NH_C1_OC,
        a->bl_bgrad.data, NH_BL_HID, a->conv2.bgrad.data, NH_C2_OC);
}

static void nethack_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    conv_init_weights(&ew->conv1, seed, stream);
    conv_init_weights(&ew->conv2, seed, stream);
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->bl_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->bl_b.data, 0, numel(ew->bl_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->proj_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->proj_b.data, 0, numel(ew->proj_b.shape) * sizeof(precision_t), stream);
}

// Param and grad registration orders must match pairwise (muon walks both flat).
static void nethack_encoder_reg_params(void* w, Allocator* alloc) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    conv_reg_params(&ew->conv1, alloc);
    conv_reg_params(&ew->conv2, alloc);
    ew->embed_w = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    ew->bl_w    = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    ew->bl_b    = {.shape = {NH_BL_HID}};
    ew->proj_w  = {.shape = {ew->hidden, NH_CONCAT}};
    ew->proj_b  = {.shape = {ew->hidden}};
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->bl_w);   alloc_register(alloc,&ew->bl_b);
    alloc_register(alloc,&ew->proj_w); alloc_register(alloc,&ew->proj_b);
}

static void nethack_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    *a = {};
    // Conv1 (fused with the embedding via the glyph_T table — no im2col)
    a->conv1.out   = {.shape = {B_TT * NH_C1_OC * NH_C1_OH * NH_C1_OW}};
    a->conv1.grad  = {.shape = {B_TT * NH_C1_OC * NH_C1_OH * NH_C1_OW}};
    a->conv1.wgrad = {.shape = {NH_C1_OC, NH_C1_IC * NH_C1_K * NH_C1_K}};
    a->conv1.bgrad = {.shape = {NH_C1_OC}};
    alloc_register(acts,&a->conv1.out); alloc_register(acts,&a->conv1.grad);
    alloc_register(grads,&a->conv1.wgrad); alloc_register(grads,&a->conv1.bgrad);
    a->w_perm  = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->glyph_T = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dT      = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dT_i    = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dw_perm = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->sort_buf = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_T + 1}};
    a->bias_acc = {.shape = {ew->hidden + 3 * NH_C1_OC}};
    alloc_register(acts,&a->w_perm); alloc_register(acts,&a->glyph_T);
    alloc_register(acts,&a->dT);     alloc_register(acts,&a->dT_i);
    alloc_register(acts,&a->dw_perm); alloc_register(acts,&a->sort_buf);
    alloc_register(acts,&a->bias_acc);
    // 89.6KB dynamic smem (int64 hot slots) needs the opt-in cap raise
    if (cudaFuncSetAttribute(nh_dT_scatter_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            NH_HOT_T * NH_TROW * (int)sizeof(long long)) != cudaSuccess) {
        fprintf(stderr, "nethack encoder: GPU lacks %zuB smem for dT scatter\n",
                NH_HOT_T * NH_TROW * sizeof(long long));
        exit(1);
    }
    // Conv2 (cuDNN NHWC)
    a->conv2.out   = {.shape = {B_TT * NH_C2_OC * NH_C2_OH * NH_C2_OW}};
    a->conv2.wgrad = {.shape = {NH_C2_OC, NH_C2_IC * NH_C2_K * NH_C2_K}};
    a->conv2.bgrad = {.shape = {NH_C2_OC}};
    a->dout2       = {.shape = {B_TT * NH_C2_SP, NH_C2_OC}};
    a->w2p         = {.shape = {NH_C2_OC, NH_C2_KK}};
    a->wgrad_krsc  = {.shape = {NH_C2_OC, NH_C2_KK}};
    alloc_register(acts,&a->conv2.out);
    alloc_register(acts,&a->dout2);     alloc_register(acts,&a->w2p);
    alloc_register(acts,&a->wgrad_krsc);
    alloc_register(grads,&a->conv2.wgrad); alloc_register(grads,&a->conv2.bgrad);
    nh_conv2_cudnn_setup(ew, a, B_TT, true);
    a->glyph_idx  = {.shape = {B_TT, NH_GRID}};
    a->bl_feats   = {.shape = {B_TT, NH_BL_FEAT}};
    a->bl_out     = {.shape = {B_TT, NH_BL_HID}};
    a->bl_grad    = {.shape = {B_TT, NH_BL_HID}};
    a->concat     = {.shape = {B_TT, NH_CONCAT}};
    a->out        = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts,&a->glyph_idx);
    alloc_register(acts,&a->bl_feats);  alloc_register(acts,&a->bl_out);
    alloc_register(acts,&a->bl_grad);
    alloc_register(acts,&a->concat);    alloc_register(acts,&a->out);
    a->embed_wgrad   = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->bl_wgrad      = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    a->bl_bgrad      = {.shape = {NH_BL_HID}};
    a->proj_wgrad    = {.shape = {ew->hidden, NH_CONCAT}};
    a->proj_bgrad    = {.shape = {ew->hidden}};
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(grads,&a->bl_wgrad);   alloc_register(grads,&a->bl_bgrad);
    alloc_register(grads,&a->proj_wgrad); alloc_register(grads,&a->proj_bgrad);
}

static void nethack_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    a->glyph_idx = {.shape = {B, NH_GRID}};
    a->w_perm    = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->glyph_T   = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    alloc_register(alloc,&a->glyph_idx);
    alloc_register(alloc,&a->w_perm); alloc_register(alloc,&a->glyph_T);
    a->conv1.out = {.shape = {B * NH_C1_OC * NH_C1_OH * NH_C1_OW}};
    alloc_register(alloc,&a->conv1.out);
    a->conv2.out = {.shape = {B * NH_C2_OC * NH_C2_OH * NH_C2_OW}};
    a->w2p       = {.shape = {NH_C2_OC, NH_C2_KK}};
    alloc_register(alloc,&a->conv2.out); alloc_register(alloc,&a->w2p);
    nh_conv2_cudnn_setup(ew, a, B, false);
    a->bl_feats = {.shape = {B, NH_BL_FEAT}};
    a->bl_out   = {.shape = {B, NH_BL_HID}};
    a->concat   = {.shape = {B, NH_CONCAT}};
    a->out      = {.shape = {B, ew->hidden}};
    alloc_register(alloc,&a->bl_feats); alloc_register(alloc,&a->bl_out);
    alloc_register(alloc,&a->concat);   alloc_register(alloc,&a->out);
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
