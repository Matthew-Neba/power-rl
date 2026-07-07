// Custom CUDA encoders for ocean envs (nmmo3, nethack).
// Included by pufferlib.cu — requires precision_t, PrecisionTensor, Allocator, puf_mm, etc.

#include "cudnn_conv2d.cu"

// ---- NMMO3 constants ----

static constexpr int N3_MAP_H = 11, N3_MAP_W = 15, N3_NFEAT = 10;
static constexpr int N3_MULTIHOT = 59;
static constexpr int N3_MAP_SIZE = N3_MAP_H * N3_MAP_W * N3_NFEAT;
static constexpr int N3_PLAYER = 47, N3_REWARD = 10;
static constexpr int N3_EMBED_DIM = 32, N3_EMBED_VOCAB = 128;
static constexpr int N3_PLAYER_EMBED = N3_PLAYER * N3_EMBED_DIM;
static constexpr int N3_C1_IC = 59, N3_C1_OC = 128, N3_C1_K = 5, N3_C1_S = 3;
static constexpr int N3_C1_OH = 3, N3_C1_OW = 4;
static constexpr int N3_C2_IC = 128, N3_C2_OC = 128, N3_C2_K = 3, N3_C2_S = 1;
static constexpr int N3_C2_OH = 1, N3_C2_OW = 2;
static constexpr int N3_CONV_FLAT = N3_C2_OC * N3_C2_OH * N3_C2_OW;
static constexpr int N3_CONCAT = N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER + N3_REWARD;

__constant__ int N3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

static cudnnDataType_t n3_cudnn_dtype() {
    return (PRECISION_SIZE == 2) ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
}

// ---- NMMO3 kernels ----

__global__ void n3_multihot_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_MAP_H * N3_MAP_W) return;
    int b = idx / (N3_MAP_H * N3_MAP_W), rem = idx % (N3_MAP_H * N3_MAP_W);
    int h = rem / N3_MAP_W, w = rem % N3_MAP_W;
    const precision_t* src = obs + b * obs_size + (h * N3_MAP_W + w) * N3_NFEAT;
    precision_t* dst = out + b * N3_MULTIHOT * N3_MAP_H * N3_MAP_W;
    for (int f = 0; f < N3_NFEAT; f++)
        dst[(N3_OFFSETS[f] + (int)to_float(src[f])) * N3_MAP_H * N3_MAP_W + h * N3_MAP_W + w] = from_float(1.0f);
}

__global__ void n3_embedding_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs,
    const precision_t* __restrict__ embed_w, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_PLAYER) return;
    int b = idx / N3_PLAYER, f = idx % N3_PLAYER;
    int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
    const precision_t* src = embed_w + val * N3_EMBED_DIM;
    precision_t* dst = out + b * N3_PLAYER_EMBED + f * N3_EMBED_DIM;
    for (int d = 0; d < N3_EMBED_DIM; d++) dst[d] = src[d];
}

__global__ void n3_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ conv_flat,
    const precision_t* __restrict__ embed, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONCAT) return;
    int b = idx / N3_CONCAT, c = idx % N3_CONCAT;
    precision_t val;
    if (c < N3_CONV_FLAT) {
        int oc = c / (N3_C2_OH * N3_C2_OW), r = c % (N3_C2_OH * N3_C2_OW);
        int oh = r / N3_C2_OW, ow = r % N3_C2_OW;
        val = conv_flat[b * N3_CONV_FLAT + oc * N3_C2_OH * N3_C2_OW + oh * N3_C2_OW + ow];
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED)
        val = embed[b * N3_PLAYER_EMBED + (c - N3_CONV_FLAT)];
    else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER)
        val = obs[b * obs_size + N3_MAP_SIZE + (c - N3_CONV_FLAT - N3_PLAYER_EMBED)];
    else
        val = obs[b * obs_size + obs_size - N3_REWARD + (c - N3_CONV_FLAT - N3_PLAYER_EMBED - N3_PLAYER)];
    out[idx] = val;
}

__global__ void n3_bias_relu_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[idx % dim])));
}

__global__ void n3_relu_backward_kernel(
    precision_t* __restrict__ grad, const precision_t* __restrict__ out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (to_float(out[idx]) <= 0.0f) grad[idx] = from_float(0.0f);
}


__global__ void bias_grad_kernel(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad, int N, int dim) {
    int d = blockIdx.x;
    if (d >= dim) return;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += to_float(grad[i * dim + d]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[d] = from_float(sum);
    }
}

// NCHW bias grad: sum over (B, OH, OW) for each OC channel
__global__ void n3_conv_bias_grad_nchw(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad,
    int B, int OC, int spatial) {
    int oc = blockIdx.x;
    if (oc >= OC) return;
    float sum = 0.0f;
    int total = B * spatial;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial, s = i % spatial;
        sum += to_float(grad[b * OC * spatial + oc * spatial + s]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[oc] = from_float(sum);
    }
}

__global__ void n3_concat_backward_conv_kernel(
    precision_t* __restrict__ conv_grad, const precision_t* __restrict__ concat_grad, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONV_FLAT) return;
    int b = idx / N3_CONV_FLAT, c = idx % N3_CONV_FLAT;
    conv_grad[b * N3_CONV_FLAT + c] = concat_grad[b * N3_CONCAT + c];
}

// Embedding backward: scatter-add grad from concat_grad's player_embed region
// into embed_wgrad (float accumulation buffer).
// Each (b, f) looked up row obs[b, MAP_SIZE+f] from the table.
__global__ void n3_embedding_backward_kernel(
    float* __restrict__ embed_wgrad_f,
    const precision_t* __restrict__ concat_grad,
    const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_PLAYER * N3_EMBED_DIM) return;
    int b = idx / (N3_PLAYER * N3_EMBED_DIM);
    int rem = idx % (N3_PLAYER * N3_EMBED_DIM);
    int f = rem / N3_EMBED_DIM;
    int d = rem % N3_EMBED_DIM;
    int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
    float g = to_float(concat_grad[b * N3_CONCAT + N3_CONV_FLAT + f * N3_EMBED_DIM + d]);
    atomicAdd(&embed_wgrad_f[val * N3_EMBED_DIM + d], g);
}

// Cast float buffer to precision_t
__global__ void n3_float_to_precision_kernel(
    precision_t* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

// ---- atomicAdd for precision_t ----
#ifdef PRECISION_FLOAT
__device__ __forceinline__ void atomicAdd_precision(precision_t* addr, precision_t val) {
    atomicAdd(addr, val);
}
#else
__device__ __forceinline__ void atomicAdd_precision(precision_t* addr, precision_t val) {
    // bf16 atomicAdd via CAS on enclosing 32-bit word
    unsigned int* addr_u32 = (unsigned int*)((size_t)addr & ~2ULL);
    bool is_high = ((size_t)addr & 2) != 0;
    unsigned int old_u32 = *addr_u32, assumed;
    do {
        assumed = old_u32;
        __nv_bfloat16* pair = (__nv_bfloat16*)&old_u32;
        float sum = __bfloat162float(pair[is_high]) + __bfloat162float(val);
        unsigned int new_u32 = assumed;
        ((__nv_bfloat16*)&new_u32)[is_high] = __float2bfloat16(sum);
        old_u32 = atomicCAS(addr_u32, assumed, new_u32);
    } while (old_u32 != assumed);
}
#endif

// ---- NCHW bias kernels for im2col conv path ----

__global__ void conv_bias_kernel(precision_t* __restrict__ data,
        const precision_t* __restrict__ bias, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int oc = (idx / spatial) % OC;
    data[idx] = from_float(to_float(data[idx]) + to_float(bias[oc]));
}

__global__ void conv_bias_relu_kernel(precision_t* __restrict__ data,
        const precision_t* __restrict__ bias, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int oc = (idx / spatial) % OC;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[oc])));
}

// ---- im2col + cuBLAS conv (no cuDNN) ----
// NCHW layout throughout. Weight stored as (OC, IC*K*K).
// im2col produces (B*OH*OW, IC*K*K), matmul with W^T gives (B*OH*OW, OC),
// then reshape to NCHW (B, OC, OH, OW).

__global__ void im2col_kernel(
    const precision_t* __restrict__ input, precision_t* __restrict__ col,
    int B, int IC, int IH, int IW, int K, int S, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OH * OW * IC * K * K;
    if (idx >= total) return;
    int col_w = IC * K * K;
    int row = idx / col_w;
    int c = idx % col_w;
    int b = row / (OH * OW);
    int rem = row % (OH * OW);
    int oh = rem / OW, ow = rem % OW;
    int ic = c / (K * K), kk = c % (K * K);
    int kh = kk / K, kw = kk % K;
    int ih = oh * S + kh, iw = ow * S + kw;
    col[idx] = input[b * IC * IH * IW + ic * IH * IW + ih * IW + iw];
}

// Backward: col2im — input-centric gather to avoid atomics.
// Each thread owns one (b, ic, ih, iw) element and sums contributions from all
// (oh, ow, kh, kw) patches that map to it.
__global__ void col2im_kernel(
    const precision_t* __restrict__ col, precision_t* __restrict__ grad_input,
    int B, int IC, int IH, int IW, int K, int S, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * IC * IH * IW;
    if (idx >= total) return;
    int iw = idx % IW;
    int ih = (idx / IW) % IH;
    int ic = (idx / (IW * IH)) % IC;
    int b  = idx / (IW * IH * IC);
    float sum = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        int ih_off = ih - kh;
        if (ih_off < 0 || ih_off % S != 0) continue;
        int oh = ih_off / S;
        if (oh >= OH) continue;
        for (int kw = 0; kw < K; kw++) {
            int iw_off = iw - kw;
            if (iw_off < 0 || iw_off % S != 0) continue;
            int ow = iw_off / S;
            if (ow >= OW) continue;
            int col_idx = (b * OH * OW + oh * OW + ow) * (IC * K * K) + ic * K * K + kh * K + kw;
            sum += to_float(col[col_idx]);
        }
    }
    grad_input[idx] = from_float(sum);
}

// Transpose (B, OC, OH, OW) -> (B*OH*OW, OC)  [NCHW to row-major spatial-first]
__global__ void nchw_to_rows_kernel(
    const precision_t* __restrict__ src, precision_t* __restrict__ dst,
    int B, int OC, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[(b * spatial + s) * OC + oc] = src[idx];
}

// Transpose (B*OH*OW, OC) -> (B, OC, OH, OW)  [row-major spatial-first to NCHW]
__global__ void rows_to_nchw_kernel(
    const precision_t* __restrict__ src, precision_t* __restrict__ dst,
    int B, int OC, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[idx] = src[(b * spatial + s) * OC + oc];
}

// Forward: im2col conv + bias + optional relu. All NCHW.
// col_buf: pre-allocated (max_B * OH * OW, IC * K * K)
// mm_buf:  pre-allocated (max_B * OH * OW, OC)  — row-major (spatial-first)
static void gemm_conv_forward(
    PrecisionTensor* weight, PrecisionTensor* bias,
    precision_t* input, precision_t* output,
    precision_t* col_buf, precision_t* mm_buf,
    int B, int IC, int IH, int IW, int OC, int K, int S, int OH, int OW,
    bool relu, cudaStream_t stream
) {
    int col_rows = B * OH * OW;
    int col_cols = IC * K * K;
    int total_col = col_rows * col_cols;
    int total_out = B * OC * OH * OW;

    // im2col: input NCHW -> col (B*OH*OW, IC*K*K)
    im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        input, col_buf, B, IC, IH, IW, K, S, OH, OW);

    // matmul: col (B*OH*OW, IC*K*K) @ W^T (IC*K*K, OC) = mm_buf (B*OH*OW, OC)
    PrecisionTensor col_t = {.data = col_buf, .shape = {col_rows, col_cols}};
    PrecisionTensor mm_t  = {.data = mm_buf,  .shape = {col_rows, OC}};
    puf_mm(&col_t, weight, &mm_t, stream);

    // transpose (B*OH*OW, OC) -> (B, OC, OH, OW) NCHW + bias + relu
    int spatial = OH * OW;
    rows_to_nchw_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
        mm_buf, output, B, OC, spatial);
    if (relu) {
        conv_bias_relu_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
            output, bias->data, B, OC, spatial);
    } else {
        conv_bias_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
            output, bias->data, B, OC, spatial);
    }
}

// Backward: weight grad + optional input grad via im2col/col2im + cuBLAS.
// grad_output is NCHW (B, OC, OH, OW). saved_input is NCHW.
// Caller handles relu backward and bias grad (same as cuDNN path).
static void gemm_conv_backward(
    PrecisionTensor* weight,
    precision_t* saved_input, precision_t* grad_output,
    precision_t* wgrad, precision_t* input_grad,
    precision_t* col_buf, precision_t* mm_buf,
    int B, int IC, int IH, int IW, int OC, int K, int S, int OH, int OW,
    cudaStream_t stream
) {
    int col_rows = B * OH * OW;
    int col_cols = IC * K * K;
    int total_col = col_rows * col_cols;
    int total_out = B * OC * OH * OW;
    int spatial = OH * OW;

    // Transpose grad_output NCHW -> (B*OH*OW, OC)
    nchw_to_rows_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
        grad_output, mm_buf, B, OC, spatial);

    // im2col of saved_input
    im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        saved_input, col_buf, B, IC, IH, IW, K, S, OH, OW);

    // Weight grad: mm_buf^T (OC, B*OH*OW) @ col_buf (B*OH*OW, IC*K*K) = wgrad (OC, IC*K*K)
    PrecisionTensor mm_t  = {.data = mm_buf,  .shape = {col_rows, OC}};
    PrecisionTensor col_t = {.data = col_buf, .shape = {col_rows, col_cols}};
    PrecisionTensor wg_t  = {.data = wgrad,   .shape = {OC, col_cols}};
    puf_mm_tn(&mm_t, &col_t, &wg_t, stream);

    // Input grad (optional): mm_buf (B*OH*OW, OC) @ weight (OC, IC*K*K) = col_grad (B*OH*OW, IC*K*K)
    if (input_grad) {
        puf_mm_nn(&mm_t, weight, &col_t, stream);  // reuse col_buf as col_grad
        col2im_kernel<<<grid_size(B * IC * IH * IW), BLOCK_SIZE, 0, stream>>>(
            col_buf, input_grad, B, IC, IH, IW, K, S, OH, OW);
    }
}

// ---- NMMO3 encoder structs ----

struct NMMO3EncoderWeights {
    ConvWeights conv1, conv2;
    PrecisionTensor embed_w, proj_w, proj_b;
    int obs_size, hidden;
};

struct NMMO3EncoderActivations {
    ConvActivations conv1, conv2;
    PrecisionTensor col1, mm1, col2, mm2;  // im2col + matmul scratch buffers
    PrecisionTensor multihot, embed_out, concat, out, saved_obs;
    PrecisionTensor embed_wgrad, proj_wgrad, proj_bgrad;
    FloatTensor embed_wgrad_f;  // float accumulation buffer for scatter-add
};

static NMMO3EncoderWeights* nmmo3_encoder_create(int obs_size, int hidden) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)calloc(1, sizeof(NMMO3EncoderWeights));
    ew->obs_size = obs_size; ew->hidden = hidden;
    conv_init(&ew->conv1, N3_C1_IC, N3_C1_OC, N3_C1_K, N3_C1_S, N3_MAP_H, N3_MAP_W, true);
    conv_init(&ew->conv2, N3_C2_IC, N3_C2_OC, N3_C2_K, N3_C2_S, N3_C1_OH, N3_C1_OW, false);
    return ew;
}

// ---- NMMO3 encoder interface ----

static PrecisionTensor nmmo3_encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = input.shape[0];

    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);

    cudaMemsetAsync(a->multihot.data, 0, (int64_t)B * N3_MULTIHOT * N3_MAP_H * N3_MAP_W * sizeof(precision_t), stream);
    n3_multihot_kernel<<<grid_size(B * N3_MAP_H * N3_MAP_W), BLOCK_SIZE, 0, stream>>>(
        a->multihot.data, input.data, B, ew->obs_size);

    gemm_conv_forward(&ew->conv1.w, &ew->conv1.b, a->multihot.data, a->conv1.out.data,
        a->col1.data, a->mm1.data, B, N3_C1_IC, N3_MAP_H, N3_MAP_W,
        N3_C1_OC, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW, true, stream);
    if (a->conv1.saved_input.data)
        cudaMemcpyAsync(a->conv1.saved_input.data, a->multihot.data,
            (int64_t)B * N3_C1_IC * N3_MAP_H * N3_MAP_W * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    gemm_conv_forward(&ew->conv2.w, &ew->conv2.b, a->conv1.out.data, a->conv2.out.data,
        a->col2.data, a->mm2.data, B, N3_C2_IC, N3_C1_OH, N3_C1_OW,
        N3_C2_OC, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW, false, stream);
    if (a->conv2.saved_input.data)
        cudaMemcpyAsync(a->conv2.saved_input.data, a->conv1.out.data,
            (int64_t)B * N3_C2_IC * N3_C1_OH * N3_C1_OW * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);

    n3_embedding_kernel<<<grid_size(B * N3_PLAYER), BLOCK_SIZE, 0, stream>>>(
        a->embed_out.data, input.data, ew->embed_w.data, B, ew->obs_size);
    n3_concat_kernel<<<grid_size(B * N3_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2.out.data, a->embed_out.data, input.data, B, ew->obs_size);

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nmmo3_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * H);
    bias_grad_kernel<<<H, 256, 0, stream>>>(
        a->proj_bgrad.data, grad.data, B, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);

    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, N3_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    n3_concat_backward_conv_kernel<<<grid_size(B * N3_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv2.grad.data, grad_concat.data, B);

    n3_conv_bias_grad_nchw<<<ew->conv2.OC, 256, 0, stream>>>(
        a->conv2.bgrad.data, a->conv2.grad.data,
        B, ew->conv2.OC, ew->conv2.OH * ew->conv2.OW);
    gemm_conv_backward(&ew->conv2.w, a->conv2.saved_input.data, a->conv2.grad.data,
        a->conv2.wgrad.data, a->conv1.grad.data,
        a->col2.data, a->mm2.data, B, N3_C2_IC, N3_C1_OH, N3_C1_OW,
        N3_C2_OC, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW, stream);

    n3_relu_backward_kernel<<<grid_size(B * ew->conv1.OC * ew->conv1.OH * ew->conv1.OW), BLOCK_SIZE, 0, stream>>>(
        a->conv1.grad.data, a->conv1.out.data,
        B * ew->conv1.OC * ew->conv1.OH * ew->conv1.OW);
    n3_conv_bias_grad_nchw<<<ew->conv1.OC, 256, 0, stream>>>(
        a->conv1.bgrad.data, a->conv1.grad.data,
        B, ew->conv1.OC, ew->conv1.OH * ew->conv1.OW);
    gemm_conv_backward(&ew->conv1.w, a->conv1.saved_input.data, a->conv1.grad.data,
        a->conv1.wgrad.data, NULL,
        a->col1.data, a->mm1.data, B, N3_C1_IC, N3_MAP_H, N3_MAP_W,
        N3_C1_OC, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW, stream);

    // Embedding backward: scatter-add from concat gradient into float buffer, then cast
    int embed_n = N3_EMBED_VOCAB * N3_EMBED_DIM;
    cudaMemsetAsync(a->embed_wgrad_f.data, 0, embed_n * sizeof(float), stream);
    n3_embedding_backward_kernel<<<grid_size(B * N3_PLAYER * N3_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad_f.data, grad_concat.data, a->saved_obs.data, B, ew->obs_size);
    n3_float_to_precision_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->embed_wgrad_f.data, embed_n);
}

static void nmmo3_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    conv_init_weights(&ew->conv1, seed, stream);
    conv_init_weights(&ew->conv2, seed, stream);
    auto init2d = [&](PrecisionTensor& t, int rows, int cols, float gain) {
        PrecisionTensor wt = {.data = t.data, .shape = {rows, cols}};
        puf_kaiming_init(&wt, gain, (*seed)++, stream);
    };
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    init2d(ew->proj_w, ew->hidden, N3_CONCAT, 1.0f);
    cudaMemsetAsync(ew->proj_b.data, 0, numel(ew->proj_b.shape) * sizeof(precision_t), stream);
}

static void nmmo3_encoder_reg_params(void* w, Allocator* alloc) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    conv_reg_params(&ew->conv1, alloc);
    conv_reg_params(&ew->conv2, alloc);
    ew->embed_w = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    ew->proj_w  = {.shape = {ew->hidden, N3_CONCAT}};
    ew->proj_b  = {.shape = {ew->hidden}};
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->proj_w);  alloc_register(alloc,&ew->proj_b);
}

static void nmmo3_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    *a = {};
    a->multihot = {.shape = {B_TT, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}};
    alloc_register(acts,&a->multihot);
    // Conv1 buffers
    a->conv1.out         = {.shape = {B_TT * N3_C1_OC * N3_C1_OH * N3_C1_OW}};
    a->conv1.grad        = {.shape = {B_TT * N3_C1_OC * N3_C1_OH * N3_C1_OW}};
    a->conv1.saved_input = {.shape = {B_TT * N3_C1_IC * N3_MAP_H * N3_MAP_W}};
    a->conv1.wgrad       = {.shape = {N3_C1_OC, N3_C1_IC * N3_C1_K * N3_C1_K}};
    a->conv1.bgrad       = {.shape = {N3_C1_OC}};
    alloc_register(acts,&a->conv1.out); alloc_register(acts,&a->conv1.grad); alloc_register(acts,&a->conv1.saved_input);
    alloc_register(grads,&a->conv1.wgrad); alloc_register(grads,&a->conv1.bgrad);
    a->col1 = {.shape = {B_TT * N3_C1_OH * N3_C1_OW, N3_C1_IC * N3_C1_K * N3_C1_K}};
    a->mm1  = {.shape = {B_TT * N3_C1_OH * N3_C1_OW, N3_C1_OC}};
    alloc_register(acts,&a->col1); alloc_register(acts,&a->mm1);
    // Conv2 buffers
    a->conv2.out         = {.shape = {B_TT * N3_C2_OC * N3_C2_OH * N3_C2_OW}};
    a->conv2.grad        = {.shape = {B_TT * N3_C2_OC * N3_C2_OH * N3_C2_OW}};
    a->conv2.saved_input = {.shape = {B_TT * N3_C2_IC * N3_C1_OH * N3_C1_OW}};
    a->conv2.wgrad       = {.shape = {N3_C2_OC, N3_C2_IC * N3_C2_K * N3_C2_K}};
    a->conv2.bgrad       = {.shape = {N3_C2_OC}};
    alloc_register(acts,&a->conv2.out); alloc_register(acts,&a->conv2.grad); alloc_register(acts,&a->conv2.saved_input);
    alloc_register(grads,&a->conv2.wgrad); alloc_register(grads,&a->conv2.bgrad);
    a->col2 = {.shape = {B_TT * N3_C2_OH * N3_C2_OW, N3_C2_IC * N3_C2_K * N3_C2_K}};
    a->mm2  = {.shape = {B_TT * N3_C2_OH * N3_C2_OW, N3_C2_OC}};
    alloc_register(acts,&a->col2); alloc_register(acts,&a->mm2);
    a->embed_out = {.shape = {B_TT, N3_PLAYER_EMBED}};
    a->concat    = {.shape = {B_TT, N3_CONCAT}};
    a->out       = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts,&a->embed_out); alloc_register(acts,&a->concat);
    alloc_register(acts,&a->out);       alloc_register(acts,&a->saved_obs);
    a->embed_wgrad = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->embed_wgrad_f = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->proj_wgrad  = {.shape = {ew->hidden, N3_CONCAT}};
    a->proj_bgrad  = {.shape = {ew->hidden}};
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(acts,&a->embed_wgrad_f);
    alloc_register(grads,&a->proj_wgrad);  alloc_register(grads,&a->proj_bgrad);
}

static void nmmo3_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    a->multihot = {.shape = {B, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}};
    alloc_register(alloc,&a->multihot);
    a->conv1.out = {.shape = {B * N3_C1_OC * N3_C1_OH * N3_C1_OW}};
    alloc_register(alloc,&a->conv1.out);
    a->col1 = {.shape = {B * N3_C1_OH * N3_C1_OW, N3_C1_IC * N3_C1_K * N3_C1_K}};
    a->mm1  = {.shape = {B * N3_C1_OH * N3_C1_OW, N3_C1_OC}};
    alloc_register(alloc,&a->col1); alloc_register(alloc,&a->mm1);
    a->conv2.out = {.shape = {B * N3_C2_OC * N3_C2_OH * N3_C2_OW}};
    alloc_register(alloc,&a->conv2.out);
    a->col2 = {.shape = {B * N3_C2_OH * N3_C2_OW, N3_C2_IC * N3_C2_K * N3_C2_K}};
    a->mm2  = {.shape = {B * N3_C2_OH * N3_C2_OW, N3_C2_OC}};
    alloc_register(alloc,&a->col2); alloc_register(alloc,&a->mm2);
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}};
    a->concat    = {.shape = {B, N3_CONCAT}};
    a->out       = {.shape = {B, ew->hidden}};
    alloc_register(alloc,&a->embed_out); alloc_register(alloc,&a->concat); alloc_register(alloc,&a->out);
}

static void* nmmo3_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return nmmo3_encoder_create(e->in_dim, e->out_dim);
}
static void nmmo3_encoder_free_weights(void* weights) { free(weights); }
static void nmmo3_encoder_free_activations(void* activations) { free(activations); }

// ---- Nethack constants ----
// Obs layout (must match ocean/nethack/nethack.h defaults):
//   [0, 2*NH_GRID)  glyphs, int16 LE, NETHACK_GLYPH_CROP^2 egocentric window
//   [2*NH_GRID, +4*NH_BL_RAW)  blstats, int32 LE

static constexpr int NH_MAP = 21;                  // NETHACK_GLYPH_CROP
static constexpr int NH_GRID = NH_MAP * NH_MAP;
static constexpr int NH_GLYPH_VOCAB = 5977;        // MAX_GLYPH + 1 (NetHack 3.6.6)
static constexpr int NH_EMBED_DIM = 32;
static constexpr int NH_BL_RAW = 27;               // NLE_BLSTATS_SIZE
static constexpr int NH_BL_HUNGER = 21, NH_BL_CONDITION = 25;
static constexpr int NH_BL_FEAT = 25 + 7 + 13;     // scalars + hunger onehot + condition bits
static constexpr int NH_BL_HID = 64;
static constexpr int NH_OBS_SIZE = NH_GRID * 2 + NH_BL_RAW * 4;
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
static constexpr int NH_HOT_T = 7;                       // hot-glyph dT smem slots (7x1600 fp32 = 44.8KB, static smem limit)

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

// conv2 bias grad: per-oc sum over grad_concat's conv slice.
__global__ void nh_bias_from_concat_kernel(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad_concat, int B) {
    int oc = blockIdx.x;
    if (oc >= NH_C2_OC) return;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < B * NH_C2_SP; i += blockDim.x) {
        int b = i / NH_C2_SP, p = i % NH_C2_SP;
        sum += to_float(grad_concat[(int64_t)b * NH_CONCAT + oc * NH_C2_SP + p]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[oc] = from_float(sum);
    }
}

// Decode int32 LE blstats and expand to NH_BL_FEAT normalized features:
// 25 scaled scalars, 7 hunger one-hot, 13 condition bits.
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

// Scatter conv1's output grad (rows layout) into fp32 dT. Hot glyphs
// accumulate in smem (consecutive oc lanes -> conflict-free) and flush once
// per block; the cold tail goes straight to global atomics.
__global__ void nh_dT_scatter_kernel(
    float* __restrict__ dT_f, const precision_t* __restrict__ grad,
    const float* __restrict__ idx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n, int B) {
    __shared__ float acc_s[NH_HOT_T][NH_TROW];
    for (int i = threadIdx.x; i < NH_HOT_T * NH_TROW; i += blockDim.x)
        acc_s[i / NH_TROW][i % NH_TROW] = 0.0f;
    __syncthreads();
    int64_t total = (int64_t)B * NH_C1_SP * NH_C1_OC;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        int oc = i % NH_C1_OC;
        int p  = (i / NH_C1_OC) % NH_C1_SP;
        int64_t b = i / (NH_C1_OC * NH_C1_SP);
        float g = to_float(grad[i]);
        if (g == 0.0f) continue;  // relu-masked
        int oh = p / NH_C1_OW, ow = p % NH_C1_OW;
        const float* gi = idx + b * NH_GRID;
        #pragma unroll
        for (int t = 0; t < NH_TAPS; t++) {
            int cell = (NH_C1_S * oh + t / NH_C1_K) * NH_MAP + NH_C1_S * ow + t % NH_C1_K;
            int gl = (int)gi[cell];
            int slot = hot_map[gl];
            if (slot >= 0) atomicAdd(&acc_s[slot][t * NH_C1_OC + oc], g);
            else atomicAdd(&dT_f[(int64_t)gl * NH_TROW + t * NH_C1_OC + oc], g);
        }
    }
    __syncthreads();
    int n = *hot_n;
    for (int i = threadIdx.x; i < n * NH_TROW; i += blockDim.x) {
        float v = acc_s[i / NH_TROW][i % NH_TROW];
        if (v != 0.0f)
            atomicAdd(&dT_f[(int64_t)hot_list[i / NH_TROW] * NH_TROW + i % NH_TROW], v);
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
    FloatTensor dT_f;                      // fp32 dT scatter staging
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
    cudnnDataType_t dt = n3_cudnn_dtype();
    cudnnHandle_t h = get_cudnn_handle();
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

    int n;
    cudnnConvolutionFwdAlgoPerf_t fp;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(h, a->c2_in, ew->c2_filt, ew->c2_conv, a->c2_out, 1, &n, &fp));
    a->c2_fwd_algo = fp.algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(h, a->c2_in, ew->c2_filt, ew->c2_conv, a->c2_out,
        a->c2_fwd_algo, &a->c2_fwd_ws_n));
    if (a->c2_fwd_ws_n) cudaMalloc(&a->c2_fwd_ws, a->c2_fwd_ws_n);
    if (!bwd) return;

    cudnnConvolutionBwdFilterAlgoPerf_t wp;
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(h, a->c2_in, a->c2_out, ew->c2_conv, ew->c2_filt, 1, &n, &wp));
    a->c2_wgrad_algo = wp.algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(h, a->c2_in, a->c2_out, ew->c2_conv, ew->c2_filt,
        a->c2_wgrad_algo, &a->c2_wgrad_ws_n));
    if (a->c2_wgrad_ws_n) cudaMalloc(&a->c2_wgrad_ws, a->c2_wgrad_ws_n);

    cudnnConvolutionBwdDataAlgoPerf_t dp;
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(h, ew->c2_filt, a->c2_out, ew->c2_conv, a->c2_in, 1, &n, &dp));
    a->c2_dgrad_algo = dp.algo;
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
    cudnnHandle_t cudnn = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(cudnn, stream));
    float c_alpha = 1.0f, c_beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn,          // bias added in concat
        &c_alpha, a->c2_in, a->conv1.out.data, ew->c2_filt, a->w2p.data,
        ew->c2_conv, a->c2_fwd_algo, a->c2_fwd_ws, a->c2_fwd_ws_n,
        &c_beta, a->c2_out, a->conv2.out.data));

    nh_blstats_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->bl_feats.data, input.data, B);
    puf_mm(&a->bl_feats, &ew->bl_w, &a->bl_out, stream);
    n3_bias_relu_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_out.data, ew->bl_b.data, B * NH_BL_HID, NH_BL_HID);

    nh_concat_kernel<<<grid_size(B * NH_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2.out.data, ew->conv2.b.data, a->bl_out.data, a->bl_feats.data, B);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nethack_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * H);
    bias_grad_kernel<<<H, 256, 0, stream>>>(
        a->proj_bgrad.data, grad.data, B, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);

    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, NH_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    // Conv branch. conv2 grads via cuDNN NHWC: forward's conv1.out is the
    // saved input for wgrad; forward's w2p (KRSC) is the filter for dgrad.
    nh_bias_from_concat_kernel<<<NH_C2_OC, 256, 0, stream>>>(
        a->conv2.bgrad.data, grad_concat.data, B);
    nh_dout_rows_kernel<<<grid_size(B * NH_C2_SP * NH_C2_OC), BLOCK_SIZE, 0, stream>>>(
        a->dout2.data, grad_concat.data, B);
    cudnnHandle_t cudnn = get_cudnn_handle();
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
    n3_relu_backward_kernel<<<grid_size(B * NH_C1_SP * NH_C1_OC), BLOCK_SIZE, 0, stream>>>(
        a->conv1.grad.data, a->conv1.out.data, B * NH_C1_SP * NH_C1_OC);
    bias_grad_kernel<<<NH_C1_OC, 256, 0, stream>>>(
        a->conv1.bgrad.data, a->conv1.grad.data, B * NH_C1_SP, NH_C1_OC);

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
    cudaMemsetAsync(a->dT_f.data, 0, (size_t)dT_n * sizeof(float), stream);
    nh_dT_scatter_kernel<<<1024, 256, 0, stream>>>(
        a->dT_f.data, a->conv1.grad.data, a->glyph_idx.data, hot_map, hot_list, hot_n, B);
    n3_float_to_precision_kernel<<<grid_size(dT_n), BLOCK_SIZE, 0, stream>>>(
        a->dT.data, a->dT_f.data, dT_n);
    puf_mm_nn(&a->dT, &a->w_perm, &a->embed_wgrad, stream);   // dE  = dT @ W'
    puf_mm_tn(&a->dT, &ew->embed_w, &a->dw_perm, stream);     // dW' = dT^T @ E
    nh_unpermute_w_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->conv1.wgrad.data, a->dw_perm.data);

    // Blstats branch (raw-feature slice of concat has no upstream params)
    nh_slice_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, grad_concat.data, B, NH_CONCAT, NH_CONV_FLAT, NH_BL_HID);
    n3_relu_backward_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, a->bl_out.data, B * NH_BL_HID);
    bias_grad_kernel<<<NH_BL_HID, 256, 0, stream>>>(
        a->bl_bgrad.data, a->bl_grad.data, B, NH_BL_HID);
    PrecisionTensor blg = {.data = a->bl_grad.data, .shape = {B, NH_BL_HID}};
    puf_mm_tn(&blg, &a->bl_feats, &a->bl_wgrad, stream);
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
    a->dT_f    = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dw_perm = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->sort_buf = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_T + 1}};
    alloc_register(acts,&a->w_perm); alloc_register(acts,&a->glyph_T);
    alloc_register(acts,&a->dT);     alloc_register(acts,&a->dT_f);
    alloc_register(acts,&a->dw_perm); alloc_register(acts,&a->sort_buf);
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

// ---- Nethack Mixer encoder (select with NETHACK_ENCODER=mixer) ----
// Non-overlapping 3x3 patch stem over glyph embeddings + one norm-free
// MLP-Mixer block (token mix over 49 patches = full-crop receptive field,
// then channel mix) + the same blstats branch and projection as the conv
// encoder. All dense GEMMs — no im2col/cuDNN — and the embedding backward
// collapses to a d=16 single-patch scatter instead of the conv path's
// 25-tap dT table.
static constexpr int NHM_D  = 16;                    // glyph embed dim
static constexpr int NHM_P  = 3;                     // patch side (21 = 7*3)
static constexpr int NHM_G  = NH_MAP / NHM_P;        // 7 patches per side
static constexpr int NHM_T  = NHM_G * NHM_G;         // 49 tokens
// Token-mix buffers/weights use the token dim padded to 64: 49 is odd, which
// disqualifies cuBLAS tensor-core kernels (16B alignment) and left the
// token-mix wgrads on a ~7ms SIMT path. Pad columns are zero-filled by the
// transpose, so they are mathematically inert (zero activations, zero grads).
static constexpr int NHM_TP = 64;
static constexpr int NHM_IN = NHM_P * NHM_P * NHM_D; // 144 per-patch input
static constexpr int NHM_C  = 64;                    // token channels
static constexpr int NHM_TH = 64;                    // token-mix hidden
static constexpr int NHM_CH = 96;                    // channel-mix hidden
static constexpr int NHM_FLAT = NHM_T * NHM_C;       // 3136
static constexpr int NHM_CONCAT = NHM_FLAT + NH_BL_HID + NH_BL_FEAT;

// gath (B, 49, 144): per-patch concat of the 9 cells' glyph embeddings.
__global__ void nhm_gather_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ embed,
    const float* __restrict__ idx, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * NHM_T * NHM_IN) return;
    int j = i % NHM_IN;
    int p = (i / NHM_IN) % NHM_T;
    int64_t b = i / ((int64_t)NHM_IN * NHM_T);
    int w = j / NHM_D, d = j % NHM_D;
    int cell = ((p / NHM_G) * NHM_P + w / NHM_P) * NH_MAP + (p % NHM_G) * NHM_P + w % NHM_P;
    int g = (int)idx[b * NH_GRID + cell];
    out[i] = embed[g * NHM_D + d];
}

// (B, T, C) -> (B, C, TP): token-dim-padded transpose, pad zero-filled.
__global__ void nhm_t_kernel(
    precision_t* __restrict__ dst, const precision_t* __restrict__ src, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * NHM_TP * NHM_C) return;
    int t = i % NHM_TP;
    int c = (i / NHM_TP) % NHM_C;
    int64_t b = i / ((int64_t)NHM_TP * NHM_C);
    dst[i] = t < NHM_T ? src[(b * NHM_T + t) * NHM_C + c] : from_float(0.0f);
}

// mix1[b,t,c] = tok_out[b,c,t] + tok_b2[t] + stem_out[b,t,c]
__global__ void nhm_untrans_bias_res_kernel(
    precision_t* __restrict__ dst, const precision_t* __restrict__ tok_out,
    const precision_t* __restrict__ bias, const precision_t* __restrict__ res, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * NHM_T * NHM_C) return;
    int c = i % NHM_C;
    int t = (i / NHM_C) % NHM_T;
    int64_t b = i / ((int64_t)NHM_C * NHM_T);
    dst[i] = from_float(to_float(tok_out[(b * NHM_C + c) * NHM_TP + t])
                        + to_float(bias[t]) + to_float(res[i]));
}

// dst[b,t,c] = gT[b,c,t] + add[b,t,c]  (backward untranspose + residual)
__global__ void nhm_untrans_add_kernel(
    precision_t* __restrict__ dst, const precision_t* __restrict__ gT,
    const precision_t* __restrict__ add, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * NHM_T * NHM_C) return;
    int c = i % NHM_C;
    int t = (i / NHM_C) % NHM_T;
    int64_t b = i / ((int64_t)NHM_C * NHM_T);
    dst[i] = from_float(to_float(gT[(b * NHM_C + c) * NHM_TP + t]) + to_float(add[i]));
}

// Coalesced bias grad for wide-row tensors ((rows, dim) row-major, dim<=256):
// grid-stride coalesced reads, smem accumulation, one global atomic per
// (block, column). The generic bias_grad_kernel strides columns and is ~10x
// slower at these row counts. acc must be pre-zeroed fp32.
__global__ void nhm_bias_grad_kernel(
    float* __restrict__ acc, const precision_t* __restrict__ grad, int64_t rows, int dim) {
    extern __shared__ float sdata[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sdata[i] = 0.0f;
    __syncthreads();
    int64_t total = rows * dim;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x)
        atomicAdd(&sdata[(int)(i % dim)], to_float(grad[i]));
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        if (sdata[i] != 0.0f) atomicAdd(&acc[i], sdata[i]);
}

// in place: data[b,t,c] += bias[c] + res[b,t,c]
__global__ void nhm_bias_res_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias,
    const precision_t* __restrict__ res, int64_t total) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    data[i] = from_float(to_float(data[i]) + to_float(bias[i % NHM_C]) + to_float(res[i]));
}

// concat = [tokens flat | bl hidden | bl raw feats]
__global__ void nhm_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ flat,
    const precision_t* __restrict__ bl_out, const precision_t* __restrict__ bl_feats, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * NHM_CONCAT) return;
    int c = i % NHM_CONCAT;
    int64_t b = i / NHM_CONCAT;
    precision_t v;
    if (c < NHM_FLAT)                     v = flat[b * NHM_FLAT + c];
    else if (c < NHM_FLAT + NH_BL_HID)    v = bl_out[b * NH_BL_HID + c - NHM_FLAT];
    else                                  v = bl_feats[b * NH_BL_FEAT + c - NHM_FLAT - NH_BL_HID];
    out[i] = v;
}

// Embedding grad scatter. Reuses the conv path's hist/hot-select machinery;
// hot glyphs accumulate in smem (NH_HOT_T x 16 floats = 448B), cold tail via
// global atomics. Each cell contributes to exactly one patch position.
__global__ void nhm_embed_scatter_kernel(
    float* __restrict__ wg, const precision_t* __restrict__ g_gath,
    const float* __restrict__ idx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n, int B) {
    __shared__ float acc[NH_HOT_T][NHM_D];
    for (int i = threadIdx.x; i < NH_HOT_T * NHM_D; i += blockDim.x)
        acc[i / NHM_D][i % NHM_D] = 0.0f;
    __syncthreads();
    int64_t total = (int64_t)B * NHM_T * NHM_IN;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        float v = to_float(g_gath[i]);
        if (v == 0.0f) continue;
        int j = i % NHM_IN;
        int p = (i / NHM_IN) % NHM_T;
        int64_t b = i / ((int64_t)NHM_IN * NHM_T);
        int w = j / NHM_D, d = j % NHM_D;
        int cell = ((p / NHM_G) * NHM_P + w / NHM_P) * NH_MAP + (p % NHM_G) * NHM_P + w % NHM_P;
        int g = (int)idx[b * NH_GRID + cell];
        int slot = hot_map[g];
        if (slot >= 0) atomicAdd(&acc[slot][d], v);
        else atomicAdd(&wg[(int64_t)g * NHM_D + d], v);
    }
    __syncthreads();
    int n = *hot_n;
    for (int i = threadIdx.x; i < n * NHM_D; i += blockDim.x) {
        float v = acc[i / NHM_D][i % NHM_D];
        if (v != 0.0f) atomicAdd(&wg[(int64_t)hot_list[i / NHM_D] * NHM_D + i % NHM_D], v);
    }
}

struct NethackMixerWeights {
    PrecisionTensor embed_w, stem_w, stem_b, tok_w1, tok_b1, tok_w2, tok_b2,
                    ch_w1, ch_b1, ch_w2, ch_b2, bl_w, bl_b, proj_w, proj_b;
    int obs_size, hidden;
    int use_mixer;   // 0 = patch-flat variant: stem -> concat, no mixer block
};

struct NethackMixerActivations {
    FloatTensor glyph_idx;
    IntTensor sort_buf;                     // counts | hot_map | hot_list | hot_n
    FloatTensor embed_wgrad_f;
    FloatTensor bias_acc;               // fp32 scratch for nhm_bias_grad
    PrecisionTensor gath, stem_out, tokT, tok_h, tok_out, mix1, ch_h, mix2;
    PrecisionTensor bl_feats, bl_out, bl_grad, concat, out;
    // backward scratch (g_tok_out reuses tok_out)
    PrecisionTensor g_mix2, g_ch_h, g_mix1, g_tok_h, g_tokT, g_stem, g_gath;
    PrecisionTensor embed_wgrad, stem_wgrad, stem_bgrad, tok_w1g, tok_b1g,
                    tok_w2g, tok_b2g, ch_w1g, ch_b1g, ch_w2g, ch_b2g,
                    bl_wgrad, bl_bgrad, proj_wgrad, proj_bgrad;
};

static PrecisionTensor nethack_mixer_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    NethackMixerWeights* ew = (NethackMixerWeights*)w;
    NethackMixerActivations* a = (NethackMixerActivations*)activations;
    int B = input.shape[0];

    nh_decode_kernel<<<grid_size(B * NH_GRID), BLOCK_SIZE, 0, stream>>>(
        a->glyph_idx.data, input.data, B);
    nhm_gather_kernel<<<grid_size(B * NHM_T * NHM_IN), BLOCK_SIZE, 0, stream>>>(
        a->gath.data, ew->embed_w.data, a->glyph_idx.data, B);

    PrecisionTensor gath = {.data = a->gath.data, .shape = {B * NHM_T, NHM_IN}};
    PrecisionTensor stem = {.data = a->stem_out.data, .shape = {B * NHM_T, NHM_C}};
    puf_mm(&gath, &ew->stem_w, &stem, stream);
    n3_bias_relu_kernel<<<grid_size(B * NHM_T * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->stem_out.data, ew->stem_b.data, B * NHM_T * NHM_C, NHM_C);

    precision_t* flat = a->stem_out.data;   // patch mode: tokens feed concat directly
    if (ew->use_mixer) {
    // token mix (padded token dim)
    nhm_t_kernel<<<grid_size(B * NHM_TP * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->tokT.data, a->stem_out.data, B);
    PrecisionTensor tokT = {.data = a->tokT.data, .shape = {B * NHM_C, NHM_TP}};
    PrecisionTensor tokh = {.data = a->tok_h.data, .shape = {B * NHM_C, NHM_TH}};
    puf_mm(&tokT, &ew->tok_w1, &tokh, stream);
    n3_bias_relu_kernel<<<grid_size(B * NHM_C * NHM_TH), BLOCK_SIZE, 0, stream>>>(
        a->tok_h.data, ew->tok_b1.data, B * NHM_C * NHM_TH, NHM_TH);
    PrecisionTensor toko = {.data = a->tok_out.data, .shape = {B * NHM_C, NHM_TP}};
    puf_mm(&tokh, &ew->tok_w2, &toko, stream);
    nhm_untrans_bias_res_kernel<<<grid_size(B * NHM_T * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->mix1.data, a->tok_out.data, ew->tok_b2.data, a->stem_out.data, B);

    // channel mix
    PrecisionTensor mix1 = {.data = a->mix1.data, .shape = {B * NHM_T, NHM_C}};
    PrecisionTensor chh = {.data = a->ch_h.data, .shape = {B * NHM_T, NHM_CH}};
    puf_mm(&mix1, &ew->ch_w1, &chh, stream);
    n3_bias_relu_kernel<<<grid_size(B * NHM_T * NHM_CH), BLOCK_SIZE, 0, stream>>>(
        a->ch_h.data, ew->ch_b1.data, B * NHM_T * NHM_CH, NHM_CH);
    PrecisionTensor mix2 = {.data = a->mix2.data, .shape = {B * NHM_T, NHM_C}};
    puf_mm(&chh, &ew->ch_w2, &mix2, stream);
    nhm_bias_res_kernel<<<grid_size(B * NHM_T * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->mix2.data, ew->ch_b2.data, a->mix1.data, (int64_t)B * NHM_T * NHM_C);
    flat = a->mix2.data;
    }

    // blstats branch + projection (same shape as the conv encoder)
    nh_blstats_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->bl_feats.data, input.data, B);
    puf_mm(&a->bl_feats, &ew->bl_w, &a->bl_out, stream);
    n3_bias_relu_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_out.data, ew->bl_b.data, B * NH_BL_HID, NH_BL_HID);

    nhm_concat_kernel<<<grid_size(B * NHM_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, flat, a->bl_out.data, a->bl_feats.data, B);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nethack_mixer_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    NethackMixerWeights* ew = (NethackMixerWeights*)w;
    NethackMixerActivations* a = (NethackMixerActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    // fp32 scratch bias grad: zero + accumulate coalesced + cast into bgrad
    auto bias_grad = [&](PrecisionTensor& bgrad, precision_t* g, int64_t rows, int dim) {
        cudaMemsetAsync(a->bias_acc.data, 0, dim * sizeof(float), stream);
        nhm_bias_grad_kernel<<<1024, 256, dim * sizeof(float), stream>>>(
            a->bias_acc.data, g, rows, dim);
        n3_float_to_precision_kernel<<<grid_size(dim), BLOCK_SIZE, 0, stream>>>(
            bgrad.data, a->bias_acc.data, dim);
    };

    // projection
    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * H);
    bias_grad_kernel<<<H, 256, 0, stream>>>(a->proj_bgrad.data, grad.data, B, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, NHM_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    // split
    nh_slice_kernel<<<grid_size(B * NHM_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->g_mix2.data, grad_concat.data, B, NHM_CONCAT, 0, NHM_FLAT);
    nh_slice_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, grad_concat.data, B, NHM_CONCAT, NHM_FLAT, NH_BL_HID);

    // blstats branch
    n3_relu_backward_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, a->bl_out.data, B * NH_BL_HID);
    bias_grad_kernel<<<NH_BL_HID, 256, 0, stream>>>(a->bl_bgrad.data, a->bl_grad.data, B, NH_BL_HID);
    PrecisionTensor blg = {.data = a->bl_grad.data, .shape = {B, NH_BL_HID}};
    puf_mm_tn(&blg, &a->bl_feats, &a->bl_wgrad, stream);

    precision_t* gstem = a->g_mix2.data;   // patch mode: concat grad IS the stem grad
    if (ew->use_mixer) {
    // channel mix backward: mix2 = mix1 + chMLP(mix1)
    PrecisionTensor gmix2 = {.data = a->g_mix2.data, .shape = {B * NHM_T, NHM_C}};
    bias_grad(a->ch_b2g, a->g_mix2.data, (int64_t)B * NHM_T, NHM_C);
    PrecisionTensor chh = {.data = a->ch_h.data, .shape = {B * NHM_T, NHM_CH}};
    puf_mm_tn_splitk(&gmix2, &chh, &a->ch_w2g, stream);
    PrecisionTensor gchh = {.data = a->g_ch_h.data, .shape = {B * NHM_T, NHM_CH}};
    puf_mm_nn(&gmix2, &ew->ch_w2, &gchh, stream);
    n3_relu_backward_kernel<<<grid_size(B * NHM_T * NHM_CH), BLOCK_SIZE, 0, stream>>>(
        a->g_ch_h.data, a->ch_h.data, B * NHM_T * NHM_CH);
    bias_grad(a->ch_b1g, a->g_ch_h.data, (int64_t)B * NHM_T, NHM_CH);
    PrecisionTensor mix1 = {.data = a->mix1.data, .shape = {B * NHM_T, NHM_C}};
    puf_mm_tn_splitk(&gchh, &mix1, &a->ch_w1g, stream);
    PrecisionTensor gmix1 = {.data = a->g_mix1.data, .shape = {B * NHM_T, NHM_C}};
    puf_mm_nn(&gchh, &ew->ch_w1, &gmix1, stream);
    add_kernel<<<grid_size(B * NHM_T * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->g_mix1.data, a->g_mix2.data, B * NHM_T * NHM_C);   // + residual

    // token mix backward: mix1 = stem_out + untrans(tokMLP(trans(stem_out)))
    nhm_t_kernel<<<grid_size(B * NHM_TP * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->tok_out.data, a->g_mix1.data, B);                  // g_tok_out (reuses tok_out)
    PrecisionTensor gtoko = {.data = a->tok_out.data, .shape = {B * NHM_C, NHM_TP}};
    bias_grad(a->tok_b2g, a->tok_out.data, (int64_t)B * NHM_C, NHM_TP);
    PrecisionTensor tokh = {.data = a->tok_h.data, .shape = {B * NHM_C, NHM_TH}};
    puf_mm_tn_splitk(&gtoko, &tokh, &a->tok_w2g, stream);
    PrecisionTensor gtokh = {.data = a->g_tok_h.data, .shape = {B * NHM_C, NHM_TH}};
    puf_mm_nn(&gtoko, &ew->tok_w2, &gtokh, stream);
    n3_relu_backward_kernel<<<grid_size(B * NHM_C * NHM_TH), BLOCK_SIZE, 0, stream>>>(
        a->g_tok_h.data, a->tok_h.data, B * NHM_C * NHM_TH);
    bias_grad(a->tok_b1g, a->g_tok_h.data, (int64_t)B * NHM_C, NHM_TH);
    PrecisionTensor tokT = {.data = a->tokT.data, .shape = {B * NHM_C, NHM_TP}};
    puf_mm_tn_splitk(&gtokh, &tokT, &a->tok_w1g, stream);
    PrecisionTensor gtokT = {.data = a->g_tokT.data, .shape = {B * NHM_C, NHM_TP}};
    puf_mm_nn(&gtokh, &ew->tok_w1, &gtokT, stream);
    nhm_untrans_add_kernel<<<grid_size(B * NHM_T * NHM_C), BLOCK_SIZE, 0, stream>>>(
        a->g_stem.data, a->g_tokT.data, a->g_mix1.data, B);   // + residual
    gstem = a->g_stem.data;
    }

    // stem backward
    n3_relu_backward_kernel<<<grid_size(B * NHM_T * NHM_C), BLOCK_SIZE, 0, stream>>>(
        gstem, a->stem_out.data, B * NHM_T * NHM_C);
    bias_grad(a->stem_bgrad, gstem, (int64_t)B * NHM_T, NHM_C);
    PrecisionTensor gstem_t = {.data = gstem, .shape = {B * NHM_T, NHM_C}};
    PrecisionTensor gath = {.data = a->gath.data, .shape = {B * NHM_T, NHM_IN}};
    puf_mm_tn_splitk(&gstem_t, &gath, &a->stem_wgrad, stream);
    PrecisionTensor ggath = {.data = a->g_gath.data, .shape = {B * NHM_T, NHM_IN}};
    puf_mm_nn(&gstem_t, &ew->stem_w, &ggath, stream);

    // embedding backward (hist + hot-select shared with the conv path)
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
    int embed_n = NH_GLYPH_VOCAB * NHM_D;
    cudaMemsetAsync(a->embed_wgrad_f.data, 0, (size_t)embed_n * sizeof(float), stream);
    nhm_embed_scatter_kernel<<<1024, 256, 0, stream>>>(
        a->embed_wgrad_f.data, a->g_gath.data, a->glyph_idx.data, hot_map, hot_list, hot_n, B);
    n3_float_to_precision_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->embed_wgrad_f.data, embed_n);
}

static void nethack_mixer_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NethackMixerWeights* ew = (NethackMixerWeights*)w;
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->stem_w, 1.0f, (*seed)++, stream);
    if (ew->use_mixer) {
        puf_kaiming_init(&ew->tok_w1, 1.0f, (*seed)++, stream);
        puf_kaiming_init(&ew->ch_w1, 1.0f, (*seed)++, stream);
        // Residual-branch output layers start at zero: the block is the
        // identity at init (their own grads are nonzero, so they train).
        for (PrecisionTensor* t : {&ew->tok_w2, &ew->ch_w2, &ew->tok_b1, &ew->tok_b2,
                                   &ew->ch_b1, &ew->ch_b2})
            cudaMemsetAsync(t->data, 0, numel(t->shape) * sizeof(precision_t), stream);
    }
    puf_kaiming_init(&ew->bl_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->proj_w, 1.0f, (*seed)++, stream);
    for (PrecisionTensor* t : {&ew->stem_b, &ew->bl_b, &ew->proj_b})
        cudaMemsetAsync(t->data, 0, numel(t->shape) * sizeof(precision_t), stream);
}

// Param and grad registration orders must match pairwise (muon walks both flat).
static void nethack_mixer_reg_params(void* w, Allocator* alloc) {
    NethackMixerWeights* ew = (NethackMixerWeights*)w;
    ew->embed_w = {.shape = {NH_GLYPH_VOCAB, NHM_D}};
    ew->stem_w  = {.shape = {NHM_C, NHM_IN}};
    ew->stem_b  = {.shape = {NHM_C}};
    ew->tok_w1  = {.shape = {NHM_TH, NHM_TP}};
    ew->tok_b1  = {.shape = {NHM_TH}};
    ew->tok_w2  = {.shape = {NHM_TP, NHM_TH}};
    ew->tok_b2  = {.shape = {NHM_TP}};
    ew->ch_w1   = {.shape = {NHM_CH, NHM_C}};
    ew->ch_b1   = {.shape = {NHM_CH}};
    ew->ch_w2   = {.shape = {NHM_C, NHM_CH}};
    ew->ch_b2   = {.shape = {NHM_C}};
    ew->bl_w    = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    ew->bl_b    = {.shape = {NH_BL_HID}};
    ew->proj_w  = {.shape = {ew->hidden, NHM_CONCAT}};
    ew->proj_b  = {.shape = {ew->hidden}};
    for (PrecisionTensor* t : {&ew->embed_w, &ew->stem_w, &ew->stem_b})
        alloc_register(alloc, t);
    if (ew->use_mixer)
        for (PrecisionTensor* t : {&ew->tok_w1, &ew->tok_b1, &ew->tok_w2, &ew->tok_b2,
                                   &ew->ch_w1, &ew->ch_b1, &ew->ch_w2, &ew->ch_b2})
            alloc_register(alloc, t);
    for (PrecisionTensor* t : {&ew->bl_w, &ew->bl_b, &ew->proj_w, &ew->proj_b})
        alloc_register(alloc, t);
}

static void nethack_mixer_reg_common(NethackMixerWeights* ew, NethackMixerActivations* a, Allocator* acts, int B) {
    a->glyph_idx = {.shape = {B, NH_GRID}};
    a->gath      = {.shape = {B * NHM_T, NHM_IN}};
    a->stem_out  = {.shape = {B * NHM_T, NHM_C}};
    a->tokT      = {.shape = {B * NHM_C, NHM_TP}};
    a->tok_h     = {.shape = {B * NHM_C, NHM_TH}};
    a->tok_out   = {.shape = {B * NHM_C, NHM_TP}};
    a->mix1      = {.shape = {B * NHM_T, NHM_C}};
    a->ch_h      = {.shape = {B * NHM_T, NHM_CH}};
    a->mix2      = {.shape = {B * NHM_T, NHM_C}};
    a->bl_feats  = {.shape = {B, NH_BL_FEAT}};
    a->bl_out    = {.shape = {B, NH_BL_HID}};
    a->concat    = {.shape = {B, NHM_CONCAT}};
    a->out       = {.shape = {B, ew->hidden}};
    for (auto* t : {&a->glyph_idx}) alloc_register(acts, t);
    for (PrecisionTensor* t : {&a->gath, &a->stem_out, &a->bl_feats, &a->bl_out,
                               &a->concat, &a->out})
        alloc_register(acts, t);
    if (ew->use_mixer)
        for (PrecisionTensor* t : {&a->tokT, &a->tok_h, &a->tok_out, &a->mix1, &a->ch_h, &a->mix2})
            alloc_register(acts, t);
}

static void nethack_mixer_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NethackMixerWeights* ew = (NethackMixerWeights*)w;
    NethackMixerActivations* a = (NethackMixerActivations*)activations;
    *a = {};
    nethack_mixer_reg_common(ew, a, acts, B_TT);
    a->sort_buf      = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_T + 1}};
    a->embed_wgrad_f = {.shape = {NH_GLYPH_VOCAB, NHM_D}};
    a->bias_acc      = {.shape = {NHM_CH}};
    a->bl_grad       = {.shape = {B_TT, NH_BL_HID}};
    a->g_mix2        = {.shape = {B_TT * NHM_T, NHM_C}};
    a->g_ch_h        = {.shape = {B_TT * NHM_T, NHM_CH}};
    a->g_mix1        = {.shape = {B_TT * NHM_T, NHM_C}};
    a->g_tok_h       = {.shape = {B_TT * NHM_C, NHM_TH}};
    a->g_tokT        = {.shape = {B_TT * NHM_C, NHM_TP}};
    a->g_stem        = {.shape = {B_TT * NHM_T, NHM_C}};
    a->g_gath        = {.shape = {B_TT * NHM_T, NHM_IN}};
    alloc_register(acts, &a->sort_buf);
    alloc_register(acts, &a->embed_wgrad_f);
    alloc_register(acts, &a->bias_acc);
    for (PrecisionTensor* t : {&a->bl_grad, &a->g_mix2, &a->g_gath})
        alloc_register(acts, t);
    if (ew->use_mixer)
        for (PrecisionTensor* t : {&a->g_ch_h, &a->g_mix1, &a->g_tok_h, &a->g_tokT, &a->g_stem})
            alloc_register(acts, t);
    a->embed_wgrad = {.shape = {NH_GLYPH_VOCAB, NHM_D}};
    a->stem_wgrad  = {.shape = {NHM_C, NHM_IN}};
    a->stem_bgrad  = {.shape = {NHM_C}};
    a->tok_w1g     = {.shape = {NHM_TH, NHM_TP}};
    a->tok_b1g     = {.shape = {NHM_TH}};
    a->tok_w2g     = {.shape = {NHM_TP, NHM_TH}};
    a->tok_b2g     = {.shape = {NHM_TP}};
    a->ch_w1g      = {.shape = {NHM_CH, NHM_C}};
    a->ch_b1g      = {.shape = {NHM_CH}};
    a->ch_w2g      = {.shape = {NHM_C, NHM_CH}};
    a->ch_b2g      = {.shape = {NHM_C}};
    a->bl_wgrad    = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    a->bl_bgrad    = {.shape = {NH_BL_HID}};
    a->proj_wgrad  = {.shape = {ew->hidden, NHM_CONCAT}};
    a->proj_bgrad  = {.shape = {ew->hidden}};
    for (PrecisionTensor* t : {&a->embed_wgrad, &a->stem_wgrad, &a->stem_bgrad})
        alloc_register(grads, t);
    if (ew->use_mixer)
        for (PrecisionTensor* t : {&a->tok_w1g, &a->tok_b1g, &a->tok_w2g, &a->tok_b2g,
                                   &a->ch_w1g, &a->ch_b1g, &a->ch_w2g, &a->ch_b2g})
            alloc_register(grads, t);
    for (PrecisionTensor* t : {&a->bl_wgrad, &a->bl_bgrad, &a->proj_wgrad, &a->proj_bgrad})
        alloc_register(grads, t);
}

static void nethack_mixer_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NethackMixerWeights* ew = (NethackMixerWeights*)w;
    NethackMixerActivations* a = (NethackMixerActivations*)activations;
    nethack_mixer_reg_common(ew, a, alloc, B);
}

static void* nethack_mixer_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    if (e->in_dim != NH_OBS_SIZE) {
        fprintf(stderr, "nethack mixer encoder: obs size %d != expected %d\n", e->in_dim, NH_OBS_SIZE);
        exit(1);
    }
    NethackMixerWeights* ew = (NethackMixerWeights*)calloc(1, sizeof(NethackMixerWeights));
    ew->obs_size = e->in_dim; ew->hidden = e->out_dim;
    const char* kind = getenv("NETHACK_ENCODER");
    ew->use_mixer = !(kind && std::string(kind) == "patch");
    return ew;
}
static void nethack_mixer_free_weights(void* weights) { free(weights); }
static void nethack_mixer_free_activations(void* activations) { free(activations); }

// Override encoder vtable for known ocean environments. No-op for unknown envs.
static void create_custom_encoder(const std::string& env_name, Encoder* enc) {
    if (env_name == "nmmo3") {
        *enc = Encoder{
            .forward = nmmo3_encoder_forward,
            .backward = nmmo3_encoder_backward,
            .init_weights = nmmo3_encoder_init_weights,
            .reg_params = nmmo3_encoder_reg_params,
            .reg_train = nmmo3_encoder_reg_train,
            .reg_rollout = nmmo3_encoder_reg_rollout,
            .create_weights = nmmo3_encoder_create_weights,
            .free_weights = nmmo3_encoder_free_weights,
            .free_activations = nmmo3_encoder_free_activations,
            .in_dim = enc->in_dim, .out_dim = enc->out_dim,
            .activation_size = sizeof(NMMO3EncoderActivations),
        };
    } else if (env_name == "nethack") {
        const char* kind = getenv("NETHACK_ENCODER");
        if (kind && (std::string(kind) == "mixer" || std::string(kind) == "patch")) {
            *enc = Encoder{
                .forward = nethack_mixer_forward,
                .backward = nethack_mixer_backward,
                .init_weights = nethack_mixer_init_weights,
                .reg_params = nethack_mixer_reg_params,
                .reg_train = nethack_mixer_reg_train,
                .reg_rollout = nethack_mixer_reg_rollout,
                .create_weights = nethack_mixer_create_weights,
                .free_weights = nethack_mixer_free_weights,
                .free_activations = nethack_mixer_free_activations,
                .in_dim = enc->in_dim, .out_dim = enc->out_dim,
                .activation_size = sizeof(NethackMixerActivations),
            };
        } else {
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
    }
}
