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

__global__ void nh_unpermute_w_kernel(
    precision_t* __restrict__ wg, const precision_t* __restrict__ wpg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int t = r / NH_C1_OC, oc = r % NH_C1_OC;
    wg[oc * (NH_EMBED_DIM * NH_TAPS) + d * NH_TAPS + t] = wpg[i];
}

// Fused embed+conv1: out[b,oc,p] = relu(bias[oc] + sum_taps T[g_tap, tap*OC+oc])
__global__ void nh_fused_conv1_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ T,
    const precision_t* __restrict__ bias, const float* __restrict__ idx, int B) {
    constexpr int spatial = NH_C1_OH * NH_C1_OW;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * spatial * NH_C1_OC) return;
    int oc = i % NH_C1_OC;
    int p  = (i / NH_C1_OC) % spatial;
    int b  = i / (NH_C1_OC * spatial);
    int oh = p / NH_C1_OW, ow = p % NH_C1_OW;
    const float* gi = idx + (int64_t)b * NH_GRID;
    float acc = to_float(bias[oc]);
    #pragma unroll
    for (int t = 0; t < NH_TAPS; t++) {
        int cell = (NH_C1_S * oh + t / NH_C1_K) * NH_MAP + NH_C1_S * ow + t % NH_C1_K;
        acc += to_float(T[(int64_t)(int)gi[cell] * NH_TROW + t * NH_C1_OC + oc]);
    }
    out[(int64_t)b * NH_C1_OC * spatial + oc * spatial + p] = from_float(fmaxf(acc, 0.0f));
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

// concat = [conv2 flat (NCHW, contiguous per sample) | bl hidden | bl raw feats]
__global__ void nh_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ conv_flat,
    const precision_t* __restrict__ bl_out, const precision_t* __restrict__ bl_feats, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_CONCAT) return;
    int b = idx / NH_CONCAT, c = idx % NH_CONCAT;
    precision_t val;
    if (c < NH_CONV_FLAT)
        val = conv_flat[b * NH_CONV_FLAT + c];
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

// Fused backward: scatter conv1's output grad into the fp32 table grad.
// dT[g, tap*OC+oc] += grad[b,oc,p] for every tap under every output position.
// Chain rule to dE and dW then happens as two small dense GEMMs on dT.
__global__ void nh_dT_scatter_kernel(
    float* __restrict__ dT_f, const precision_t* __restrict__ grad,
    const float* __restrict__ idx, int B) {
    constexpr int spatial = NH_C1_OH * NH_C1_OW;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * spatial * NH_C1_OC) return;
    int oc = i % NH_C1_OC;
    int p  = (i / NH_C1_OC) % spatial;
    int b  = i / (NH_C1_OC * spatial);
    float g = to_float(grad[(int64_t)b * NH_C1_OC * spatial + oc * spatial + p]);
    if (g == 0.0f) return;  // relu-masked: skip all 25 atomics
    int oh = p / NH_C1_OW, ow = p % NH_C1_OW;
    const float* gi = idx + (int64_t)b * NH_GRID;
    #pragma unroll
    for (int t = 0; t < NH_TAPS; t++) {
        int cell = (NH_C1_S * oh + t / NH_C1_K) * NH_MAP + NH_C1_S * ow + t % NH_C1_K;
        atomicAdd(&dT_f[(int64_t)(int)gi[cell] * NH_TROW + t * NH_C1_OC + oc], g);
    }
}

// ---- Nethack encoder structs ----

struct NethackEncoderWeights {
    ConvWeights conv1, conv2;
    PrecisionTensor embed_w, bl_w, bl_b, proj_w, proj_b;
    int obs_size, hidden;
};

struct NethackEncoderActivations {
    ConvActivations conv1, conv2;
    PrecisionTensor col2, mm2;             // conv2 im2col scratch
    PrecisionTensor w_perm, glyph_T;       // fused embed+conv1 table (T = E @ W'^T)
    PrecisionTensor dT, dw_perm;           // fused backward scratch (train)
    FloatTensor dT_f;                      // fp32 scatter buffer (train)
    FloatTensor glyph_idx;                 // decoded glyph ids
    PrecisionTensor bl_feats, bl_out, bl_grad, concat, out;
    PrecisionTensor embed_wgrad, bl_wgrad, bl_bgrad, proj_wgrad, proj_bgrad;
};

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
    nh_fused_conv1_kernel<<<grid_size(B * NH_C1_OH * NH_C1_OW * NH_C1_OC), BLOCK_SIZE, 0, stream>>>(
        a->conv1.out.data, a->glyph_T.data, ew->conv1.b.data, a->glyph_idx.data, B);
    gemm_conv_forward(&ew->conv2.w, &ew->conv2.b, a->conv1.out.data, a->conv2.out.data,
        a->col2.data, a->mm2.data, B, NH_C2_IC, NH_C1_OH, NH_C1_OW,
        NH_C2_OC, NH_C2_K, NH_C2_S, NH_C2_OH, NH_C2_OW, false, stream);

    nh_blstats_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->bl_feats.data, input.data, B);
    puf_mm(&a->bl_feats, &ew->bl_w, &a->bl_out, stream);
    n3_bias_relu_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_out.data, ew->bl_b.data, B * NH_BL_HID, NH_BL_HID);

    nh_concat_kernel<<<grid_size(B * NH_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2.out.data, a->bl_out.data, a->bl_feats.data, B);
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

    // Conv branch: concat slice -> conv2 -> conv1 -> embedding map grad
    nh_slice_kernel<<<grid_size(B * NH_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv2.grad.data, grad_concat.data, B, NH_CONCAT, 0, NH_CONV_FLAT);
    n3_conv_bias_grad_nchw<<<NH_C2_OC, 256, 0, stream>>>(
        a->conv2.bgrad.data, a->conv2.grad.data, B, NH_C2_OC, NH_C2_OH * NH_C2_OW);
    gemm_conv_backward(&ew->conv2.w, a->conv1.out.data, a->conv2.grad.data,
        a->conv2.wgrad.data, a->conv1.grad.data,
        a->col2.data, a->mm2.data, B, NH_C2_IC, NH_C1_OH, NH_C1_OW,
        NH_C2_OC, NH_C2_K, NH_C2_S, NH_C2_OH, NH_C2_OW, stream);
    n3_relu_backward_kernel<<<grid_size(B * NH_C1_OC * NH_C1_OH * NH_C1_OW), BLOCK_SIZE, 0, stream>>>(
        a->conv1.grad.data, a->conv1.out.data, B * NH_C1_OC * NH_C1_OH * NH_C1_OW);
    n3_conv_bias_grad_nchw<<<NH_C1_OC, 256, 0, stream>>>(
        a->conv1.bgrad.data, a->conv1.grad.data, B, NH_C1_OC, NH_C1_OH * NH_C1_OW);
    // Fused conv1+embedding backward: scatter into dT, then split via GEMMs
    int dT_n = NH_GLYPH_VOCAB * NH_TROW;
    cudaMemsetAsync(a->dT_f.data, 0, (size_t)dT_n * sizeof(float), stream);
    nh_dT_scatter_kernel<<<grid_size(B * NH_C1_OH * NH_C1_OW * NH_C1_OC), BLOCK_SIZE, 0, stream>>>(
        a->dT_f.data, a->conv1.grad.data, a->glyph_idx.data, B);
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
    alloc_register(acts,&a->w_perm); alloc_register(acts,&a->glyph_T);
    alloc_register(acts,&a->dT);     alloc_register(acts,&a->dT_f);
    alloc_register(acts,&a->dw_perm);
    // Conv2
    a->conv2.out   = {.shape = {B_TT * NH_C2_OC * NH_C2_OH * NH_C2_OW}};
    a->conv2.grad  = {.shape = {B_TT * NH_C2_OC * NH_C2_OH * NH_C2_OW}};
    a->conv2.wgrad = {.shape = {NH_C2_OC, NH_C2_IC * NH_C2_K * NH_C2_K}};
    a->conv2.bgrad = {.shape = {NH_C2_OC}};
    alloc_register(acts,&a->conv2.out); alloc_register(acts,&a->conv2.grad);
    alloc_register(grads,&a->conv2.wgrad); alloc_register(grads,&a->conv2.bgrad);
    a->col2 = {.shape = {B_TT * NH_C2_OH * NH_C2_OW, NH_C2_IC * NH_C2_K * NH_C2_K}};
    a->mm2  = {.shape = {B_TT * NH_C2_OH * NH_C2_OW, NH_C2_OC}};
    alloc_register(acts,&a->col2); alloc_register(acts,&a->mm2);
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
    alloc_register(alloc,&a->conv2.out);
    a->col2 = {.shape = {B * NH_C2_OH * NH_C2_OW, NH_C2_IC * NH_C2_K * NH_C2_K}};
    a->mm2  = {.shape = {B * NH_C2_OH * NH_C2_OW, NH_C2_OC}};
    alloc_register(alloc,&a->col2); alloc_register(alloc,&a->mm2);
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
