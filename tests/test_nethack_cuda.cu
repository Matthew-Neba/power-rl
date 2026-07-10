// Test harness for the Nethack encoder — thin wrapper around nethack.cu's real
// implementation. Built as a float (PRECISION_FLOAT) shared lib so finite-diff
// gradient checking is numerically meaningful.
//
// Build:
//   nvcc -shared -o nethack_test.so tests/test_nethack_cuda.cu -I src \
//        -lcublas -lcudnn -lcurand -Xcompiler -fPIC -O2
#define PRECISION_FLOAT
#include <string>
#include <cstdio>
#include "../src/models.cu"
#include "../src/cudnn_conv2d.cu"
#include "../src/nethack.cu"

extern "C" {

static Encoder g_enc;
static NethackEncoderWeights* g_w = nullptr;
static NethackEncoderActivations* g_a = nullptr;
static Allocator g_pa = {}, g_aa = {}, g_ga = {};
static int g_hidden = 32;

void nh_init(int B, int hidden) {
    g_hidden = hidden;
    g_enc = {};
    g_enc.in_dim = NH_OBS_SIZE;
    g_enc.out_dim = hidden;
    create_nethack_encoder(&g_enc);
    g_w = (NethackEncoderWeights*)g_enc.create_weights(&g_enc);
    g_pa = {};
    g_enc.reg_params(g_w, &g_pa);
    alloc_create(&g_pa);
    g_a = (NethackEncoderActivations*)calloc(1, sizeof(NethackEncoderActivations));
    g_aa = {}; g_ga = {};
    g_enc.reg_train(g_w, g_a, &g_aa, &g_ga, B);
    alloc_create(&g_aa);
    alloc_create(&g_ga);
    uint64_t seed = 1234;
    g_enc.init_weights(g_w, &seed, 0);
    cudaDeviceSynchronize();
}

int nh_obs_size()    { return NH_OBS_SIZE; }
int nh_bl_feat()     { return NH_BL_FEAT; }
int nh_glyph_vocab() { return NH_GLYPH_VOCAB; }
int nh_embed_dim()   { return NH_EMBED_DIM; }
int nh_concat()      { return NH_CONCAT; }
int nh_grid()        { return NH_GRID; }

void nh_forward(void* out, void* obs, int B) {
    PrecisionTensor in = {.data = (precision_t*)obs, .shape = {B, NH_OBS_SIZE}};
    PrecisionTensor r = g_enc.forward(g_w, g_a, in, 0);
    cudaMemcpy(out, r.data, (size_t)B * g_hidden * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void nh_backward(void* grad, int B) {
    PrecisionTensor g = {.data = (precision_t*)grad, .shape = {B, g_hidden}};
    g_enc.backward(g_w, g_a, g, 0);
    cudaDeviceSynchronize();
}

// value get/set + grad get for each learnable tensor (all device float ptrs)
#define TENSOR_ACC(name, field) \
    void nh_get_##name(void* dst) { cudaMemcpy(dst, g_w->field.data, numel(g_w->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); } \
    void nh_set_##name(void* src) { cudaMemcpy(g_w->field.data, src, numel(g_w->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); cudaDeviceSynchronize(); } \
    int  nh_numel_##name()        { return (int)numel(g_w->field.shape); }
TENSOR_ACC(embed_w, embed_w)
TENSOR_ACC(bl_w,    bl_w)
TENSOR_ACC(bl_b,    bl_b)
TENSOR_ACC(proj_w,  proj_w)
TENSOR_ACC(proj_b,  proj_b)
TENSOR_ACC(conv1_w, conv1.w)
TENSOR_ACC(conv2_w, conv2.w)

#define GRAD_ACC(name, field) \
    void nh_grad_##name(void* dst) { cudaMemcpy(dst, g_a->field.data, numel(g_a->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); }
GRAD_ACC(embed_w, embed_wgrad)
GRAD_ACC(bl_w,    bl_wgrad)
GRAD_ACC(bl_b,    bl_bgrad)
GRAD_ACC(proj_w,  proj_wgrad)
GRAD_ACC(proj_b,  proj_bgrad)
GRAD_ACC(conv1_w, conv1.wgrad)
GRAD_ACC(conv2_w, conv2.wgrad)

}  // extern "C"
