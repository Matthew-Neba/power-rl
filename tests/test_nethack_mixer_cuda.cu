// Test harness for the Nethack Mixer/Patch encoder — thin wrapper around ocean.cu.
// Built as a float (PRECISION_FLOAT) shared lib so finite-diff gradient
// checking is numerically meaningful. See test_nethack_mixer.py.
#define PRECISION_FLOAT
#include <string>
#include <cstdio>
#include <cstdlib>
#include "../src/models.cu"
#include "../src/ocean.cu"

extern "C" {

static Encoder g_enc;
static NethackMixerWeights* g_w = nullptr;
static NethackMixerActivations* g_a = nullptr;
static Allocator g_pa = {}, g_aa = {}, g_ga = {};
static int g_hidden = 32;

void nhm_init(int B, int hidden, int mode, int C, int D) {
    const char* kind = mode == 1 ? "mixer" : mode == 2 ? "minpatch"
                 : mode == 3 ? "cellflat" : "patch";
    setenv("NETHACK_ENCODER", kind, 1);
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", C); setenv("NETHACK_ENCODER_C", buf, 1);
    snprintf(buf, sizeof(buf), "%d", D); setenv("NETHACK_ENCODER_D", buf, 1);
    g_hidden = hidden;
    g_enc = {};
    g_enc.in_dim = NH_OBS_SIZE;
    g_enc.out_dim = hidden;
    create_custom_encoder("nethack", &g_enc);
    g_w = (NethackMixerWeights*)g_enc.create_weights(&g_enc);
    g_pa = {};
    g_enc.reg_params(g_w, &g_pa);
    alloc_create(&g_pa);
    g_a = (NethackMixerActivations*)calloc(1, sizeof(NethackMixerActivations));
    g_aa = {}; g_ga = {};
    g_enc.reg_train(g_w, g_a, &g_aa, &g_ga, B);
    alloc_create(&g_aa);
    alloc_create(&g_ga);
    uint64_t seed = 1234;
    g_enc.init_weights(g_w, &seed, 0);
    cudaDeviceSynchronize();
}

int nhm_obs_size()  { return NH_OBS_SIZE; }
int nhm_grid()      { return NH_GRID; }
int nhm_vocab()     { return NH_GLYPH_VOCAB; }
int nhm_embed_dim() { return NHM_D; }

void nhm_forward(void* out, void* obs, int B) {
    PrecisionTensor in = {.data = (precision_t*)obs, .shape = {B, NH_OBS_SIZE}};
    PrecisionTensor r = g_enc.forward(g_w, g_a, in, 0);
    cudaMemcpy(out, r.data, (size_t)B * g_hidden * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void nhm_backward(void* grad, int B) {
    PrecisionTensor g = {.data = (precision_t*)grad, .shape = {B, g_hidden}};
    g_enc.backward(g_w, g_a, g, 0);
    cudaDeviceSynchronize();
}

#define TENSOR_ACC(name, wfield, gfield) \
    void nhm_get_##name(void* dst) { cudaMemcpy(dst, g_w->wfield.data, numel(g_w->wfield.shape) * sizeof(float), cudaMemcpyDeviceToDevice); } \
    void nhm_set_##name(void* src) { cudaMemcpy(g_w->wfield.data, src, numel(g_w->wfield.shape) * sizeof(float), cudaMemcpyDeviceToDevice); cudaDeviceSynchronize(); } \
    void nhm_grad_##name(void* dst) { cudaMemcpy(dst, g_a->gfield.data, numel(g_a->gfield.shape) * sizeof(float), cudaMemcpyDeviceToDevice); } \
    int  nhm_numel_##name()        { return (int)numel(g_w->wfield.shape); }
TENSOR_ACC(embed_w, embed_w, embed_wgrad)
TENSOR_ACC(stem_w,  stem_w,  stem_wgrad)
TENSOR_ACC(stem_b,  stem_b,  stem_bgrad)
TENSOR_ACC(tok_w1,  tok_w1,  tok_w1g)
TENSOR_ACC(tok_b1,  tok_b1,  tok_b1g)
TENSOR_ACC(tok_w2,  tok_w2,  tok_w2g)
TENSOR_ACC(tok_b2,  tok_b2,  tok_b2g)
TENSOR_ACC(ch_w1,   ch_w1,   ch_w1g)
TENSOR_ACC(ch_b1,   ch_b1,   ch_b1g)
TENSOR_ACC(ch_w2,   ch_w2,   ch_w2g)
TENSOR_ACC(ch_b2,   ch_b2,   ch_b2g)
TENSOR_ACC(pool_w,  pool_w,  pool_wgrad)
TENSOR_ACC(bl_w,    bl_w,    bl_wgrad)
TENSOR_ACC(bl_b,    bl_b,    bl_bgrad)
TENSOR_ACC(proj_w,  proj_w,  proj_wgrad)
TENSOR_ACC(proj_b,  proj_b,  proj_bgrad)

}  // extern "C"
