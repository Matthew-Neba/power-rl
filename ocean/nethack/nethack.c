#include <time.h>
#include <unistd.h>
#include <string.h>
#include "nethack.h"
#include "../../src/puffernet.h"
#include "../../src/nethack_glyph_map.h"

// labels for the end-of-run histogram, in NETHACK_ACTION_TABLE order
static const char* NETHACK_ACTION_NAMES[NETHACK_NUM_ACTIONS] = {
    "MOVE","RUN","DOWN_STAIRS","UP_STAIRS","KICK","SEARCH",
    "ELBERETH","WEAR","EAT","QUAFF","PRAY","THROW","ZAP","REST",
};

// single-agent env, reset immediately (training's c_reset is lazy)
static void env_open(Nethack* env) {
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env->actions      = (float*)calloc(7, sizeof(float));   // {verb, 5 per-verb slots, direction}
    env->action_mask  = (unsigned char*)calloc(NETHACK_NUM_ACTIONS
                        + 5 * NETHACK_INV_SLOTS + NETHACK_NUM_DIRS, 1);
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (float*)calloc(1, sizeof(float));
    init(env);
    nethack_do_reset(env);
}

static void env_close(Nethack* env) {
    c_close(env);
    free(env->observations); free(env->actions); free(env->rewards); free(env->terminals);
    free(env->action_mask);
}

// CPU port of the CUDA encoder (src/nethack.cu) + puffernet MinGRU/decoder;
// weight order matches param registration: encoder, decoder, mingru
#define DEMO_VOCAB   5977
#define DEMO_EMBED   32
#define DEMO_BL_FEAT (25 + 7 + 13 + 1 + NETHACK_NUM_ACTIONS + NETHACK_NUM_OCLASSES)
#define DEMO_INV_HID 16   // 16-dim slot rep: pool bottleneck + decoder key (unified)
#define DEMO_INV_FLAT (NETHACK_INV_SLOTS * DEMO_INV_HID)
#define DEMO_INV_POOL 128
#define DEMO_SFEAT 17    // buc4 + known+spe + quan + ero2 + flags7 + tk
#define DEMO_OD (NETHACK_NUM_ACTIONS + 5 * NETHACK_INV_SLOTS + NETHACK_NUM_DIRS)
#define DEMO_NUM_HEADS 7
#define DEMO_PTR_HEADS 5
#define DEMO_QDIM (DEMO_PTR_HEADS * DEMO_INV_HID)
#define DEMO_DEC_PAD 24
#define DEMO_DEC_LIN (NETHACK_NUM_ACTIONS + NETHACK_NUM_DIRS + 1)
#define DEMO_LOC_IN  (NETHACK_CROP_GRID * DEMO_EMBED)   // 9x9 crop, per-cell embeds
#define DEMO_LOC_HID 256
#define DEMO_PW 5
#define DEMO_PH 5
#define DEMO_PX 16
#define DEMO_PY 5
#define DEMO_TOK     (DEMO_PX * DEMO_PY)                // 5x5 patches over 79x21
#define DEMO_PCELLS  (DEMO_PW * DEMO_PH)                // off-map cells read the pad glyph
#define DEMO_P1      16
#define DEMO_GLB_IN  (DEMO_PCELLS * DEMO_EMBED)         // per-patch flatten (glyph slice)
#define DEMO_GLB_HID 128
// trigram message branch, mirroring NH_MSG_* in src/nethack.cu
#define DEMO_MSG_LEN   NETHACK_MSG_LEN
#define DEMO_MSG_VOCAB 4096
#define DEMO_MSG_LOG2V 12
#define DEMO_MSG_HID   32
#define DEMO_MSG_CONCAT_OFF (DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL + 64 + DEMO_BL_FEAT)
#define DEMO_CONCAT  (DEMO_MSG_CONCAT_OFF + DEMO_MSG_HID)

// per-blstat normalization, mirroring NH_BL_SCALE / NH_BL_ISLOG in src/nethack.cu
static const float DEMO_BL_SCALE[27] = {
    1.f/79, 1.f/21,
    1.f/25, 1.f/125, 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
    0.1f, 1.f/200, 1.f/200, 1.f/50, 0.1f,
    1.f/100, 1.f/100, 1.f/10, 1.f/10, 1.f/30,
    0.1f, 0.1f, 0.f, 1.f/4, 1.f/10, 1.f/50, 0.f, 1.f,
};
static const int DEMO_BL_ISLOG[27] =
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0};

typedef struct {
    float *embed;               // (5977, 32) E_res
    float *ekind_w, *esub_w;    // (14, 32), (944, 32) factor tables
    float *e_eff;               // materialized E_res + E_kind + E_sub
    float *loc_w, *loc_b;       // (256, 2592), (256)
    float *g1_w, *g1_xy, *g1_b; // (16, 800), (16, 2), (16): per-patch embed+flatten + hero dx,dy -> 16
    float *g2_w, *g2_b;         // (128, 16), (128): 16 -> 128, maxed over tokens
    float *inv1_w, *inv1_b;     // (32, 32), (32): per-slot features (pointer keys)
    float *inv1s_w;             // (32, 17): gated item-state path into the slot MLP
    float *inv2_w, *inv2_b;     // (128, 32), (128): pooled trunk summary (max over slots)
    float *bl_w, *bl_b;         // (64, DEMO_BL_FEAT), (64)
    float *proj_w, *proj_b;     // (H, DEMO_CONCAT), (H)
    float *msg_w;               // (4096, 32) trigram embedding table
    float *dec_lin;             // (24, H) bias-free; rows [14 verb | 8 dir | value], 23 used
    float *dec_q;               // (160, H): five stacked 32-dim query projections
    float *dec_k;               // (32, 32): key projection over slot features
    float *dec_tau;             // (5,): per-head log cosine temperature
    MinGRU* mingru;
    Multidiscrete* md;
    int hidden_size, num_layers, num_actions;
    float x[DEMO_LOC_IN];       // crop cell embeds, flattened
    float px[DEMO_GLB_IN];      // one patch's cell embeds, flattened
    float t16[DEMO_P1];
    float t128[DEMO_GLB_HID];
    float slots[DEMO_INV_FLAT]; // per-slot post-relu features (decoder keys)
    float concat[DEMO_CONCAT];  // [local hid | global hid | inv pool | bl hidden | bl feats | msg]
    float logits[DEMO_OD + 1];  // assembled decoder output; last entry is value
    float* hidden;              // (hidden_size)
} NethackNet;

// (hidden, layers) from the checkpoint float count:
//   total = ENC_FIXED + H*(DEMO_CONCAT + 1) + H*(24 + 160) + DEC_FIXED + L * 3*H*H
// All tensors land on 8-float boundaries; only tau (5) needs padding (+3).
#define DEMO_ENC_FIXED (DEMO_VOCAB*DEMO_EMBED \
                        + NH_GM_NKIND*DEMO_EMBED + NH_GM_NSUB*DEMO_EMBED \
                        + DEMO_LOC_HID*DEMO_LOC_IN + DEMO_LOC_HID \
                        + DEMO_P1*DEMO_GLB_IN + DEMO_P1*2 + DEMO_P1 \
                        + DEMO_GLB_HID*DEMO_P1 + DEMO_GLB_HID \
                        + DEMO_INV_HID*DEMO_EMBED + DEMO_INV_HID \
                        + DEMO_INV_HID*DEMO_SFEAT \
                        + DEMO_INV_POOL*DEMO_INV_HID + DEMO_INV_POOL \
                        + 64*DEMO_BL_FEAT + 64 \
                        + DEMO_MSG_VOCAB*DEMO_MSG_HID)
#define DEMO_DEC_FIXED (DEMO_INV_HID*DEMO_INV_HID + 8)   // k_w + tau padded 5->8
// ambiguities are possible; prefer the fewest layers (real configs have <= 8)
static int demo_infer_arch(int total, int* hidden, int* layers, int* actions) {
    int best_l = 1 << 30;
    for (int H = 8; H <= 4096; H += 8) {
        long rem = (long)total - DEMO_ENC_FIXED - DEMO_DEC_FIXED
                 - (long)H * (DEMO_CONCAT + 1 + DEMO_DEC_PAD + DEMO_QDIM);
        long per_layer = 3L * H * H;
        if (rem <= 0) break;
        if (rem % per_layer) continue;
        long L = rem / per_layer;
        if (L >= 1 && L < best_l) { best_l = (int)L; *hidden = H; *layers = (int)L; *actions = NETHACK_NUM_ACTIONS; }
    }
    return best_l == 1 << 30 ? -1 : 0;
}

static NethackNet* make_nethack_net(Weights* w) {
    NethackNet* net = (NethackNet*)calloc(1, sizeof(NethackNet));
    if (demo_infer_arch(w->size - 7, &net->hidden_size, &net->num_layers, &net->num_actions) != 0) {
        fprintf(stderr, "nethack demo: cannot infer arch from %d floats — "
                "checkpoint is not a nethack policy with %d actions?\n",
                w->size - 7, NETHACK_NUM_ACTIONS);
        exit(1);
    }
    fprintf(stderr, "nethack demo: hidden=%d layers=%d actions=%d (%d floats)\n",
            net->hidden_size, net->num_layers, net->num_actions, w->size - 7);
    net->hidden = (float*)calloc(net->hidden_size, sizeof(float));
    net->embed   = get_weights_aligned(w, DEMO_VOCAB * DEMO_EMBED);
    net->ekind_w = get_weights_aligned(w, NH_GM_NKIND * DEMO_EMBED);
    net->esub_w  = get_weights_aligned(w, NH_GM_NSUB * DEMO_EMBED);
    net->loc_w   = get_weights_aligned(w, DEMO_LOC_HID * DEMO_LOC_IN);
    net->loc_b   = get_weights_aligned(w, DEMO_LOC_HID);
    net->g1_w    = get_weights_aligned(w, DEMO_P1 * DEMO_GLB_IN);
    net->g1_xy   = get_weights_aligned(w, DEMO_P1 * 2);
    net->g1_b    = get_weights_aligned(w, DEMO_P1);
    net->g2_w    = get_weights_aligned(w, DEMO_GLB_HID * DEMO_P1);
    net->g2_b    = get_weights_aligned(w, DEMO_GLB_HID);
    net->inv1_w  = get_weights_aligned(w, DEMO_INV_HID * DEMO_EMBED);
    net->inv1_b  = get_weights_aligned(w, DEMO_INV_HID);
    net->inv1s_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_SFEAT);
    net->inv2_w  = get_weights_aligned(w, DEMO_INV_POOL * DEMO_INV_HID);
    net->inv2_b  = get_weights_aligned(w, DEMO_INV_POOL);
    net->bl_w    = get_weights_aligned(w, 64 * DEMO_BL_FEAT);
    net->bl_b    = get_weights_aligned(w, 64);
    net->proj_w  = get_weights_aligned(w, net->hidden_size * DEMO_CONCAT);
    net->proj_b  = get_weights_aligned(w, net->hidden_size);
    net->msg_w   = get_weights_aligned(w, DEMO_MSG_VOCAB * DEMO_MSG_HID);
    net->dec_lin = get_weights_aligned(w, DEMO_DEC_PAD * net->hidden_size);
    net->dec_q   = get_weights_aligned(w, DEMO_QDIM * net->hidden_size);
    net->dec_k   = get_weights_aligned(w, DEMO_INV_HID * DEMO_INV_HID);
    net->dec_tau = get_weights_aligned(w, DEMO_PTR_HEADS);
    net->mingru  = make_mingru(w, 1, net->hidden_size, net->num_layers);
    static int logit_sizes[DEMO_NUM_HEADS] = {
        NETHACK_NUM_ACTIONS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_NUM_DIRS};
    net->md = make_multidiscrete(1, logit_sizes, DEMO_NUM_HEADS);
    assert(w->idx == w->size - 7);
    // materialize the residual-factorized embedding once (host, load time)
    net->e_eff = (float*)malloc((size_t)DEMO_VOCAB * DEMO_EMBED * sizeof(float));
    for (int g = 0; g < DEMO_VOCAB; g++)
        for (int d = 0; d < DEMO_EMBED; d++)
            net->e_eff[g * DEMO_EMBED + d] = net->embed[g * DEMO_EMBED + d]
                + net->ekind_w[nh_glyph_kind[g] * DEMO_EMBED + d]
                + net->esub_w[nh_glyph_sub[g] * DEMO_EMBED + d];
    return net;
}

static inline int demo_msg_lc(int c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;   // lowercase; keep spaces/punct
}
static inline int demo_msg_hash(int c0, int c1, int c2) {
    unsigned key = ((unsigned)c0 << 16) | ((unsigned)c1 << 8) | (unsigned)c2;
    return (int)((key * 2654435761u) >> (32 - DEMO_MSG_LOG2V));
}
// normalized-sum trigram bag over the null-terminated topline; scaled by
// 1/sqrt(count+1), no relu (raw signed summary)
static void demo_msg_pool(NethackNet* net, const unsigned char* obs, float* out) {
    const unsigned char* m = obs + NETHACK_OFF_MSG;
    for (int d = 0; d < DEMO_MSG_HID; d++) out[d] = 0.0f;
    int count = 0;
    for (int t = 0; t <= DEMO_MSG_LEN - 3; t++) {
        int c0 = m[t], c1 = m[t + 1], c2 = m[t + 2];
        if (c0 == 0 || c1 == 0 || c2 == 0) break;
        int id = demo_msg_hash(demo_msg_lc(c0), demo_msg_lc(c1), demo_msg_lc(c2));
        count++;
        for (int d = 0; d < DEMO_MSG_HID; d++)
            out[d] += net->msg_w[(size_t)id * DEMO_MSG_HID + d];
    }
    float scale = 1.0f / sqrtf((float)count + 1.0f);
    for (int d = 0; d < DEMO_MSG_HID; d++) out[d] *= scale;
}

static int demo_glyph_at(const int16_t* glyphs, int r, int c) {
    if (r < 0 || r >= NH_ROWS || c < 0 || c >= NH_COLS) return NETHACK_PAD_GLYPH;
    int g = glyphs[r * NH_COLS + c];
    if (g < 0) g = 0;
    if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
    return g;
}

static int nethack_net_forward(NethackNet* net, const unsigned char* obs) {   // fills decoder->output
    const int16_t* glyphs = (const int16_t*)(obs + NETHACK_OFF_GLYPHS);
    const int32_t* bl = (const int32_t*)(obs + NETHACK_OFF_BLSTATS);

    // local view: per-cell embeds of the egocentric crop, flattened
    int hx = bl[0], hy = bl[1];
    int half = NETHACK_CROP / 2;
    for (int p = 0; p < NETHACK_CROP_GRID; p++) {
        int g = demo_glyph_at(glyphs, hy - half + p / NETHACK_CROP,
                              hx - half + p % NETHACK_CROP);
        memcpy(net->x + p * DEMO_EMBED, net->e_eff + g * DEMO_EMBED,
               DEMO_EMBED * sizeof(float));
    }
    _linear(net->x, net->loc_w, net->loc_b, net->concat, 1, DEMO_LOC_IN, DEMO_LOC_HID);
    _relu(net->concat, net->concat, DEMO_LOC_HID);

    // global view: per patch embed+flatten + normalized hero (dx,dy) -> 16 ->
    // 128, elementwise max over the 80 tokens (off-map cells of ragged edge
    // patches read the pad glyph)
    float* glb = net->concat + DEMO_LOC_HID;
    for (int o = 0; o < DEMO_GLB_HID; o++) glb[o] = -1e30f;
    for (int tk = 0; tk < DEMO_TOK; tk++) {
        int r0 = (tk / DEMO_PX) * DEMO_PH, c0 = (tk % DEMO_PX) * DEMO_PW;
        for (int pos = 0; pos < DEMO_PCELLS; pos++) {
            int g = demo_glyph_at(glyphs, r0 + pos / DEMO_PW, c0 + pos % DEMO_PW);
            memcpy(net->px + pos * DEMO_EMBED, net->e_eff + g * DEMO_EMBED,
                   DEMO_EMBED * sizeof(float));
        }
        float dx = (c0 + 0.5f * (DEMO_PW - 1) - hx) / (float)NH_COLS;
        float dy = (r0 + 0.5f * (DEMO_PH - 1) - hy) / (float)NH_ROWS;
        _linear(net->px, net->g1_w, net->g1_b, net->t16, 1, DEMO_GLB_IN, DEMO_P1);
        for (int k = 0; k < DEMO_P1; k++) {
            net->t16[k] += dx * net->g1_xy[k * 2] + dy * net->g1_xy[k * 2 + 1];
            if (net->t16[k] < 0.f) net->t16[k] = 0.f;
        }
        _linear(net->t16, net->g2_w, net->g2_b, net->t128, 1, DEMO_P1, DEMO_GLB_HID);
        for (int o = 0; o < DEMO_GLB_HID; o++)
            if (net->t128[o] > glb[o]) glb[o] = net->t128[o];
    }
    _relu(glb, glb, DEMO_GLB_HID);

    // inventory entities: per-slot embed -> shared 32->32 linear+relu (kept
    // as the pointer decoder's keys), then 32 -> 128 with max over slots for
    // the trunk (matches the CUDA fused pool)
    const int16_t* inv = (const int16_t*)(obs + NETHACK_OFF_INV);
    const signed char* invst = (const signed char*)(obs + NETHACK_OFF_INVST);
    for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
        int g = inv[slot];
        if (g < 0) g = 0;
        if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
        const signed char* st = invst + slot * NLE_INV_STATE_FIELDS;
        float sf[DEMO_SFEAT];
        for (int c = 0; c < 4; c++) sf[c] = st[0] == c ? 1.0f : 0.0f;
        int spe_known = st[1] != -128;
        sf[4] = (float)spe_known;
        sf[5] = spe_known ? (float)st[1] * 0.1f : 0.0f;
        sf[6] = log1pf(fmaxf((float)st[2], 0.0f)) * 0.5f;
        sf[7] = (float)st[3] * (1.0f / 3.0f);
        sf[8] = (float)st[4] * (1.0f / 3.0f);
        for (int c = 0; c < 7; c++) sf[9 + c] = (float)((st[5] >> c) & 1);
        sf[16] = (float)st[6];
        float* h32 = net->slots + slot * DEMO_INV_HID;
        _linear(net->e_eff + g * DEMO_EMBED, net->inv1_w, net->inv1_b,
                h32, 1, DEMO_EMBED, DEMO_INV_HID);
        for (int k = 0; k < DEMO_INV_HID; k++)
            for (int j = 0; j < DEMO_SFEAT; j++)
                h32[k] += net->inv1s_w[k * DEMO_SFEAT + j] * sf[j];
        _relu(h32, h32, DEMO_INV_HID);
    }
    float* invp = net->concat + DEMO_LOC_HID + DEMO_GLB_HID;
    for (int o = 0; o < DEMO_INV_POOL; o++) {
        float best = -1e30f;
        for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
            float v = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                v += net->inv2_w[o * DEMO_INV_HID + k] * net->slots[slot * DEMO_INV_HID + k];
            if (v > best) best = v;
        }
        invp[o] = fmaxf(best + net->inv2_b[o], 0.0f);
    }

    // blstats+extra features (25 scalars, hunger 7, cond bits 13, cooldown,
    // prev verb one-hot, inv class counts)
    float* f = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL + 64;
    int j = 0;
    for (int i = 0; i < 27; i++) {
        if (i == 21 || i == 25) continue;   // hunger, condition: expanded below
        float v = (float)bl[i];
        f[j++] = DEMO_BL_ISLOG[i] ? log1pf(fmaxf(v, 0.f)) * DEMO_BL_SCALE[i]
                                  : v * DEMO_BL_SCALE[i];
    }
    int hunger = bl[21] < 0 ? 0 : (bl[21] > 6 ? 6 : bl[21]);
    for (int h = 0; h < 7; h++) f[j++] = (h == hunger) ? 1.f : 0.f;
    for (int k = 0; k < 13; k++) f[j++] = (float)(((uint32_t)bl[25] >> k) & 1u);
    const int32_t* ex = (const int32_t*)(obs + NETHACK_OFF_EXTRA);
    f[j++] = log1pf(fmaxf((float)ex[0], 0.f)) * 0.1f;   // prayer cooldown
    for (int h = 0; h < NETHACK_NUM_ACTIONS; h++) f[j++] = (h == ex[1]) ? 1.f : 0.f;
    for (int k = 0; k < NETHACK_NUM_OCLASSES; k++) f[j++] = (float)ex[2 + k] * 0.125f;

    float* blout = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL;
    _linear(f, net->bl_w, net->bl_b, blout, 1, DEMO_BL_FEAT, 64);
    _relu(blout, blout, 64);

    demo_msg_pool(net, obs, net->concat + DEMO_MSG_CONCAT_OFF);

    _linear(net->concat, net->proj_w, net->proj_b, net->hidden, 1, DEMO_CONCAT, net->hidden_size);
    _relu(net->hidden, net->hidden, net->hidden_size);

    mingru(net->mingru, net->hidden);

    // pointer decoder: [14 verb | 5x55 slots | 8 dir | value]. verb/dir/value
    // from one bias-free linear; slot logit i = tau_h * cos(q_h, k_i) with
    // keys k_i projected from the per-slot features above.
    float* hs = net->mingru->output;
    int H = net->hidden_size;
    float tmp[DEMO_DEC_LIN];
    for (int r = 0; r < DEMO_DEC_LIN; r++) {
        float acc = 0.0f;
        for (int k = 0; k < H; k++) acc += net->dec_lin[r * H + k] * hs[k];
        tmp[r] = acc;
    }
    float q[DEMO_QDIM];
    for (int r = 0; r < DEMO_QDIM; r++) {
        float acc = 0.0f;
        for (int k = 0; k < H; k++) acc += net->dec_q[r * H + k] * hs[k];
        q[r] = acc;
    }
    float kmat[DEMO_INV_FLAT], kn[NETHACK_INV_SLOTS];
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        float nk = 0.0f;
        for (int r = 0; r < DEMO_INV_HID; r++) {
            float acc = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                acc += net->dec_k[r * DEMO_INV_HID + k] * net->slots[i * DEMO_INV_HID + k];
            kmat[i * DEMO_INV_HID + r] = acc;
            nk += acc * acc;
        }
        kn[i] = sqrtf(nk) + 1e-6f;
    }
    for (int a = 0; a < NETHACK_NUM_ACTIONS; a++) net->logits[a] = tmp[a];
    for (int h = 0; h < DEMO_PTR_HEADS; h++) {
        const float* qh = q + h * DEMO_INV_HID;
        float nq = 0.0f;
        for (int k = 0; k < DEMO_INV_HID; k++) nq += qh[k] * qh[k];
        nq = sqrtf(nq) + 1e-6f;
        for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
            float dot = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                dot += qh[k] * kmat[i * DEMO_INV_HID + k];
            net->logits[NETHACK_NUM_ACTIONS + h * NETHACK_INV_SLOTS + i] =
                expf(net->dec_tau[h]) * dot / (nq * kn[i]);
        }
    }
    for (int d = 0; d <= NETHACK_NUM_DIRS; d++)   // 8 dirs + value
        net->logits[NETHACK_NUM_ACTIONS + DEMO_PTR_HEADS * NETHACK_INV_SLOTS + d] =
            tmp[NETHACK_NUM_ACTIONS + d];
    return 0;
}

// masked multi-head sampling: illegal entries (mask 0) can't be drawn
static void demo_sample(NethackNet* net, const unsigned char* mask, float* out_acts) {
    if (mask != NULL)
        for (int i = 0; i < DEMO_OD; i++)
            if (!mask[i]) net->logits[i] = -1e9f;
    softmax_multidiscrete(net->md, net->logits, out_acts);
}

static void run_demo(long max_steps, int frame_ms) {
    const char* wpath = getenv("NH_WEIGHTS");
    if (!wpath) wpath = "resources/nethack/nethack_weights.bin";
    Weights* w = load_weights((char*)wpath);
    if (!w) {
        fprintf(stderr, "nethack demo: %s missing.\n"
                "Train first, then copy a checkpoint:\n"
                "  cp checkpoints/nethack/<run>/<final>.bin resources/nethack/nethack_weights.bin\n",
                wpath);
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env;
    env_open(&env);
    const char* seed_env = getenv("NH_SEED");   // fixed seed pairs A/B sampling
    srand(seed_env ? (unsigned)strtoul(seed_env, NULL, 10) : (unsigned)time(NULL));

    long max_dlvl = 1;
    long acts[NETHACK_NUM_ACTIONS] = {0};
    float ep_score = 0.0f, ep_len = 0.0f;   // log totals at last episode end
    float p_comb=0, p_starv=0, p_smite=0, p_trunc=0, p_mdep=0, p_xp=0,
          p_klvl=0, p_mhp=0, p_adj=0, p_burst=0;   // per-episode death deltas
    float acts_f[DEMO_NUM_HEADS];
    const int keep_state = getenv("NH_KEEP_STATE") != NULL;
    const char* max_ep_env = getenv("NH_MAX_EPISODES");
    const long max_episodes = max_ep_env ? strtol(max_ep_env, NULL, 10) : 0;   // 0 = unbounded
    long nep = 0;
    for (long t = 0; t < max_steps; t++) {
        nethack_net_forward(net, env.observations);
        demo_sample(net, env.action_mask, acts_f);
        int a = (int)acts_f[0];
        acts[a]++;
        for (int h = 0; h < DEMO_NUM_HEADS; h++) env.actions[h] = acts_f[h];
        c_step(&env);
        if (env.blstats[NLE_BL_DEPTH] > max_dlvl) max_dlvl = env.blstats[NLE_BL_DEPTH];
        if (frame_ms > 0) {
            c_render(&env);
            usleep(frame_ms * 1000);
        } else if (t % 2000 == 1999) {
            fprintf(stderr, "step %ld: score=%ld dlvl=%ld T=%ld len=%d\n", t + 1,
                    env.blstats[NLE_BL_SCORE], env.blstats[NLE_BL_DEPTH],
                    env.blstats[NLE_BL_TIME], env.stats.length);
        }
        if (env.terminals[0] > 0.5f) {
            // c_step already reset the env; per-episode values via log deltas
            float dc=env.log.death_combat-p_comb, dst=env.log.death_starved-p_starv,
                  dsm=env.log.death_smited-p_smite, dtr=env.log.truncated-p_trunc;
            const char* how = dc>0.5f?"combat": dst>0.5f?"starved":
                              dsm>0.5f?"smited": dtr>0.5f?"trunc":"other";
            fprintf(stderr, "episode end: score=%.0f len=%.0f how=%s mdepth=%.0f xp=%.0f "
                    "klvl=%.0f maxhp=%.0f adj=%.0f burst=%.0f\n",
                    env.log.score - ep_score, env.log.episode_length - ep_len, how,
                    env.log.max_depth - p_mdep, env.log.max_xp_level - p_xp,
                    env.log.death_mon_level - p_klvl, env.log.death_maxhp - p_mhp,
                    env.log.death_adj_monsters - p_adj, env.log.death_burst_turns - p_burst);
            ep_score = env.log.score;
            ep_len = env.log.episode_length;
            p_comb=env.log.death_combat; p_starv=env.log.death_starved;
            p_smite=env.log.death_smited; p_trunc=env.log.truncated;
            p_mdep=env.log.max_depth; p_xp=env.log.max_xp_level;
            p_klvl=env.log.death_mon_level; p_mhp=env.log.death_maxhp;
            p_adj=env.log.death_adj_monsters; p_burst=env.log.death_burst_turns;
            if (!keep_state)
                memset(net->mingru->state, 0,
                       (size_t)net->num_layers * net->hidden_size * sizeof(float));
            if (max_episodes && ++nep >= max_episodes) break;
        }
    }
    fprintf(stderr, "actions:");
    for (int i = 0; i < NETHACK_NUM_ACTIONS; i++)
        fprintf(stderr, " %s=%ld", NETHACK_ACTION_NAMES[i], acts[i]);
    fprintf(stderr, "\n");
    if (env.log.n > 0) {
        printf("demo: episodes=%.0f  avg_score=%.1f  avg_len=%.0f  avg_depth=%.2f  max_dlvl=%ld\n",
               env.log.n, env.log.score / env.log.n, env.log.episode_length / env.log.n,
               env.log.depth / env.log.n, max_dlvl);
        printf("areas: mines=%.2f  minetown=%.2f  deep_mines=%.2f  main_d5=%.2f  sokoban=%.2f\n",
               env.log.reach_mines / env.log.n, env.log.reach_minetown / env.log.n,
               env.log.reach_deep_mines / env.log.n, env.log.reach_main_d5 / env.log.n,
               env.log.reach_sokoban / env.log.n);
        printf("rest/burden: regen_ticks=%.1f ups_hurt=%.2f burdened=%.1f%% pack_full=%.1f%% searches=%.1f (per ep)\n",
               env.log.regen_ticks / env.log.n, env.log.ups_hurt / env.log.n,
               100.f * env.log.burdened_frac / env.log.n, 100.f * env.log.pack_full_frac / env.log.n,
               env.log.searches / env.log.n);
        printf("deaths: combat=%.2f starved=%.2f killer_lvl=%.1f burst_turns=%.1f adj=%.1f maxhp=%.1f\n",
               env.log.death_combat / env.log.n, env.log.death_starved / env.log.n,
               env.log.death_mon_level / env.log.n, env.log.death_burst_turns / env.log.n,
               env.log.death_adj_monsters / env.log.n, env.log.death_maxhp / env.log.n);
        printf("items: wears=%.2f eats=%.2f (floor %.2f) quaffs=%.2f throws=%.2f zaps=%.2f prayers=%.2f engraves=%.2f (per ep)\n",
               env.log.wears / env.log.n, env.log.eats / env.log.n,
               env.log.floor_eats / env.log.n, env.log.quaffs / env.log.n,
               env.log.throws / env.log.n, env.log.zaps / env.log.n,
               env.log.prayers / env.log.n, env.log.engraves / env.log.n);
    }
    env_close(&env);
    free_mingru(net->mingru);
    free(net->md); free(net->hidden); free(net->e_eff); free(net);
    free(w);
}

// ./nethack [N_STEPS] [MS_PER_FRAME (0 = headless)]
int main(int argc, char** argv) {
    run_demo((argc >= 2) ? atol(argv[1]) : 1000000,
             (argc >= 3) ? atoi(argv[2]) : 50);
    return 0;
}
