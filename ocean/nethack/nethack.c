#include <time.h>
#include <unistd.h>
#include <string.h>
#include "nethack.h"
#include "../../src/puffernet.h"

// labels for the end-of-run histogram, in NETHACK_ACTION_TABLE order
static const char* NETHACK_ACTION_NAMES[NETHACK_NUM_ACTIONS] = {
    "N","S","W","E","NW","NE","SW","SE",
    "N_RUN","S_RUN","W_RUN","E_RUN","NW_RUN","NE_RUN","SW_RUN","SE_RUN",
    "DOWN_STAIRS","UP_STAIRS","KICK","SEARCH","PRAY","ELBERETH","WEAR","EAT",
};

// single-agent env, reset immediately (training's c_reset is lazy)
static void env_open(Nethack* env) {
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env->actions      = (float*)calloc(1, sizeof(float));
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (float*)calloc(1, sizeof(float));
    init(env);
    nethack_do_reset(env);
}

static void env_close(Nethack* env) {
    c_close(env);
    free(env->observations); free(env->actions); free(env->rewards); free(env->terminals);
}

// CPU port of the CUDA encoder (src/nethack.cu) + puffernet MinGRU/decoder;
// weight order matches param registration: encoder, decoder, mingru
#define DEMO_VOCAB   5977
#define DEMO_EMBED   32
#define DEMO_BL_FEAT 88   // 25 scalars + hunger 7 + cond 13 + cooldown + prevact 24 + counts 18
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
#define DEMO_CONCAT  (DEMO_LOC_HID + DEMO_GLB_HID + 64 + DEMO_BL_FEAT)

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
    float *embed;               // (5977, 32)
    float *loc_w, *loc_b;       // (256, 2592), (256)
    float *g1_w, *g1_xy, *g1_b; // (16, 800), (16, 2), (16): per-patch embed+flatten + hero dx,dy -> 16
    float *g2_w, *g2_b;         // (128, 16), (128): 16 -> 128, maxed over tokens
    float *bl_w, *bl_b;         // (64, 88), (64)
    float *proj_w, *proj_b;     // (H, 536), (H)
    Linear* decoder;            // (24+1, H), bias-free; last output is value
    MinGRU* mingru;
    Multidiscrete* md;
    int hidden_size, num_layers, num_actions;
    float x[DEMO_LOC_IN];       // crop cell embeds, flattened
    float px[DEMO_GLB_IN];      // one patch's cell embeds, flattened
    float t16[DEMO_P1];
    float t128[DEMO_GLB_HID];
    float concat[DEMO_CONCAT];  // [local hid | global hid | bl hidden | bl feats]
    float* hidden;              // (hidden_size)
} NethackNet;

// (hidden, layers) from the checkpoint float count:
//   total = ENC_FIXED + H*(DEMO_CONCAT + 1 + A + 1) + L * 3*H*H
#define DEMO_ENC_FIXED (DEMO_VOCAB*DEMO_EMBED + DEMO_LOC_HID*DEMO_LOC_IN + DEMO_LOC_HID \
                        + DEMO_P1*DEMO_GLB_IN + DEMO_P1*2 + DEMO_P1 \
                        + DEMO_GLB_HID*DEMO_P1 + DEMO_GLB_HID + 64*DEMO_BL_FEAT + 64)
// ambiguities are possible; prefer the fewest layers (real configs have <= 8)
static int demo_infer_arch(int total, int* hidden, int* layers, int* actions) {
    int best_l = 1 << 30;
    for (int H = 8; H <= 4096; H += 8) {
        long rem = (long)total - DEMO_ENC_FIXED - (long)H * (DEMO_CONCAT + 1 + NETHACK_NUM_ACTIONS + 1);
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
    net->loc_w   = get_weights_aligned(w, DEMO_LOC_HID * DEMO_LOC_IN);
    net->loc_b   = get_weights_aligned(w, DEMO_LOC_HID);
    net->g1_w    = get_weights_aligned(w, DEMO_P1 * DEMO_GLB_IN);
    net->g1_xy   = get_weights_aligned(w, DEMO_P1 * 2);
    net->g1_b    = get_weights_aligned(w, DEMO_P1);
    net->g2_w    = get_weights_aligned(w, DEMO_GLB_HID * DEMO_P1);
    net->g2_b    = get_weights_aligned(w, DEMO_GLB_HID);
    net->bl_w    = get_weights_aligned(w, 64 * DEMO_BL_FEAT);
    net->bl_b    = get_weights_aligned(w, 64);
    net->proj_w  = get_weights_aligned(w, net->hidden_size * DEMO_CONCAT);
    net->proj_b  = get_weights_aligned(w, net->hidden_size);
    net->decoder = make_linear(w, 1, net->hidden_size, net->num_actions + 1);
    net->mingru  = make_mingru(w, 1, net->hidden_size, net->num_layers);
    int logit_sizes[1] = {net->num_actions};
    net->md = make_multidiscrete(1, logit_sizes, 1);
    assert(w->idx == w->size - 7);
    return net;
}

static int demo_glyph_at(const int16_t* glyphs, int r, int c) {
    if (r < 0 || r >= NH_ROWS || c < 0 || c >= NH_COLS) return NETHACK_PAD_GLYPH;
    int g = glyphs[r * NH_COLS + c];
    if (g < 0) g = 0;
    if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
    return g;
}

static int nethack_net_forward(NethackNet* net, const unsigned char* obs) {
    const int16_t* glyphs = (const int16_t*)(obs + NETHACK_OFF_GLYPHS);
    const int32_t* bl = (const int32_t*)(obs + NETHACK_OFF_BLSTATS);

    // local view: per-cell embeds of the egocentric crop, flattened
    int hx = bl[0], hy = bl[1];
    int half = NETHACK_CROP / 2;
    for (int p = 0; p < NETHACK_CROP_GRID; p++) {
        int g = demo_glyph_at(glyphs, hy - half + p / NETHACK_CROP,
                              hx - half + p % NETHACK_CROP);
        memcpy(net->x + p * DEMO_EMBED, net->embed + g * DEMO_EMBED,
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
            memcpy(net->px + pos * DEMO_EMBED, net->embed + g * DEMO_EMBED,
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

    // blstats+extra -> 88 features (25 scalars, hunger 7, cond bits 13,
    // cooldown, prev action 24, inv counts 18)
    float* f = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + 64;
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

    float* blout = net->concat + DEMO_LOC_HID + DEMO_GLB_HID;
    _linear(f, net->bl_w, net->bl_b, blout, 1, DEMO_BL_FEAT, 64);
    _relu(blout, blout, 64);

    _linear(net->concat, net->proj_w, net->proj_b, net->hidden, 1, DEMO_CONCAT, net->hidden_size);
    _relu(net->hidden, net->hidden, net->hidden_size);

    mingru(net->mingru, net->hidden);
    linear(net->decoder, net->mingru->output);
    float action = 0.f;
    softmax_multidiscrete(net->md, net->decoder->output, &action);
    return (int)action;
}

static void run_demo(long max_steps, int frame_ms) {
    Weights* w = load_weights("resources/nethack/nethack_weights.bin");
    if (!w) {
        fprintf(stderr, "nethack demo: resources/nethack/nethack_weights.bin missing.\n"
                "Train first, then copy a checkpoint:\n"
                "  cp checkpoints/nethack/<run>/<final>.bin resources/nethack/nethack_weights.bin\n");
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env;
    env_open(&env);
    srand((unsigned)time(NULL));

    long max_dlvl = 1;
    long acts[NETHACK_NUM_ACTIONS] = {0};
    float ep_score = 0.0f, ep_len = 0.0f;   // log totals at last episode end
    for (long t = 0; t < max_steps; t++) {
        int a = nethack_net_forward(net, env.observations);
        acts[a]++;
        env.actions[0] = (float)a;
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
            fprintf(stderr, "episode end: score=%.0f len=%.0f\n",
                    env.log.score - ep_score, env.log.episode_length - ep_len);
            ep_score = env.log.score;
            ep_len = env.log.episode_length;
            memset(net->mingru->state, 0,
                   (size_t)net->num_layers * net->hidden_size * sizeof(float));
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
        printf("items: wears=%.2f  eats=%.2f  engraves=%.2f (per episode)\n",
               env.log.wears / env.log.n, env.log.eats / env.log.n,
               env.log.engraves / env.log.n);
    }
    env_close(&env);
    free_mingru(net->mingru);
    free(net->decoder); free(net->md); free(net->hidden); free(net);
    free(w);
}

// ./nethack [N_STEPS] [MS_PER_FRAME (0 = headless)]
int main(int argc, char** argv) {
    run_demo((argc >= 2) ? atol(argv[1]) : 1000000,
             (argc >= 3) ? atoi(argv[2]) : 50);
    return 0;
}
