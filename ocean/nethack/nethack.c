#include <time.h>
#include <unistd.h>
#include <string.h>
#include "nethack.h"
#include "../../src/puffernet.h"

// Action index -> printable label for the trajectory log.
// Matches NETHACK_ACTION_TABLE: k j h l y u b n  K J H L Y U B N  > < ^D s M-p
static const char* NETHACK_ACTION_NAMES[NETHACK_NUM_ACTIONS] = {
    "N","S","W","E","NW","NE","SW","SE",
    "N_RUN","S_RUN","W_RUN","E_RUN","NW_RUN","NE_RUN","SW_RUN","SE_RUN",
    "DOWN_STAIRS","UP_STAIRS","KICK","SEARCH","PRAY",
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------
// Trained-policy demo: CPU port of the custom CUDA encoder (src/ocean.cu
// NethackEncoder) feeding puffernet's MinGRU + decoder. Weights =
// resources/nethack/nethack_weights.bin, the fp32 dump training saves.
// Weight order matches param registration: encoder tensors, decoder, mingru.
// ---------------------------------------------------------------------------
#define DEMO_VOCAB   5977
#define DEMO_EMBED   32
#define DEMO_BL_FEAT 45
#define DEMO_CONV    1024                       // conv2 out, 64ch x 4x4, NCHW flat
#define DEMO_CONCAT  (DEMO_CONV + 64 + DEMO_BL_FEAT)

// Per-blstat normalization, mirroring NH_BL_SCALE / NH_BL_ISLOG in src/ocean.cu.
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
    float *conv1_w, *conv1_b;   // (64, 32*5*5), (64)
    float *conv2_w, *conv2_b;   // (64, 64*3*3), (64)
    float *embed;               // (5977, 32)
    float *bl_w, *bl_b;         // (64, 45), (64)
    float *proj_w, *proj_b;     // (H, 1133), (H)
    Linear* decoder;            // (18+1, H), bias-free; last output is value
    MinGRU* mingru;
    Multidiscrete* md;
    int hidden_size, num_layers, num_actions;
    float x[DEMO_EMBED * NETHACK_CROP_GRID];   // embedded crop, NCHW (32,21,21)
    float c1[64 * 6 * 6];
    float concat[DEMO_CONCAT];                 // [conv2 flat | bl hidden | bl feats]
    float* hidden;              // (hidden_size)
} NethackNet;

// The checkpoint's float count uniquely determines (hidden, layers): all
// encoder tensors are fixed except proj, and every tensor is a multiple of
// 8 floats (the dump's 16-byte alignment), so
//   total = ENC_FIXED + H*(1133 + 1 + A + 1) + L * 3*H*H.
// A (action count) is also inferred: 21 = current, 20 = pre-pray checkpoints
// (actions are appended, so old action indices are unchanged).
#define DEMO_ENC_FIXED (51200 + 64 + 36864 + 64 + DEMO_VOCAB*DEMO_EMBED + 64*DEMO_BL_FEAT + 64)
// Ambiguities are possible (e.g. 380768 floats = 64/2/18 = 32/20/19);
// prefer the fewest layers — real configs have <= 8.
static int demo_infer_arch(int total, int* hidden, int* layers, int* actions) {
    static const int CAND_A[2] = {NETHACK_NUM_ACTIONS, NETHACK_NUM_ACTIONS - 1};
    int best_l = 1 << 30;
    for (int a = 0; a < 2; a++) {
        for (int H = 8; H <= 4096; H += 8) {
            long rem = (long)total - DEMO_ENC_FIXED - (long)H * (DEMO_CONCAT + 1 + CAND_A[a] + 1);
            long per_layer = 3L * H * H;
            if (rem <= 0) break;
            if (rem % per_layer) continue;
            long L = rem / per_layer;
            if (L >= 1 && L < best_l) { best_l = (int)L; *hidden = H; *layers = (int)L; *actions = CAND_A[a]; }
        }
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
    net->conv1_w = get_weights_aligned(w, 64 * DEMO_EMBED * 25);
    net->conv1_b = get_weights_aligned(w, 64);
    net->conv2_w = get_weights_aligned(w, 64 * 64 * 9);
    net->conv2_b = get_weights_aligned(w, 64);
    net->embed   = get_weights_aligned(w, DEMO_VOCAB * DEMO_EMBED);
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

static int nethack_net_forward(NethackNet* net, const unsigned char* obs) {
    const int16_t* glyphs = (const int16_t*)(obs + NETHACK_OFF_GLYPHS);
    const int32_t* bl = (const int32_t*)(obs + NETHACK_OFF_BLSTATS);

    for (int p = 0; p < NETHACK_CROP_GRID; p++) {
        int g = glyphs[p];
        if (g < 0) g = 0;
        if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
        const float* e = net->embed + g * DEMO_EMBED;
        for (int d = 0; d < DEMO_EMBED; d++)
            net->x[d * NETHACK_CROP_GRID + p] = e[d];
    }
    _conv2d(net->x, net->conv1_w, net->conv1_b, net->c1,
            1, NETHACK_CROP, NETHACK_CROP, DEMO_EMBED, 64, 5, 3);
    _relu(net->c1, net->c1, 64 * 36);
    _conv2d(net->c1, net->conv2_w, net->conv2_b, net->concat, 1, 6, 6, 64, 64, 3, 1);

    // blstats -> 45 features: 25 scaled scalars, hunger one-hot 7, condition bits 13
    float* f = net->concat + DEMO_CONV + 64;
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

    float* blout = net->concat + DEMO_CONV;
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

static void standalone_reset(Nethack* env);

static void run_demo(long max_steps, int frame_ms) {
    Weights* w = load_weights("resources/nethack/nethack_weights.bin");
    if (!w) {
        fprintf(stderr, "nethack demo: resources/nethack/nethack_weights.bin missing.\n"
                "Train first, then copy a checkpoint:\n"
                "  cp checkpoints/nethack/<run>/<final>.bin resources/nethack/nethack_weights.bin\n");
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env; memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));
    init(&env);
    standalone_reset(&env);
    srand((unsigned)time(NULL));

    long max_dlvl = 1;
    long acts[NETHACK_NUM_ACTIONS] = {0};
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
                    env.blstats[NLE_BL_TIME], env.episode_length);
        }
        if (env.terminals[0] > 0.5f) {
            fprintf(stderr, "episode end: score=%ld len=%d\n", env.prev_score, env.episode_length);
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
    }
    c_close(&env);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    free_mingru(net->mingru);
    free(net->decoder); free(net->md); free(net->hidden); free(net);
    free(w);
}

static void standalone_reset(Nethack* env) {
    env->pending_reset = 0;
    nethack_do_reset(env);
}

static void run_interactive(int max_steps) {
    Nethack env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    init(&env);
    standalone_reset(&env);
    c_render(&env);

    srand((unsigned)time(NULL));
    for (int t = 0; t < max_steps; t++) {
        env.actions[0] = (float)(rand() % NETHACK_NUM_ACTIONS);
        c_step(&env);
        c_render(&env);
        printf("step=%d action=%d reward=%.2f done=%.0f\n",
               t, (int)env.actions[0], env.rewards[0], env.terminals[0]);
        usleep(50 * 1000);
    }
    c_close(&env);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

static void run_benchmark(long steps, int policy) {
    // policy: 0 = random, 1 = hold position (move N into a wall is the
    // closest thing to waiting in the 18-action set)
    Nethack env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    init(&env);
    standalone_reset(&env);
    srand(0xC0FFEE);
    long episodes = 1;
    double sum_reward = 0.0;

    double t0 = now_sec();
    for (long t = 0; t < steps; t++) {
        env.actions[0] = (policy == 1) ? 0.0f : (float)(rand() % NETHACK_NUM_ACTIONS);
        c_step(&env);
        sum_reward += env.rewards[0];
        if (env.terminals[0] > 0.5f) episodes++;
    }
    double dt = now_sec() - t0;

    printf("steps=%ld  episodes=%ld  time=%.3fs  steps/sec=%.0f  sum_reward=%.1f\n",
           steps, episodes, dt, steps / dt, sum_reward);
    printf("  valid_moves=%ld (%.1f%%)  illegal_actions=%ld (%.1f%%)\n",
           env.episode_valid_moves,
           100.0 * env.episode_valid_moves / (double)steps,
           env.episode_illegal_actions,
           100.0 * env.episode_illegal_actions / (double)steps);
    printf("  final HP=%ld/%ld  AC=%ld  Dlvl=%ld  Score=%ld  GameTime=%ld\n",
           env.blstats[NLE_BL_HP], env.blstats[NLE_BL_HPMAX],
           env.blstats[NLE_BL_AC],
           env.blstats[NLE_BL_DEPTH], env.blstats[NLE_BL_SCORE],
           env.blstats[NLE_BL_TIME]);

    c_close(&env);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

// Dump one frame as plain text (chars grid + status + message + action label).
static void dump_frame(FILE* f, Nethack* env, long step, int action_idx, float reward) {
    fprintf(f, "=== step %ld | action=%s | reward=%+.2f | done=%.0f | in_normal_game=%d ===\n",
            step, NETHACK_ACTION_NAMES[action_idx], reward,
            env->terminals[0], env->obs.in_normal_game);
    fprintf(f, "HP %ld/%ld  AC %ld  Dlvl %ld  Score %ld  GameTime %ld  Hunger %ld\n",
            env->blstats[NLE_BL_HP], env->blstats[NLE_BL_HPMAX],
            env->blstats[NLE_BL_AC], env->blstats[NLE_BL_DEPTH],
            env->blstats[NLE_BL_SCORE], env->blstats[NLE_BL_TIME],
            env->blstats[NLE_BL_HUNGER]);
    fprintf(f, "Msg: %.256s\n", env->message);
    for (int r = 0; r < NH_ROWS; r++) {
        fputs("  |", f);
        for (int c = 0; c < NH_COLS; c++) {
            unsigned char ch = env->chars[r * NH_COLS + c];
            fputc(ch >= 32 && ch < 127 ? ch : (ch == 0 ? ' ' : '?'), f);
        }
        fputs("|\n", f);
    }
    fputc('\n', f);
}

static void run_record(const char* path, long steps, int policy) {
    FILE* f = (strcmp(path, "-") == 0) ? stdout : fopen(path, "w");
    if (!f) { perror("fopen"); return; }

    Nethack env; memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));
    init(&env);
    standalone_reset(&env);
    dump_frame(f, &env, -1, 0, 0.0f);   // post-reset frame, "action" col is meaningless

    srand(0xC0FFEE);
    for (long t = 0; t < steps; t++) {
        int action_idx = (policy == 1) ? 0 : (rand() % NETHACK_NUM_ACTIONS);
        env.actions[0] = (float)action_idx;
        c_step(&env);
        dump_frame(f, &env, t, action_idx, env.rewards[0]);
        if (env.terminals[0] > 0.5f) {
            fprintf(f, "##### EPISODE END at step %ld #####\n\n", t);
            break;
        }
    }
    c_close(&env);
    if (f != stdout) fclose(f);
    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "record") == 0) {
        // Usage: ./nethack record OUTFILE [N_STEPS] [random|wait]
        const char* path = (argc >= 3) ? argv[2] : "trajectory.txt";
        long steps      = (argc >= 4) ? atol(argv[3]) : 100;
        int policy      = (argc >= 5 && strcmp(argv[4], "wait") == 0) ? 1 : 0;
        run_record(path, steps, policy);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        long steps = (argc >= 3) ? atol(argv[2]) : 100000;
        int policy = (argc >= 4 && strcmp(argv[3], "wait") == 0) ? 1 : 0;
        printf("policy=%s\n", policy ? "wait" : "random");
        run_benchmark(steps, policy);
    } else if (argc >= 2 && strcmp(argv[1], "resets") == 0) {
        long n = (argc >= 3) ? atol(argv[2]) : 50;
        Nethack env; memset(&env, 0, sizeof(env));
        env.num_agents = 1;
        env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
        env.actions      = (float*)calloc(1, sizeof(float));
        env.rewards      = (float*)calloc(1, sizeof(float));
        env.terminals    = (float*)calloc(1, sizeof(float));
        init(&env);
        standalone_reset(&env); c_step(&env);  // warm-up not counted
        double t0 = now_sec();
        // c_reset is lazy (sets a flag); the step forces the actual reset.
        for (long i = 0; i < n; i++) { c_reset(&env); env.actions[0] = 0.0f; c_step(&env); }
        double dt = now_sec() - t0;
        printf("resets=%ld  time=%.3fs  per_reset=%.2fms  resets/sec=%.0f\n",
               n, dt, 1000.0 * dt / n, n / dt);
        c_close(&env);
        free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    } else if (argc >= 2 && strcmp(argv[1], "stepreset") == 0) {
        // step+reset cycle test. Args: STEPS_PER_EPISODE NUM_RESETS
        long steps_per = (argc >= 3) ? atol(argv[2]) : 100;
        long n_resets  = (argc >= 4) ? atol(argv[3]) : 20;
        Nethack env; memset(&env, 0, sizeof(env));
        env.num_agents = 1;
        env.observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
        env.actions      = (float*)calloc(1, sizeof(float));
        env.rewards      = (float*)calloc(1, sizeof(float));
        env.terminals    = (float*)calloc(1, sizeof(float));
        init(&env);
        standalone_reset(&env);
        srand(0xC0FFEE);
        double t0 = now_sec();
        for (long r = 0; r < n_resets; r++) {
            for (long t = 0; t < steps_per; t++) {
                env.actions[0] = (float)(rand() % NETHACK_NUM_ACTIONS);
                c_step(&env);
            }
            c_reset(&env);
        }
        double dt = now_sec() - t0;
        long total = steps_per * n_resets;
        printf("stepreset: %ld cycles x %ld steps = %ld steps  time=%.3fs  sps=%.0f\n",
               n_resets, steps_per, total, dt, total / dt);
        c_close(&env);
        free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    } else if (argc >= 2 && strcmp(argv[1], "random") == 0) {
        int max_steps = (argc >= 3) ? atoi(argv[2]) : 100000;
        run_interactive(max_steps);
    } else {
        // Default: play the trained policy from resources/nethack.
        // Usage: ./nethack [N_STEPS] [MS_PER_FRAME (0 = headless)]
        long steps   = (argc >= 2) ? atol(argv[1]) : 1000000;
        int frame_ms = (argc >= 3) ? atoi(argv[2]) : 50;
        run_demo(steps, frame_ms);
    }
    return 0;
}
