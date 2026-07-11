// NetHack env: 21x21 egocentric glyph crop + blstats + extra stats obs
// (1070 bytes, matching src/nethack.cu), 24 actions, decomposed-score reward.
// One env per agent, each owning an nle_ctx_t and a private vardir on tmpfs.
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>
#include "nethack_fs.h"

// nletypes.h, not nle.h: nle.h's `settings` macro would rewrite env->settings
#include "nletypes.h"

extern nle_ctx_t* nle_start(nle_obs*, FILE*, nle_settings*);
extern nle_ctx_t* nle_step(nle_ctx_t*, nle_obs*);
extern void       nle_end(nle_ctx_t*);

#define NH_ROWS 21
#define NH_COLS 79
#define NH_GRID (NH_ROWS * NH_COLS)

// encoder views (GPU-side, cut from the full grid): odd-side egocentric crop
// padded with NO_GLYPH (== MAX_GLYPH) off-map, plus 5x5 patches over the map
#define NETHACK_CROP       9
#define NETHACK_CROP_GRID  (NETHACK_CROP * NETHACK_CROP)
#define NETHACK_PAD_GLYPH  5976

// corpse glyph range [GLYPH_BODY_OFF, +NUMMONS), display.h
#define NETHACK_GLYPH_BODY_OFF 1144
#define NETHACK_NUMMONS        381

// obs layout: full glyph grid | blstats | extra (prayer cooldown, prev action,
// inv counts). The grid ships whole (map memory included); the encoder derives
// the egocentric crop from blstats x,y.
#define NETHACK_NUM_OCLASSES 18   // MAXOCLASSES; inv_oclasses pads with 18
#define NETHACK_OFF_GLYPHS  0
#define NETHACK_OFF_BLSTATS (NH_GRID * 2)
#define NETHACK_OFF_EXTRA   (NETHACK_OFF_BLSTATS + NLE_BLSTATS_SIZE * 4)
#define NETHACK_EXTRA_INTS  (2 + NETHACK_NUM_OCLASSES)
// inventory entity block: 55 slot glyphs int16 LE, in inventory order (the
// item-slot action head indexes these positions); empty slots = pad glyph
#define NETHACK_INV_SLOTS   NLE_INVENTORY_SIZE
#define NETHACK_OFF_INV     (NETHACK_OFF_EXTRA + NETHACK_EXTRA_INTS * 4)
#define NETHACK_OBS_SIZE    (NETHACK_OFF_INV + NETHACK_INV_SLOTS * 2)
#define NETHACK_INTERNAL_UBLESSCNT 5   // u.ublesscnt, vendored winrl.cc patch
#define NETHACK_INTERNAL_KILLER_MNUM 9 // killer monster index + 1 (0 = not a monster), death only
#define NETHACK_INTERNAL_KILLER_MLEV 10 // killer monster level, death only

#define NETHACK_MAX_EPISODE_STEPS 10000
#define NETHACK_AUTODISMISS_MAX   64   // cap on prompt-dismiss keystrokes per step
#define NETHACK_MAX_DEPTH         64   // scout bitmaps tracked per episode

// Stats.areas bits; logged as reach_* proportions
#define NETHACK_AREA_MINES      1u   // Gnomish Mines (dnum 2)
#define NETHACK_AREA_MINETOWN   2u   // Mines level 3+ (Minetown band)
#define NETHACK_AREA_DEEP_MINES 4u   // Mines level 5+ (past Minetown)
#define NETHACK_AREA_MAIN_D5    8u   // Dungeons of Doom depth 5+ (Oracle route)
#define NETHACK_AREA_SOKOBAN    16u  // Sokoban (dnum 4)

// nle_obs.misc[] prompt-state flags
enum { NETHACK_MISC_YN = 0, NETHACK_MISC_GETLIN = 1, NETHACK_MISC_XWAIT = 2 };

// Factored action space: verb head (12) + item-slot head (55) + direction
// head (8, vi-key order). MOVE/RUN/KICK/THROW consume the direction; the
// direction head trains dense on every move, so directional verbs inherit it.
#define NETHACK_NUM_ACTIONS 13
#define NETHACK_NUM_DIRS    8
static const int NETHACK_DIR_KEYS[NETHACK_NUM_DIRS] =
    {'k','j','h','l','y','u','b','n'};   // N S W E NW NE SW SE

enum {
    NETHACK_ACT_MOVE     = 0,
    NETHACK_ACT_RUN      = 1,
    NETHACK_ACT_DOWN     = 2,
    NETHACK_ACT_UP       = 3,
    NETHACK_ACT_KICK     = 4,
    NETHACK_ACT_SEARCH   = 5,
    NETHACK_ACT_ELBERETH = 6,
    NETHACK_ACT_WEAR     = 7,
    NETHACK_ACT_EAT      = 8,
    NETHACK_ACT_QUAFF    = 9,
    NETHACK_ACT_PRAY     = 10,
    NETHACK_ACT_THROW    = 11,
    NETHACK_ACT_ZAP      = 12,
};

// !status_updates skips the status renderer + recalc_mapseen (~25% of engine)
#define NETHACK_DEFAULT_OPTIONS \
    "name:Agent-mon-hum-neu-mal," \
    "autopickup,color,disclose:+i +a +v +g +c +o," \
    "mention_walls,nobones,nocmdassist,nolegacy,nosparkle," \
    "pickup_burden:unencumbered,pickup_types:$[%!)/," \
    "runmode:teleport,showexp,showscore,time," \
    "!status_updates"

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float depth;
    float valid_moves;        // steps that advanced NetHack's turn counter
    float illegal_actions;    // steps that hit a sub-prompt we ESC'd
    float new_tiles;
    float max_depth;          // deepest level reached (depth under-reports at death)
    float searches;
    float engraves;
    float wears;              // macro presses that chose an item
    float eats;
    float floor_eats;         // eats that accepted a floor "eat it?" offer
    float quaffs;             // quaff presses that drank a potion
    float prayers;
    float prayers_low_hp;     // prayers at <=25% max HP
    float throws;             // throw presses that launched an item
    float zaps;               // zap presses that fired a wand
    // slot position-dependence diagnostic: if uses sit far below where valid
    // items live, the flat slot head isn't generalizing across positions
    float use_slot_mean;      // mean slot index of successful item uses
    float valid_slot_mean;    // mean slot index of mask-legal slots
    float damage_taken;
    float reward_saturated;   // fraction of steps with |reward| > 1
    float game_time;          // NetHack turns survived
    float max_xp_level;
    // episode end reason, one-hot (game_end_types in hack.h)
    float death_combat;
    float death_starved;
    float death_smited;       // god's wrath (NLE_HOW_WRATH), not a monster kill
    float death_other;
    // combat-death anatomy (0 for non-combat episodes; ~95% are combat)
    float death_mon_level;    // killer's monster level (vs max_xp_level = the mismatch)
    float death_burst_turns;  // turns from last >=75% HP to death (small = ambush/burst)
    float death_adj_monsters; // hostile monsters adjacent on the last obs before death
    float death_maxhp;        // max HP at death (progression measure)
    float truncated;          // hit NETHACK_MAX_EPISODE_STEPS
    // 0/1 per episode; the logged mean is the proportion
    float reach_mines;
    float reach_minetown;
    float reach_deep_mines;
    float reach_main_d5;
    float reach_sokoban;
    float n;
} Log;

// per-episode stats; cleared with one memset per reset
typedef struct Stats {
    long valid_moves;
    long illegal_actions;
    long new_tiles;
    long searches;
    long engraves;
    long wears;
    long eats;
    long quaffs;
    long prayers;
    long throws;
    long zaps;
    long use_slot_sum, use_cnt;       // slot indices of successful item uses
    long valid_slot_sum, valid_cnt;   // slot indices legal in the mask, per step
    long last_ok_turn;   // last turn with HP >= 75% max
    long last_maxhp;
    int  last_adj;       // hostile monsters adjacent, last obs
    long prayers_low_hp;
    long floor_eats;
    long damage;
    long saturated;
    int max_depth;
    int max_xp;
    unsigned areas;         // NETHACK_AREA_* bits
    float ret;
    int length;
    // per-level first-visit bitmaps; branch levels sharing a depth share one
    unsigned char visited[NETHACK_MAX_DEPTH][(NH_GRID + 7) / 8];
} Stats;

typedef struct Nethack {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;   // (12+55+8,) — NULL unless MY_ACTION_MASK wired
    int num_agents;

    // deferred reset for main-thread c_reset (vecenv setup): NLE's coroutine
    // must be reset on a stepping thread; episode-end resets are eager
    int pending_reset;

    nle_ctx_t* ctx;
    nle_obs obs;
    nle_settings settings;
    char vardir[1024];

    // NLE-written buffers (inv_strs stays unbound: it alone pays for doname())
    short          glyphs[NH_GRID];
    long           blstats[NLE_BLSTATS_SIZE];
    unsigned char  chars[NH_GRID];
    unsigned char  message[NLE_MESSAGE_SIZE];
    int            misc[NLE_MISC_SIZE];
    int            internal[NLE_INTERNAL_SIZE];
    short          inv_glyphs[NLE_INVENTORY_SIZE];
    unsigned char  inv_letters[NLE_INVENTORY_SIZE];
    unsigned char  inv_oclasses[NLE_INVENTORY_SIZE];

    Stats stats;

    // reward-delta trackers, seeded from blstats at reset
    int prev_action;    // -1 at episode start
    long prev_score;    // logs only
    long prev_exp;
    long prev_gold;
    long start_gold;
    long prev_hp;
    long prev_hunger;   // clamped to [1,6]; Satiated counts as NotHungry
    long prev_time;
    int prev_depth;
    unsigned int rng;   // required by vecenv.h (seeded with env index)

    // reward coefs from config/nethack.ini, 0 disables a term. gold/exp/descent
    // decompose botl_score() = gold(net of start, floored at 0) + urexp
    // (4x uexp per kill, never drops) + 50*(deepest-1) — same signal,
    // separable knobs. The death step pays death_penalty - hp_coef*prev_hp
    // only; truncation pays neither.
    float gold_coef;
    float exp_coef;
    float descent_coef;
    float xp_coef;
    float scout_coef;
    float hp_coef;
    float hunger_coef;
    float illegal_penalty;
    float death_penalty;

    // advanced each reset; NetHack's own seeding can hang mklev/topologize
    unsigned long seed;
} Nethack;

// prompt handling + item macros (needs the struct above)
#include "macros.h"

static void nethack_bind_obs(Nethack* env) {
    nle_obs* o = &env->obs;
    memset(o, 0, sizeof(*o));
    o->glyphs   = env->glyphs;
    o->blstats  = env->blstats;
    o->chars    = env->chars;
    o->message  = env->message;
    o->misc     = env->misc;
    o->internal = env->internal;
    o->inv_glyphs   = env->inv_glyphs;
    o->inv_letters  = env->inv_letters;
    o->inv_oclasses = env->inv_oclasses;
}

static void nethack_init_settings(Nethack* env) {
    memset(&env->settings, 0, sizeof(env->settings));
    const char* source = getenv("NETHACKDIR");
    if (source == NULL) source = "./vendor/fast-nle/build/dat";

    if (nethack_make_vardir(source, env->vardir, sizeof(env->vardir)) != 0) {
        fprintf(stderr, "nethack: failed to create vardir from source=%s\n", source);
        strncpy(env->settings.hackdir, source, sizeof(env->settings.hackdir) - 1);
    } else {
        strncpy(env->settings.hackdir, env->vardir, sizeof(env->settings.hackdir) - 1);
    }
    env->settings.spawn_monsters = 1;
    env->settings.underfoot_glyphs = 1;   // hero tile shows top object/terrain, not the hero
    snprintf(env->settings.options, sizeof(env->settings.options), "@%s",
             nethack_rc_path(NETHACK_DEFAULT_OPTIONS));
    env->settings.fix_moon_phase = true;  // moon phase from seed, not host clock
}

// Expects a zeroed env (vecenv callocs; the demo memsets).
void init(Nethack* env) {
    env->seed = 0xCAFEBEEFUL + (unsigned long)env->rng;   // rng = env index
    // nle_start is deferred to the first c_reset (~180 ms per env).
    nethack_init_settings(env);
}

void c_close(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }
    nethack_rm_vardir(env->vardir);
    env->vardir[0] = '\0';
}

static void nethack_pack_obs(Nethack* env) {
    memcpy(env->observations + NETHACK_OFF_GLYPHS, env->glyphs, sizeof(env->glyphs));
    unsigned char* bl = env->observations + NETHACK_OFF_BLSTATS;
    for (int i = 0; i < NLE_BLSTATS_SIZE; i++) {
        uint32_t v = (uint32_t)(int32_t)env->blstats[i];
        bl[4*i + 0] = (unsigned char)(v & 0xffu);
        bl[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        bl[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        bl[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    int32_t extra[NETHACK_EXTRA_INTS] = {0};
    // ublesscnt ablation: the prayer clock is the one obs channel a human
    // can never see (prev_action + game time bound it; the exact re-roll is
    // privileged). Zeroed pending the smite-rate verdict; slot kept so the
    // obs layout is stable.
    extra[0] = 0;
    extra[1] = env->prev_action;
    for (int i = 0; i < NLE_INVENTORY_SIZE; i++) {
        int oc = env->inv_oclasses[i];
        if (oc >= NETHACK_NUM_OCLASSES) break;   // padded tail
        extra[2 + oc]++;
    }
    unsigned char* ex = env->observations + NETHACK_OFF_EXTRA;
    for (int i = 0; i < NETHACK_EXTRA_INTS; i++) {
        uint32_t v = (uint32_t)extra[i];
        ex[4*i + 0] = (unsigned char)(v & 0xffu);
        ex[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        ex[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        ex[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    // inventory entities: slot k's glyph at position k (slot head indexes
    // these); inv_oclasses is fully padded with 18, inv_glyphs is not
    unsigned char* iv = env->observations + NETHACK_OFF_INV;
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        uint16_t g = env->inv_oclasses[i] < NETHACK_NUM_OCLASSES
                   ? (uint16_t)env->inv_glyphs[i] : (uint16_t)NETHACK_PAD_GLYPH;
        iv[2*i + 0] = (unsigned char)(g & 0xffu);
        iv[2*i + 1] = (unsigned char)((g >> 8) & 0xffu);
    }

    // action mask, aligned with this obs: verbs all legal except EAT while
    // Satiated; four PER-VERB slot heads, each legal only for its own item
    // class (wear=armor 3, eat=food 7, quaff=potion 8, throw=weapon 2) —
    // verb-conditioned selection with class-tight masks. The prompt-answer
    // path still ESCs residual mismatches (worn armor, carried corpses).
    if (env->action_mask != NULL) {
        unsigned char* m = env->action_mask;
        memset(m, 1, NETHACK_NUM_ACTIONS);
        if (env->blstats[NLE_BL_HUNGER] == 0) m[NETHACK_ACT_EAT] = 0;
        static const int head_oc[5] = {3, 7, 8, 2, 11};   // wear, eat, quaff, throw, zap
        for (int h = 0; h < 5; h++) {
            unsigned char* s = m + NETHACK_NUM_ACTIONS + h * NETHACK_INV_SLOTS;
            memset(s, 0, NETHACK_INV_SLOTS);
            int any = 0;
            for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
                int oc = env->inv_oclasses[i];
                if (oc >= NETHACK_NUM_OCLASSES) break;
                if (oc == head_oc[h]) {
                    s[i] = 1; any = 1;
                    env->stats.valid_slot_sum += i; env->stats.valid_cnt++;
                }
            }
            if (!any) s[0] = 1;   // sampling needs >=1 legal entry per head
        }
        memset(m + NETHACK_NUM_ACTIONS + 5 * NETHACK_INV_SLOTS, 1, NETHACK_NUM_DIRS);
    }
}

// how = nle_obs.how_done, or -1 when truncated at the step cap
static void nethack_add_log(Nethack* env, int how) {
    env->log.perf            += (float)env->prev_score;
    env->log.score           += (float)env->prev_score;
    env->log.depth           += (float)env->prev_depth;
    env->log.valid_moves     += (float)env->stats.valid_moves;
    env->log.illegal_actions += (float)env->stats.illegal_actions;
    env->log.new_tiles       += (float)env->stats.new_tiles;
    env->log.max_depth       += (float)env->stats.max_depth;
    env->log.searches        += (float)env->stats.searches;
    env->log.engraves        += (float)env->stats.engraves;
    env->log.wears           += (float)env->stats.wears;
    env->log.eats            += (float)env->stats.eats;
    env->log.quaffs          += (float)env->stats.quaffs;
    env->log.prayers         += (float)env->stats.prayers;
    env->log.prayers_low_hp  += (float)env->stats.prayers_low_hp;
    env->log.throws          += (float)env->stats.throws;
    env->log.zaps            += (float)env->stats.zaps;
    env->log.floor_eats      += (float)env->stats.floor_eats;
    env->log.damage_taken    += (float)env->stats.damage;
    env->log.use_slot_mean += env->stats.use_cnt > 0
        ? (float)env->stats.use_slot_sum / (float)env->stats.use_cnt : 0.0f;
    env->log.valid_slot_mean += env->stats.valid_cnt > 0
        ? (float)env->stats.valid_slot_sum / (float)env->stats.valid_cnt : 0.0f;
    env->log.reward_saturated += env->stats.length > 0
        ? (float)env->stats.saturated / (float)env->stats.length : 0.0f;
    env->log.game_time       += (float)env->prev_time;
    env->log.max_xp_level    += (float)env->stats.max_xp;
    env->log.episode_return  += env->stats.ret;
    env->log.episode_length  += env->stats.length;
    if (how == -1)     env->log.truncated      += 1.0f;
    else if (how == 0) env->log.death_combat   += 1.0f;
    else if (how == 3) env->log.death_starved  += 1.0f;
    else if (how == NLE_HOW_WRATH) env->log.death_smited += 1.0f;
    if (how == 0) {
        env->log.death_mon_level    += (float)env->internal[NETHACK_INTERNAL_KILLER_MLEV];
        env->log.death_burst_turns  += (float)(env->prev_time - env->stats.last_ok_turn);
        env->log.death_adj_monsters += (float)env->stats.last_adj;
        env->log.death_maxhp        += (float)env->stats.last_maxhp;
    }
    else               env->log.death_other    += 1.0f;
    env->log.reach_mines      += (env->stats.areas & NETHACK_AREA_MINES)      ? 1.0f : 0.0f;
    env->log.reach_minetown   += (env->stats.areas & NETHACK_AREA_MINETOWN)   ? 1.0f : 0.0f;
    env->log.reach_deep_mines += (env->stats.areas & NETHACK_AREA_DEEP_MINES) ? 1.0f : 0.0f;
    env->log.reach_main_d5    += (env->stats.areas & NETHACK_AREA_MAIN_D5)    ? 1.0f : 0.0f;
    env->log.reach_sokoban    += (env->stats.areas & NETHACK_AREA_SOKOBAN)    ? 1.0f : 0.0f;
    env->log.n               += 1.0f;
}

// end the old game, seed+start a fresh one, drain the welcome prompts, zero
// the bookkeeping; concurrent resets are safe (NLE first-init is CAS-guarded)
static void nethack_do_reset(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }

    nethack_bind_obs(env);
    env->obs.how_done = -2;   // only really_done() sets it; abnormal end -> death_other

    // MMIX LCG advance; seeds = {core, disp} RNG streams, time_seed fixes the
    // in-game date/moon phase so games reproduce from seeds alone
    env->seed = env->seed * 6364136223846793005UL + 1442695040888963407UL;
    env->settings.initial_seeds.seeds[0] = env->seed;
    env->settings.initial_seeds.seeds[1] = env->seed ^ 0x9E3779B97F4A7C15UL;
    env->settings.initial_seeds.use_init_seeds = true;
    env->settings.time_seed = env->seed;
    env->settings.time_seed_is_set = true;
    env->ctx = nle_start(&env->obs, NULL, &env->settings);

    nethack_drain_prompts(env);

    env->prev_score = 0;
    env->prev_exp = env->blstats[NLE_BL_EXP];
    env->prev_gold = env->blstats[NLE_BL_GOLD];
    env->start_gold = env->prev_gold;
    env->prev_hp = env->blstats[NLE_BL_HP];
    env->prev_hunger = env->blstats[NLE_BL_HUNGER];
    if (env->prev_hunger < 1) env->prev_hunger = 1;
    else if (env->prev_hunger > 6) env->prev_hunger = 6;
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_action = -1;
    memset(&env->stats, 0, sizeof(env->stats));
    env->stats.max_depth = env->prev_depth;
    env->stats.max_xp = (int)env->blstats[NLE_BL_XP];
    nethack_pack_obs(env);
}

void c_reset(Nethack* env) {
    env->pending_reset = 1;
}

// log-only telemetry; runs before the reward, which advances prev_hp
static void nethack_update_stats(Nethack* env) {
    int depth = (int)env->blstats[NLE_BL_DEPTH];
    long dnum = env->blstats[NLE_BL_DNUM];
    if (dnum == 2) {
        env->stats.areas |= NETHACK_AREA_MINES;
        long mlvl = env->blstats[NLE_BL_DLEVEL];
        if (mlvl >= 3) env->stats.areas |= NETHACK_AREA_MINETOWN;
        if (mlvl >= 5) env->stats.areas |= NETHACK_AREA_DEEP_MINES;
    }
    else if (dnum == 0 && depth >= 5) env->stats.areas |= NETHACK_AREA_MAIN_D5;
    else if (dnum == 4) env->stats.areas |= NETHACK_AREA_SOKOBAN;

    long hp = env->blstats[NLE_BL_HP];
    if (hp < env->prev_hp) env->stats.damage += env->prev_hp - hp;

    // combat-death anatomy trackers, read back at death (blstats zero then)
    long hpmax = env->blstats[NLE_BL_HPMAX];
    if (4 * hp >= 3 * hpmax) env->stats.last_ok_turn = env->blstats[NLE_BL_TIME];
    env->stats.last_maxhp = hpmax;
    long hx = env->blstats[NLE_BL_X], hy = env->blstats[NLE_BL_Y];
    int adj = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            long r = hy + dy, c = hx + dx;
            if (r < 0 || r >= NH_ROWS || c < 0 || c >= NH_COLS) continue;
            int g = env->glyphs[r * NH_COLS + c];
            if (g >= 0 && g < NETHACK_NUMMONS) adj++;   // hostile monster range
        }
    env->stats.last_adj = adj;

    env->prev_score = env->blstats[NLE_BL_SCORE];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_depth = depth;
}

// never evaluated on the terminal step: NLE zeroes blstats when the game
// ends; the death step pays death_penalty + the HP cash-out instead (c_step)
static float nethack_shaped_reward(Nethack* env, int illegal) {
    int depth = (int)env->blstats[NLE_BL_DEPTH];
    long px = env->blstats[NLE_BL_X];
    long py = env->blstats[NLE_BL_Y];

    // exp points, positive deltas only — mirrors urexp, which never drops
    long exp = env->blstats[NLE_BL_EXP];
    float r = exp > env->prev_exp ? env->exp_coef * (float)(exp - env->prev_exp) : 0.0f;
    env->prev_exp = exp;

    // gold net of the starting purse, floored at 0, as botl_score() counts it
    long gold = env->blstats[NLE_BL_GOLD];
    long g_now = gold - env->start_gold;
    long g_prev = env->prev_gold - env->start_gold;
    if (g_now < 0) g_now = 0;
    if (g_prev < 0) g_prev = 0;
    r += env->gold_coef * (float)(g_now - g_prev);
    env->prev_gold = gold;

    // new episode-max depth only: the per-transition form was yo-yo farmable
    if (depth > env->stats.max_depth) {
        r += env->descent_coef * (float)(depth - env->stats.max_depth);
        env->stats.max_depth = depth;
    }

    // HP potential: heals repay damage; the death step cashes out the rest
    long hp = env->blstats[NLE_BL_HP];
    r += env->hp_coef * (float)(hp - env->prev_hp);
    env->prev_hp = hp;

    // hunger potential, linear below NotHungry (Satiated == NotHungry, so
    // overeating toward choke earns nothing): eating pays at the meal, decay
    // charges as it happens, potential form is farm-proof
    long hu = env->blstats[NLE_BL_HUNGER];
    if (hu < 1) hu = 1;
    else if (hu > 6) hu = 6;
    r += env->hunger_coef * (float)(env->prev_hunger - hu);
    env->prev_hunger = hu;

    // new episode-max experience level only, immune to drain re-earn
    int xp = (int)env->blstats[NLE_BL_XP];
    if (xp > env->stats.max_xp) {
        r += env->xp_coef * (float)(xp - env->stats.max_xp);
        env->stats.max_xp = xp;
    }

    // scout: first visit to a tile this episode; bitmaps block stair yo-yo
    if (px >= 0 && px < NH_COLS && py >= 0 && py < NH_ROWS) {
        int d = depth < 1 ? 0 : (depth > NETHACK_MAX_DEPTH ? NETHACK_MAX_DEPTH - 1 : depth - 1);
        int bit = (int)py * NH_COLS + (int)px;
        unsigned char mask = (unsigned char)(1 << (bit & 7));
        if (!(env->stats.visited[d][bit >> 3] & mask)) {
            env->stats.visited[d][bit >> 3] |= mask;
            r += env->scout_coef;
            env->stats.new_tiles++;
        }
    }

    if (illegal) r += env->illegal_penalty;
    return r;
}

void c_step(Nethack* env) {
    if (env->pending_reset) {
        env->pending_reset = 0;
        nethack_do_reset(env);
    }

    int a = (int)env->actions[0];
    if (a < 0 || a >= NETHACK_NUM_ACTIONS) a = 0;
    // per-verb slot heads (verb-conditioned item selection): actions[1..4] =
    // wear/eat/quaff/throw slots; only the sampled verb's head is consumed
    int slot = 0;
    switch (a) {
    case NETHACK_ACT_WEAR:  slot = (int)env->actions[1]; break;
    case NETHACK_ACT_EAT:   slot = (int)env->actions[2]; break;
    case NETHACK_ACT_QUAFF: slot = (int)env->actions[3]; break;
    case NETHACK_ACT_THROW: slot = (int)env->actions[4]; break;
    case NETHACK_ACT_ZAP:   slot = (int)env->actions[5]; break;
    }
    if (slot < 0 || slot >= NETHACK_INV_SLOTS) slot = 0;
    int dir = (int)env->actions[6];    // direction head, consumed by MOVE/RUN/KICK/THROW/ZAP
    if (dir < 0 || dir >= NETHACK_NUM_DIRS) dir = 0;
    int dirkey = NETHACK_DIR_KEYS[dir];

    long time_before = env->blstats[NLE_BL_TIME];
    int stepped = 1, bad_pick = 0, used;
    switch (a) {
    case NETHACK_ACT_MOVE:
        env->obs.action = dirkey;
        env->ctx = nle_step(env->ctx, &env->obs);
        break;
    case NETHACK_ACT_RUN:
        env->obs.action = dirkey - 32;   // uppercase vi-key = run
        env->ctx = nle_step(env->ctx, &env->obs);
        break;
    case NETHACK_ACT_DOWN:
        env->obs.action = '>';
        env->ctx = nle_step(env->ctx, &env->obs);
        break;
    case NETHACK_ACT_UP:
        env->obs.action = '<';
        env->ctx = nle_step(env->ctx, &env->obs);
        break;
    case NETHACK_ACT_KICK:
        env->obs.action = 4;   // ^D
        env->ctx = nle_step(env->ctx, &env->obs);
        // answer the direction prompt in-step: kick is atomic now
        if (!env->obs.done && env->misc[NETHACK_MISC_YN]
            && nethack_msg_contains(env, "n what direction")) {
            env->obs.action = dirkey;
            env->ctx = nle_step(env->ctx, &env->obs);
        }
        break;
    case NETHACK_ACT_SEARCH:
        env->obs.action = 's';
        env->stats.searches++;
        env->ctx = nle_step(env->ctx, &env->obs);
        break;
    case NETHACK_ACT_ELBERETH:
        env->stats.engraves++;
        nethack_do_elbereth(env);
        break;
    case NETHACK_ACT_WEAR:
        used = nethack_do_use_item(env, 'W', "want to wear", NULL, slot);
        if (used > 0) {
            env->stats.wears++;
            env->stats.use_slot_sum += slot; env->stats.use_cnt++;
        }
        else if (used < 0) bad_pick = 1;
        break;
    case NETHACK_ACT_EAT:
        // satiated gate: eating past Satiated is NetHack's choke() death
        if (env->blstats[NLE_BL_HUNGER] == 0) stepped = 0;
        else {
            long fe0 = env->stats.floor_eats;
            used = nethack_do_use_item(env, 'e', "want to eat", "eat it", slot);
            if (used > 0) {
                env->stats.eats++;
                if (env->stats.floor_eats == fe0) {   // slot-directed, not floor
                    env->stats.use_slot_sum += slot; env->stats.use_cnt++;
                }
            }
            else if (used < 0) bad_pick = 1;
        }
        break;
    case NETHACK_ACT_QUAFF:
        used = nethack_do_use_item(env, 'q', "want to drink", "rink from the", slot);
        if (used > 0) {
            env->stats.quaffs++;
            env->stats.use_slot_sum += slot; env->stats.use_cnt++;
        }
        else if (used < 0) bad_pick = 1;
        break;
    case NETHACK_ACT_THROW:
        used = nethack_do_use_item(env, 't', "want to throw", NULL, slot);
        if (used > 0) {
            env->stats.throws++;
            env->stats.use_slot_sum += slot; env->stats.use_cnt++;
            // answer the direction prompt in-step: throw is atomic now
            if (!env->obs.done && env->misc[NETHACK_MISC_YN]
                && nethack_msg_contains(env, "n what direction")) {
                env->obs.action = dirkey;
                env->ctx = nle_step(env->ctx, &env->obs);
            }
        }
        else if (used < 0) bad_pick = 1;
        break;
    case NETHACK_ACT_ZAP:
        used = nethack_do_use_item(env, 'z', "want to zap", NULL, slot);
        if (used > 0) {
            env->stats.zaps++;
            env->stats.use_slot_sum += slot; env->stats.use_cnt++;
            // directional wands raise "In what direction?" — answer in-step
            if (!env->obs.done && env->misc[NETHACK_MISC_YN]
                && nethack_msg_contains(env, "n what direction")) {
                env->obs.action = dirkey;
                env->ctx = nle_step(env->ctx, &env->obs);
            }
        }
        else if (used < 0) bad_pick = 1;
        break;
    case NETHACK_ACT_PRAY: {
        long hp = env->blstats[NLE_BL_HP], hpmax = env->blstats[NLE_BL_HPMAX];
        if (4 * hp <= hpmax) env->stats.prayers_low_hp++;   // HP at decision time
        env->stats.prayers++;
        env->obs.action = 0x80 | 'p';
        env->ctx = nle_step(env->ctx, &env->obs);
        break;
    }
    default:
        env->obs.action = dirkey;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
    env->prev_action = a;
    int illegal = stepped ? nethack_handle_prompts(env) : 0;
    if (bad_pick) { illegal = 1; env->stats.illegal_actions++; }
    if (env->blstats[NLE_BL_TIME] > time_before) env->stats.valid_moves++;
    env->stats.length++;

    float reward;
    if (env->obs.done) {
        reward = env->death_penalty - env->hp_coef * (float)env->prev_hp;
    } else {
        nethack_update_stats(env);
        reward = nethack_shaped_reward(env, illegal);
    }
    env->rewards[0] = reward;
    env->stats.ret += reward;
    if (reward > 1.0f || reward < -1.0f) env->stats.saturated++;

    int done = env->obs.done || env->stats.length >= NETHACK_MAX_EPISODE_STEPS;
    env->terminals[0] = done ? 1.0f : 0.0f;   // truncation reported as terminal too
    if (done) {
        nethack_add_log(env, env->obs.done ? env->obs.how_done : -1);
        // eager same-thread reset: the terminal step returns the fresh obs
        nethack_do_reset(env);
    } else {
        nethack_pack_obs(env);
    }
}

void c_render(Nethack* env) {
    printf("\x1b[H\x1b[2J");
    for (int r = 0; r < NH_ROWS; r++) {
        for (int c = 0; c < NH_COLS; c++) {
            unsigned char ch = env->chars[r * NH_COLS + c];
            putchar(ch ? ch : ' ');
        }
        putchar('\n');
    }
    printf("HP %ld/%ld  AC %ld  Dlvl %ld  Score %ld  T %ld\n",
           env->blstats[NLE_BL_HP], env->blstats[NLE_BL_HPMAX],
           env->blstats[NLE_BL_AC], env->blstats[NLE_BL_DEPTH],
           env->blstats[NLE_BL_SCORE], env->blstats[NLE_BL_TIME]);
    printf("Msg: %.*s\n", NLE_MESSAGE_SIZE, env->message);
    fflush(stdout);
}
