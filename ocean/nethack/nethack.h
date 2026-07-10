// NetHack env for the pufferlib vecenv. Opinionated, fixed configuration:
//   obs     = 21x21 egocentric glyph crop (int16 LE) + 27 blstats (int32 LE)
//             + 20 extra stats (int32 LE: prayer cooldown, previous action,
//             18 per-class inventory counts), 1070 bytes — exactly what the
//             custom CUDA encoder in src/nethack.cu expects. chars/message/inv_*
//             are kept in side buffers for the standalone tools and the item
//             macros but never enter the obs tensor raw.
//   actions = 24: 8 compass moves, 8 run-moves, down (>), up (<), kick (^D),
//             search (s), pray (M-p), engrave Elbereth (macro: E - Elbereth),
//             wear armor (macro: W + listed letter), eat (macro: e + letter).
//   reward  = gold + exp + descent + xp + scout + hp + illegal penalty
//             + death penalty, per-env coefs. gold/exp/descent decompose
//             NetHack's score (see the coef block comment).
// One Nethack struct per agent; each owns an nle_ctx_t (all per-game NLE
// state) and a private vardir on tmpfs for NetHack's file I/O.
// libnethack.so is linked directly at build time.
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

// ---------------------------------------------------------------------------
// Fixed configuration
// ---------------------------------------------------------------------------
#define NH_ROWS 21
#define NH_COLS 79
#define NH_GRID (NH_ROWS * NH_COLS)

// Egocentric crop: odd side length, centered on the agent, padded with
// NO_GLYPH (== MAX_GLYPH == 5976 in NetHack 3.6.6) outside the map.
#define NETHACK_CROP       21
#define NETHACK_CROP_GRID  (NETHACK_CROP * NETHACK_CROP)
#define NETHACK_PAD_GLYPH  5976

// Corpse glyph range [GLYPH_BODY_OFF, +NUMMONS) from display.h, NetHack 3.6.6
#define NETHACK_GLYPH_BODY_OFF 1144
#define NETHACK_NUMMONS        381

// Obs tensor layout (byte offsets). Glyphs, blstats, then extra stats:
// [prayer cooldown (u.ublesscnt, vendored into internal[5]) | previous
// action index, -1 at episode start | 18 per-class inventory item counts].
#define NETHACK_NUM_OCLASSES 18   // MAXOCLASSES (objclass.h); inv_oclasses pads with 18
#define NETHACK_OFF_GLYPHS  0
#define NETHACK_OFF_BLSTATS (NETHACK_CROP_GRID * 2)
#define NETHACK_OFF_EXTRA   (NETHACK_OFF_BLSTATS + NLE_BLSTATS_SIZE * 4)
#define NETHACK_EXTRA_INTS  (2 + NETHACK_NUM_OCLASSES)
#define NETHACK_OBS_SIZE    (NETHACK_OFF_EXTRA + NETHACK_EXTRA_INTS * 4)
#define NETHACK_INTERNAL_UBLESSCNT 5   // vendored winrl.cc patch (was core seed)

#define NETHACK_MAX_EPISODE_STEPS 10000
#define NETHACK_AUTODISMISS_MAX   64   // cap on prompt-dismiss keystrokes per step
#define NETHACK_MAX_DEPTH         64   // scout bitmaps tracked per episode

// episode_areas bits; logged as reach_* proportions
#define NETHACK_AREA_MINES      1u   // Gnomish Mines (dnum 2)
#define NETHACK_AREA_MINETOWN   2u   // Mines level 3+ (Minetown band)
#define NETHACK_AREA_DEEP_MINES 4u   // Mines level 5+ (past Minetown)
#define NETHACK_AREA_MAIN_D5    8u   // Dungeons of Doom depth 5+ (Oracle route)
#define NETHACK_AREA_SOKOBAN    16u  // Sokoban (dnum 4)

// nle_obs.misc[] prompt-state flags
enum { NETHACK_MISC_YN = 0, NETHACK_MISC_GETLIN = 1, NETHACK_MISC_XWAIT = 2 };

#define NETHACK_NUM_ACTIONS 24
static const int NETHACK_ACTION_TABLE[NETHACK_NUM_ACTIONS] = {
    'k','j','h','l','y','u','b','n',   // N S W E NW NE SW SE
    'K','J','H','L','Y','U','B','N',   // run variants
    '>','<',                           // stairs down, up
    4,                                 // ^D kick (locked doors); asks a direction
    's',                               // search once (hidden doors/passages)
    0x80|'p',                          // M-p pray (heals low HP, cures starving)
    'E',                               // engrave Elbereth; macro, see nethack_do_elbereth
    'W',                               // wear armor; macro, see nethack_do_use_item
    'e',                               // eat carried food; macro, see nethack_do_use_item
};

// !status_updates: disables the status renderer and recalc_mapseen
// (~25% of engine instructions); blstats come from update_blstats().
#define NETHACK_DEFAULT_OPTIONS \
    "name:Agent-mon-hum-neu-mal," \
    "autopickup,color,disclose:+i +a +v +g +c +o," \
    "mention_walls,nobones,nocmdassist,nolegacy,nosparkle," \
    "pickup_burden:unencumbered,pickup_types:$[%," \
    "runmode:teleport,showexp,showscore,time," \
    "!status_updates"

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float depth;
    float valid_moves;        // c_steps where the action advanced NetHack's turn counter
    float illegal_actions;    // c_steps where the action hit a sub-prompt we ESC'd out of
    float new_tiles;          // unique tiles entered per episode
    float max_depth;          // deepest level reached this episode (depth-at-death can under-report)
    float prayers;            // prayer actions per episode
    float prayers_low_hp;     // prayers issued at <=25% max HP (learned panic button)
    float searches;           // search actions per episode
    float engraves;           // Elbereth macro actions per episode
    float wears;              // wear macro presses that chose an item
    float eats;               // eat macro presses that chose an item
    float damage_taken;       // sum of HP decreases per episode
    float reward_saturated;   // fraction of steps with |reward| > 1 (trainer clamps there)
    float game_time;          // NetHack turns survived (run-moves advance many per step)
    float max_xp_level;       // highest experience level reached
    // Episode end reason, one-hot per episode (game_end_types in hack.h)
    float death_combat;       // DIED (monsters, traps)
    float death_starved;      // STARVING
    float death_other;        // any other how_done (drowning, poison, quit, ...)
    float truncated;          // hit NETHACK_MAX_EPISODE_STEPS
    // Areas reached, 0/1 per episode; the logged mean is the proportion
    float reach_mines;
    float reach_minetown;
    float reach_deep_mines;
    float reach_main_d5;
    float reach_sokoban;
    float n;
} Log;

typedef struct Nethack {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    // Lazy reset: c_reset sets this flag, c_step performs the actual reset.
    // Guarantees reset and step run on the same OMP thread (NLE's coroutine
    // captures the stack pointer; resetting on another thread corrupts it).
    int pending_reset;

    nle_ctx_t* ctx;
    nle_obs obs;
    nle_settings settings;
    char vardir[1024];

    // NLE-written buffers. glyphs+blstats feed the obs tensor; chars/message
    // are for the standalone tools and the '?'-prompt heuristic; misc/internal
    // drive prompt auto-dismissal; inv_* feed the item macros and the
    // per-class count obs (inv_strs stays unbound — it alone pays for
    // doname() string formatting in the vendored NLE).
    short          glyphs[NH_GRID];
    long           blstats[NLE_BLSTATS_SIZE];
    unsigned char  chars[NH_GRID];
    unsigned char  message[NLE_MESSAGE_SIZE];
    int            misc[NLE_MISC_SIZE];
    int            internal[NLE_INTERNAL_SIZE];
    short          inv_glyphs[NLE_INVENTORY_SIZE];
    unsigned char  inv_letters[NLE_INVENTORY_SIZE];
    unsigned char  inv_oclasses[NLE_INVENTORY_SIZE];

    // Per-dungeon-level exploration bitmaps, cleared once per episode. Indexed
    // by depth so revisiting a level cannot re-earn scout reward (branch
    // levels sharing a depth share a bitmap; slight under-count, acceptable).
    unsigned char visited[NETHACK_MAX_DEPTH][(NH_GRID + 7) / 8];

    long episode_valid_moves;
    long episode_illegal_actions;
    long episode_new_tiles;
    long episode_prayers;
    long episode_prayers_low_hp;
    long episode_searches;
    long episode_engraves;
    long episode_wears;
    long episode_eats;
    long episode_damage;
    int prev_action;    // last executed action index, -1 at episode start
    long prev_score;    // logs only (perf/score metrics); not rewarded directly
    long prev_exp;
    long prev_gold;
    long start_gold;    // starting purse; score counts gold net of it, floored at 0
    long prev_hp;
    long prev_time;
    long episode_saturated;   // steps with |reward| > 1 (trainer clamp range)
    int prev_depth;
    int episode_max_depth;
    unsigned episode_areas;
    int episode_max_xp;
    float episode_return;
    int episode_length;
    unsigned int rng;   // required by vecenv.h (seeded with env index)

    // Reward shaping coefficients, all independently toggleable (0 = off).
    // Set from kwargs in binding.c via `[env]` keys in config/nethack.ini.
    // Score is rewarded DECOMPOSED rather than as one score_coef: botl_score()
    // = gold(net of start, floored at 0) + urexp + 50*(deepest-1), and
    // urexp accrues 4x uexp on kills and never drops on drain. So
    // score_coef*dscore == gold_coef*dgold + exp_coef*max(0,duexp)
    // + 50*score_coef*ddeepest with gold_coef = score_coef, exp_coef =
    // 4*score_coef — same signal, separable knobs.
    //   reward = gold_coef    * d(max(0, gold - start_gold))
    //          + exp_coef     * max(0, exp - prev_exp)   (dense per kill)
    //          + descent_coef * (depth - episode_max_depth), new maxima only
    //          + xp_coef      * (xp level - episode_max_xp), new maxima only
    //          + scout_coef   * (1 if new tile on this level this episode)
    //          + hp_coef      * (hp - prev_hp)   (potential; heals reward, damage costs)
    //          + illegal_penalty * (1 if action triggered a sub-prompt)
    // The death step carries no shaping (NLE zeroes blstats when the game
    // ends, so the terms would read garbage); it carries death_penalty plus
    // the HP potential cash-out (-hp_coef * prev_hp — dying at high HP
    // forfeits more). Truncation at the step cap carries neither.
    float gold_coef;
    float exp_coef;
    float descent_coef;
    float xp_coef;
    float scout_coef;
    float hp_coef;
    float illegal_penalty;
    float death_penalty;

    // Explicit per-env seeds for nle_start; NetHack's own seeding can hit
    // infinite loops in level generation (mklev/topologize).
    unsigned long seed_a;
    unsigned long seed_b;
} Nethack;

// ---------------------------------------------------------------------------
// NLE plumbing
// ---------------------------------------------------------------------------
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

// True when the current message ends with '?' — catches single-key prompts
// NLE doesn't expose via misc[] ("In what direction...?"). Real game messages
// ("It's a wall.") don't end in '?'.
static int nethack_msg_is_prompt(const Nethack* env) {
    const unsigned char* m = env->message;
    if (!m[0]) return 0;
    int e = 0;
    while (e < NLE_MESSAGE_SIZE && m[e]) e++;
    while (e > 0 && m[e-1] == ' ') e--;
    return e > 0 && m[e-1] == '?';
}

static int nethack_msg_contains(const Nethack* env, const char* needle) {
    char buf[NLE_MESSAGE_SIZE + 1];
    memcpy(buf, env->message, NLE_MESSAGE_SIZE);
    buf[NLE_MESSAGE_SIZE] = '\0';
    return strstr(buf, needle) != NULL;
}

// Auto-dismiss passive prompts (welcome screen, --More--, getline) until the
// game is back at the main command prompt.
static void nethack_drain_prompts(Nethack* env) {
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]) break;
        env->obs.action = yn ? 27 : '\r';
        env->ctx = nle_step(env->ctx, &env->obs);
    }
}

// The agent's key can land the game in a sub-prompt it can't answer itself
// ("Really attack?", "In what direction?", --More--). yn prompts commit the
// agent's choice with 'y' — EXCEPT the dlvl-1 up-stairs confirm ("Beware,
// there will be no return! Still climb?"), where 'y' ends the game as
// ESCAPED, and the peaceful-attack confirm ("Really attack the watchman?"),
// where 'y' hostilizes the Minetown watch/shopkeepers and gets low-level
// agents killed; those and everything else are dismissed with ESC. Returns 1
// if the action triggered a sub-prompt (the illegal_penalty condition).
static int nethack_handle_prompts(Nethack* env) {
    // Direction prompts ("In what direction?" after kick) are answerable:
    // leave them live — the agent's next key is consumed as the direction.
    // They arrive via yn_function, so this must precede the yn auto-answer.
    if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "n what direction"))
        return 0;
    // The pray confirm is a deliberate action's own prompt, not an illegal
    // sub-prompt: commit it below but exempt it from the penalty.
    int praying = env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "to pray");
    int illegal = !praying && (env->misc[NETHACK_MISC_YN] || env->misc[NETHACK_MISC_GETLIN]
               || nethack_msg_is_prompt(env));
    if (!illegal && !praying) {
        if (env->misc[NETHACK_MISC_XWAIT]) nethack_drain_prompts(env);
        return 0;
    }
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]
            && !nethack_msg_is_prompt(env)) break;
        int commit = yn && !nethack_msg_contains(env, "no return")
                        && !nethack_msg_contains(env, "eally attack");
        env->obs.action = commit ? 'y' : 27;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
    if (illegal) env->episode_illegal_actions++;
    return illegal;
}

// Action 21 macro: engrave Elbereth with a fingertip (E, '-', "Elbereth", RET).
// Most melee monsters won't step onto the square while it's legible, making
// this the early game's strongest panic button. Every stage is gated on its
// expected prompt (the getobj stylus prompt polls through yn_function), so
// aborts — headstone, levitation, a pending kick direction — fall through to
// the generic prompt cleanup in nethack_handle_prompts.
static void nethack_do_elbereth(Nethack* env) {
    env->obs.action = 'E';
    env->ctx = nle_step(env->ctx, &env->obs);
    if (env->obs.done || !env->misc[NETHACK_MISC_YN]
        || !nethack_msg_contains(env, "write with")) return;
    env->obs.action = '-';
    env->ctx = nle_step(env->ctx, &env->obs);
    // Flag-priority loop: "You write in the dust..." raises an xwait --More--
    // ALONGSIDE the getlin, and it swallows any key but space/return — clear
    // it before typing. An existing dust engraving asks "add to the current
    // engraving?" first: decline, wiping it, so the fresh full "Elbereth"
    // replaces whatever half-smudged text was there.
    const char* c = "Elbereth\r";
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done && *c; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN]
                 && nethack_msg_contains(env, "current engraving")) env->obs.action = 'n';
        else if (env->misc[NETHACK_MISC_GETLIN]) env->obs.action = (unsigned char)*c++;
        else break;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
}

// Corpses are excluded from the eat macro: their age is invisible and old or
// poisonous ones kill (food poisoning = the choke death's quieter sibling).
static int nethack_letter_is_corpse(const Nethack* env, char letter) {
    for (int i = 0; i < NLE_INVENTORY_SIZE && env->inv_letters[i]; i++)
        if ((char)env->inv_letters[i] == letter) {
            int g = env->inv_glyphs[i];
            return g >= NETHACK_GLYPH_BODY_OFF
                && g < NETHACK_GLYPH_BODY_OFF + NETHACK_NUMMONS;
        }
    return 0;
}

// Random candidate letter from a getobj prompt's bracket list, e.g.
// "What do you want to wear? [b-d f or ?*]". Random, not first: an item the
// game refuses (second cloak, blocked slot) stays listed, and a first-letter
// pick would retry it forever while everything behind it is never chosen.
// 0 if the list is empty/unparseable.
static int nethack_pick_candidate(Nethack* env) {
    const unsigned char* m = env->message;
    int i = 0;
    while (i < NLE_MESSAGE_SIZE && m[i] && m[i] != '[') i++;
    char cand[52];
    int n = 0;
    for (i++; i < NLE_MESSAGE_SIZE && m[i] && n < (int)sizeof(cand); i++) {
        unsigned char c = m[i];
        if (n == 0 && (c == '-' || c == ' ')) continue;   // allownone's leading "- "
        if (c == '-' && i + 1 < NLE_MESSAGE_SIZE) {       // compactified run
            for (char x = cand[n-1] + 1; x <= (char)m[i+1] && n < (int)sizeof(cand); x++)
                cand[n++] = x;
            i++;
            continue;
        }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) cand[n++] = (char)c;
        else break;   // ' ' before "or ?*", ']', '#', ...: end of the letter list
    }
    int k = 0;
    for (int j = 0; j < n; j++)
        if (!nethack_letter_is_corpse(env, cand[j])) cand[k++] = cand[j];
    n = k;
    if (n == 0) return 0;
    env->rng = env->rng * 1103515245u + 12345u;
    return cand[(env->rng >> 16) % (unsigned)n];
}

// Actions 22/23 macros: use a carried item through a getobj command (W = wear
// armor, e = eat). getobj filters the prompt's candidate list to items valid
// for the verb, so the macro answers with a random listed letter and the game
// does the rest (multi-turn donning runs as an occupation). `decline` skips
// leading floor-item offers ("There is a lichen corpse here; eat it?") — floor
// corpses have hidden age and old ones are lethal; carried food is safe.
// Nothing usable -> no prompt -> clean no-op. Unexpected prompts fall through
// to nethack_handle_prompts. Returns 1 if an item letter was sent.
static int nethack_do_use_item(Nethack* env, int cmd, const char* gate, const char* decline) {
    env->obs.action = cmd;
    env->ctx = nle_step(env->ctx, &env->obs);
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN] && decline && nethack_msg_contains(env, decline))
            env->obs.action = 'n';
        else if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, gate)) {
            int c = nethack_pick_candidate(env);
            // no acceptable candidate (e.g. corpses only): dismiss the prompt
            // here, penalty-free — same outcome as having nothing usable
            env->obs.action = c ? c : 27;
            env->ctx = nle_step(env->ctx, &env->obs);
            return c != 0;
        }
        else return 0;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
    return 0;
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
    strncpy(env->settings.options, NETHACK_DEFAULT_OPTIONS, sizeof(env->settings.options) - 1);
    env->settings.fix_moon_phase = true;  // moon phase from seed, not host clock
}

void init(Nethack* env) {
    env->ctx = NULL;
    env->prev_score = 0;
    env->prev_depth = 1;
    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->vardir[0] = '\0';

    // Defaults used when a binding doesn't pass kwargs (standalone tools).
    env->gold_coef       = 0.0f;
    env->exp_coef        = 0.1f;
    env->descent_coef    = 1.0f;
    env->xp_coef         = 0.0f;
    env->scout_coef      = 0.1f;
    env->hp_coef         = 0.0f;
    env->illegal_penalty = -0.5f;
    env->death_penalty   = -1.0f;

    // Per-env seeds derived from env->rng (vecenv sets it to the env index)
    // and a process-wide base overridable via NETHACK_SEED_BASE.
    unsigned long base = 0xCAFEBEEFUL;
    const char* sb = getenv("NETHACK_SEED_BASE");
    if (sb) {
        char* end = NULL;
        unsigned long v = strtoul(sb, &end, 0);
        if (end && end != sb) base = v;
    }
    env->seed_a = base + (unsigned long)env->rng;
    env->seed_b = (base ^ 0x9E3779B97F4A7C15UL) + (unsigned long)env->rng;
    // nle_start is deferred to the first c_reset (~180 ms per env).
    nethack_init_settings(env);
    nethack_bind_obs(env);
}

void c_close(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }
    nethack_rm_vardir(env->vardir);
    env->vardir[0] = '\0';
}

// ---------------------------------------------------------------------------
// Obs packing: egocentric glyph crop + blstats truncated to int32
// ---------------------------------------------------------------------------
static void nethack_pack_obs(Nethack* env) {
    int cx = (int)env->blstats[NLE_BL_X];   // 0-based, aligned with glyphs
    int cy = (int)env->blstats[NLE_BL_Y];
    int half = NETHACK_CROP / 2;
    short* dst = (short*)(env->observations + NETHACK_OFF_GLYPHS);
    for (int r = 0; r < NETHACK_CROP; r++) {
        short* row = dst + r * NETHACK_CROP;
        int gy = cy - half + r;
        if (gy < 0 || gy >= NH_ROWS) {
            for (int c = 0; c < NETHACK_CROP; c++) row[c] = NETHACK_PAD_GLYPH;
            continue;
        }
        int gx0 = cx - half;
        int c0 = gx0 < 0 ? -gx0 : 0;
        int c1 = gx0 + NETHACK_CROP > NH_COLS ? NH_COLS - gx0 : NETHACK_CROP;
        for (int c = 0; c < c0; c++) row[c] = NETHACK_PAD_GLYPH;
        memcpy(row + c0, env->glyphs + gy * NH_COLS + gx0 + c0, (size_t)(c1 - c0) * sizeof(short));
        for (int c = c1; c < NETHACK_CROP; c++) row[c] = NETHACK_PAD_GLYPH;
    }
    unsigned char* bl = env->observations + NETHACK_OFF_BLSTATS;
    for (int i = 0; i < NLE_BLSTATS_SIZE; i++) {
        uint32_t v = (uint32_t)(int32_t)env->blstats[i];
        bl[4*i + 0] = (unsigned char)(v & 0xffu);
        bl[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        bl[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        bl[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    int32_t extra[NETHACK_EXTRA_INTS] = {0};
    extra[0] = (int32_t)env->internal[NETHACK_INTERNAL_UBLESSCNT];
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
}

// how = nle_obs.how_done when the game ended (0=DIED, 3=STARVING, ...),
// -1 when the episode was truncated at the step cap.
static void nethack_add_log(Nethack* env, int how) {
    env->log.perf            += (float)env->prev_score;
    env->log.score           += (float)env->prev_score;
    env->log.depth           += (float)env->prev_depth;
    env->log.valid_moves     += (float)env->episode_valid_moves;
    env->log.illegal_actions += (float)env->episode_illegal_actions;
    env->log.new_tiles       += (float)env->episode_new_tiles;
    env->log.max_depth       += (float)env->episode_max_depth;
    env->log.prayers         += (float)env->episode_prayers;
    env->log.prayers_low_hp  += (float)env->episode_prayers_low_hp;
    env->log.searches        += (float)env->episode_searches;
    env->log.engraves        += (float)env->episode_engraves;
    env->log.wears           += (float)env->episode_wears;
    env->log.eats            += (float)env->episode_eats;
    env->log.damage_taken    += (float)env->episode_damage;
    env->log.reward_saturated += env->episode_length > 0
        ? (float)env->episode_saturated / (float)env->episode_length : 0.0f;
    env->log.game_time       += (float)env->prev_time;
    env->log.max_xp_level    += (float)env->episode_max_xp;
    env->log.episode_return  += env->episode_return;
    env->log.episode_length  += env->episode_length;
    if (how == -1)     env->log.truncated      += 1.0f;
    else if (how == 0) env->log.death_combat   += 1.0f;
    else if (how == 3) env->log.death_starved  += 1.0f;
    else               env->log.death_other    += 1.0f;
    env->log.reach_mines      += (env->episode_areas & NETHACK_AREA_MINES)      ? 1.0f : 0.0f;
    env->log.reach_minetown   += (env->episode_areas & NETHACK_AREA_MINETOWN)   ? 1.0f : 0.0f;
    env->log.reach_deep_mines += (env->episode_areas & NETHACK_AREA_DEEP_MINES) ? 1.0f : 0.0f;
    env->log.reach_main_d5    += (env->episode_areas & NETHACK_AREA_MAIN_D5)    ? 1.0f : 0.0f;
    env->log.reach_sokoban    += (env->episode_areas & NETHACK_AREA_SOKOBAN)    ? 1.0f : 0.0f;
    env->log.n               += 1.0f;
}

// End the previous game (if any), start a fresh one, drain the welcome
// prompts, zero the episode bookkeeping. Concurrent resets across envs are
// safe: first-init paths in the vendored NLE are CAS-guarded idempotent and
// per-game state lives on nle_ctx_t.
static void nethack_do_reset(Nethack* env) {

    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }

    nethack_bind_obs(env);
    env->obs.how_done = -2;   // only really_done() sets it; abnormal end -> death_other

    // Advance the per-env LCG (Numerical Recipes / MMIX constants) so each
    // reset gets a fresh, known-safe seed.
    env->seed_a = env->seed_a * 6364136223846793005UL + 1442695040888963407UL;
    env->seed_b = env->seed_b * 6364136223846793005UL + 1442695040888963407UL;
    env->settings.initial_seeds.seeds[0] = env->seed_a;
    env->settings.initial_seeds.seeds[1] = env->seed_b;
    env->settings.initial_seeds.use_init_seeds = true;
    env->settings.time_seed = env->seed_a;
    env->settings.time_seed_is_set = true;
    env->ctx = nle_start(&env->obs, NULL, &env->settings);

    nethack_drain_prompts(env);

    env->prev_score = 0;
    env->prev_exp = env->blstats[NLE_BL_EXP];
    env->prev_gold = env->blstats[NLE_BL_GOLD];
    env->start_gold = env->prev_gold;
    env->prev_hp = env->blstats[NLE_BL_HP];
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->episode_max_depth = env->prev_depth;
    env->episode_areas = 0;
    env->episode_max_xp = (int)env->blstats[NLE_BL_XP];
    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->episode_valid_moves = 0;
    env->episode_illegal_actions = 0;
    env->episode_new_tiles = 0;
    env->episode_prayers = 0;
    env->episode_prayers_low_hp = 0;
    env->episode_searches = 0;
    env->episode_engraves = 0;
    env->episode_wears = 0;
    env->episode_eats = 0;
    env->episode_damage = 0;
    env->episode_saturated = 0;
    env->prev_action = -1;
    memset(env->visited, 0, sizeof(env->visited));
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    nethack_pack_obs(env);
}

void c_reset(Nethack* env) {
    env->pending_reset = 1;
}

// reward = gold + exp + one-shot descent/XL bonuses + first-visit scout bonus
// + HP potential + illegal penalty (see the coef block comment for the exact
// form). Never evaluated on the terminal step: NLE zeroes blstats when the
// game ends, so the terms would read garbage there. The terminal step gets
// death_penalty plus the HP cash-out instead (see c_step).
static float nethack_shaped_reward(Nethack* env, int illegal) {
    int depth = (int)env->blstats[NLE_BL_DEPTH];
    long px = env->blstats[NLE_BL_X];
    long py = env->blstats[NLE_BL_Y];

    // Exp points, positive deltas only: dense per-kill signal. The positive
    // part mirrors score's urexp, which accrues on gains but never drops on
    // level drain (losexp touches uexp only).
    long exp = env->blstats[NLE_BL_EXP];
    float r = exp > env->prev_exp ? env->exp_coef * (float)(exp - env->prev_exp) : 0.0f;
    env->prev_exp = exp;

    // Gold, net of the starting purse and floored at 0, exactly as
    // botl_score() counts it (theft below the starting purse scores nothing).
    long gold = env->blstats[NLE_BL_GOLD];
    long g_now = gold - env->start_gold;
    long g_prev = env->prev_gold - env->start_gold;
    if (g_now < 0) g_now = 0;
    if (g_prev < 0) g_prev = 0;
    r += env->gold_coef * (float)(g_now - g_prev);
    env->prev_gold = gold;

    // Descent: paid only on new episode-max depth. The per-transition form
    // was yo-yo farmable (down-up-down re-earned); the max form is potential-
    // based (phi = deepest level reached) and never taxes retreating up.
    if (depth > env->episode_max_depth) {
        r += env->descent_coef * (float)(depth - env->episode_max_depth);
        env->episode_max_depth = depth;
    }

    long dnum = env->blstats[NLE_BL_DNUM];
    if (dnum == 2) {
        env->episode_areas |= NETHACK_AREA_MINES;
        long mlvl = env->blstats[NLE_BL_DLEVEL];
        if (mlvl >= 3) env->episode_areas |= NETHACK_AREA_MINETOWN;
        if (mlvl >= 5) env->episode_areas |= NETHACK_AREA_DEEP_MINES;
    }
    else if (dnum == 0 && depth >= 5) env->episode_areas |= NETHACK_AREA_MAIN_D5;
    else if (dnum == 4) env->episode_areas |= NETHACK_AREA_SOKOBAN;

    // HP potential: heals give back what damage took away; net signal is the
    // dense precursor of combat death. Cash-out on the death step claws back
    // the remaining potential.
    long hp = env->blstats[NLE_BL_HP];
    r += env->hp_coef * (float)(hp - env->prev_hp);
    if (hp < env->prev_hp) env->episode_damage += env->prev_hp - hp;
    env->prev_hp = hp;
    env->prev_time = env->blstats[NLE_BL_TIME];

    // XP: paid only on new episode-max experience level, like descent —
    // rewards the power progression that must precede safe descent (agents
    // die underleveled at XL~4 vs depth 5-7 monsters). Max form is immune
    // to level-drain re-earn cycles.
    int xp = (int)env->blstats[NLE_BL_XP];
    if (xp > env->episode_max_xp) {
        r += env->xp_coef * (float)(xp - env->episode_max_xp);
        env->episode_max_xp = xp;
    }

    // Scout: first visit to a tile of this dungeon level this episode.
    // Per-level bitmaps persist across level changes, so stair yo-yo
    // cannot re-earn the bonus.
    if (px >= 0 && px < NH_COLS && py >= 0 && py < NH_ROWS) {
        int d = depth < 1 ? 0 : (depth > NETHACK_MAX_DEPTH ? NETHACK_MAX_DEPTH - 1 : depth - 1);
        int bit = (int)py * NH_COLS + (int)px;
        unsigned char mask = (unsigned char)(1 << (bit & 7));
        if (!(env->visited[d][bit >> 3] & mask)) {
            env->visited[d][bit >> 3] |= mask;
            r += env->scout_coef;
            env->episode_new_tiles++;
        }
    }

    if (illegal) r += env->illegal_penalty;
    env->prev_score = env->blstats[NLE_BL_SCORE];   // logs only
    env->prev_depth = depth;
    return r;
}

void c_step(Nethack* env) {
    if (env->pending_reset) {
        env->pending_reset = 0;
        nethack_do_reset(env);
    }

    int a = (int)env->actions[0];
    if (a < 0 || a >= NETHACK_NUM_ACTIONS) a = 0;
    env->obs.action = NETHACK_ACTION_TABLE[a];

    if (a == 19) env->episode_searches++;
    if (a == 20) {
        env->episode_prayers++;
        long hp = env->blstats[NLE_BL_HP], hpmax = env->blstats[NLE_BL_HPMAX];
        if (4 * hp <= hpmax) env->episode_prayers_low_hp++;
    }
    if (a == 21) env->episode_engraves++;

    long time_before = env->blstats[NLE_BL_TIME];
    int stepped = 1;
    if (a == 21) nethack_do_elbereth(env);
    else if (a == 22) env->episode_wears += nethack_do_use_item(env, 'W', "want to wear", NULL);
    else if (a == 23) {
        // Satiated gate: eating past Satiated is NetHack's choke() death —
        // without this, EAT-happy policies choke in a handful of presses.
        if (env->blstats[NLE_BL_HUNGER] == 0) stepped = 0;
        else env->episode_eats += nethack_do_use_item(env, 'e', "want to eat", "eat it");
    }
    else env->ctx = nle_step(env->ctx, &env->obs);
    env->prev_action = a;
    int illegal = stepped ? nethack_handle_prompts(env) : 0;
    if (env->blstats[NLE_BL_TIME] > time_before) env->episode_valid_moves++;
    env->episode_length++;

    float reward = env->obs.done
        ? env->death_penalty - env->hp_coef * (float)env->prev_hp
        : nethack_shaped_reward(env, illegal);
    env->rewards[0] = reward;
    env->episode_return += reward;
    if (reward > 1.0f || reward < -1.0f) env->episode_saturated++;

    int done = env->obs.done || env->episode_length >= NETHACK_MAX_EPISODE_STEPS;
    env->terminals[0] = done ? 1.0f : 0.0f;   // truncation reported as terminal too
    if (done) {
        nethack_add_log(env, env->obs.done ? env->obs.how_done : -1);
        c_reset(env);
    }
    nethack_pack_obs(env);
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
