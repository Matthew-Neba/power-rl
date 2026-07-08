#include "nethack.h"
#define OBS_SIZE NETHACK_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {NETHACK_NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#define Env Nethack
#include "vecenv.h"

// Read a float kwarg if present; otherwise keep the value already set in
// init() (the per-term legacy default). This means config/nethack.ini can
// safely omit a key — the env still works.
static void nethack_read_coef(Dict* kwargs, const char* key, float* dst) {
    DictItem* it = dict_get_unsafe(kwargs, key);
    if (it != NULL) *dst = (float)it->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    init(env);
    // Reward-shaping coefficients (see ocean/nethack/REWARDS.md). All four
    // are optional; defaults are baked in init(). To add a new coef:
    //   1) Add `float my_coef;` to Nethack struct in nethack.h
    //   2) Set its default in init()
    //   3) Add a nethack_read_coef line here
    //   4) Add `my_coef = <default>` to [env] in config/nethack.ini
    nethack_read_coef(kwargs, "score_coef",      &env->score_coef);
    nethack_read_coef(kwargs, "descent_coef",    &env->descent_coef);
    nethack_read_coef(kwargs, "scout_coef",      &env->scout_coef);
    nethack_read_coef(kwargs, "xp_coef",         &env->xp_coef);
    nethack_read_coef(kwargs, "hp_coef",         &env->hp_coef);
    nethack_read_coef(kwargs, "illegal_penalty", &env->illegal_penalty);
    nethack_read_coef(kwargs, "death_penalty",   &env->death_penalty);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "depth", log->depth);
    dict_set(out, "valid_moves", log->valid_moves);
    dict_set(out, "illegal_actions", log->illegal_actions);
    dict_set(out, "new_tiles", log->new_tiles);
    dict_set(out, "max_depth", log->max_depth);
    dict_set(out, "prayers", log->prayers);
    dict_set(out, "prayers_low_hp", log->prayers_low_hp);
    dict_set(out, "searches", log->searches);
    dict_set(out, "engraves", log->engraves);
    dict_set(out, "wears", log->wears);
    dict_set(out, "eats", log->eats);
    dict_set(out, "damage_taken", log->damage_taken);
    dict_set(out, "game_time", log->game_time);
    dict_set(out, "max_xp_level", log->max_xp_level);
    dict_set(out, "death_combat", log->death_combat);
    dict_set(out, "death_starved", log->death_starved);
    dict_set(out, "death_other", log->death_other);
    dict_set(out, "truncated", log->truncated);
    dict_set(out, "reach_mines", log->reach_mines);
    dict_set(out, "reach_minetown", log->reach_minetown);
    dict_set(out, "reach_deep_mines", log->reach_deep_mines);
    dict_set(out, "reach_main_d5", log->reach_main_d5);
    dict_set(out, "reach_sokoban", log->reach_sokoban);
}
