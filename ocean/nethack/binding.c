#include "nethack.h"
#define OBS_SIZE NETHACK_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {NETHACK_NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#define Env Nethack
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    init(env);
    env->gold_coef = dict_get(kwargs, "gold_coef")->value;
    env->exp_coef = dict_get(kwargs, "exp_coef")->value;
    env->descent_coef = dict_get(kwargs, "descent_coef")->value;
    env->scout_coef = dict_get(kwargs, "scout_coef")->value;
    env->xp_coef = dict_get(kwargs, "xp_coef")->value;
    env->hp_coef = dict_get(kwargs, "hp_coef")->value;
    env->illegal_penalty = dict_get(kwargs, "illegal_penalty")->value;
    env->death_penalty = dict_get(kwargs, "death_penalty")->value;
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
    dict_set(out, "reward_saturated", log->reward_saturated);
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
