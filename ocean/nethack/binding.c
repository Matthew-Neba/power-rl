#include "nethack.h"
#define OBS_SIZE NETHACK_OBS_SIZE
#define NUM_ATNS 14
#define ACT_SIZES {NETHACK_NUM_ACTIONS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_NUM_DIRS}
#define OBS_TENSOR_T ByteTensor
#define MY_ACTION_MASK (NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS + NETHACK_NUM_DIRS)

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
    env->hunger_coef = dict_get(kwargs, "hunger_coef")->value;
    env->illegal_penalty = dict_get(kwargs, "illegal_penalty")->value;
    env->death_penalty = dict_get(kwargs, "death_penalty")->value;
    env->ac_coef = dict_get(kwargs, "ac_coef")->value;
    env->heal_coef = dict_get(kwargs, "heal_coef")->value;
    env->status_coef = dict_get(kwargs, "status_coef")->value;
}

void my_log(Log* log, Dict* out) {
    for (int v = 0; v < NETHACK_NUM_ACTIONS; v++)
        if (NETHACK_VERB_STAT[v]) dict_set(out, NETHACK_VERB_STAT[v], log->verb_uses[v]);
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "valid_moves", log->valid_moves);
    dict_set(out, "illegal_actions", log->illegal_actions);
    dict_set(out, "new_tiles", log->new_tiles);
    dict_set(out, "max_depth", log->max_depth);
    dict_set(out, "enhances", log->enhances);
    dict_set(out, "floor_eats", log->floor_eats);
    dict_set(out, "prayers_low_hp", log->prayers_low_hp);
    dict_set(out, "prayers_starving", log->prayers_starving);
    dict_set(out, "burdened_frac", log->burdened_frac);
    dict_set(out, "damage_taken", log->damage_taken);
    dict_set(out, "ac", log->ac);
    dict_set(out, "min_ac", log->min_ac);
    dict_set(out, "armor_swaps", log->armor_swaps);
    dict_set(out, "heal_hp", log->heal_hp);
    dict_set(out, "cures", log->cures);
    dict_set(out, "game_time", log->game_time);
    dict_set(out, "max_xp_level", log->max_xp_level);
    dict_set(out, "death_combat", log->death_combat);
    dict_set(out, "death_starved", log->death_starved);
    dict_set(out, "death_smited", log->death_smited);
    dict_set(out, "death_other", log->death_other);
    dict_set(out, "death_mon_level", log->death_mon_level);
    dict_set(out, "death_adj_monsters", log->death_adj_monsters);
    dict_set(out, "death_maxhp", log->death_maxhp);
    dict_set(out, "truncated", log->truncated);
    dict_set(out, "reach_mines", log->reach_mines);
    dict_set(out, "reach_minetown", log->reach_minetown);
    dict_set(out, "reach_deep_mines", log->reach_deep_mines);
    dict_set(out, "reach_main_d5", log->reach_main_d5);
    dict_set(out, "reach_sokoban", log->reach_sokoban);
}
