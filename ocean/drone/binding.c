#include "drone.h"
#include "render.h"

#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"

#include "task_hover.h"
#include "task_race.h"

static void hover_config(DroneEnv* env, Dict* kwargs) {
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = dict_get(kwargs, "hover_target_dist")->value;
    cfg->hover_dist = dict_get(kwargs, "hover_dist")->value;
    cfg->hover_omega = dict_get(kwargs, "hover_omega")->value;
    cfg->hover_vel = dict_get(kwargs, "hover_vel")->value;
    cfg->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    cfg->alpha_omega = dict_get(kwargs, "alpha_omega")->value;
    env->task_config = cfg;
}

static void race_config(DroneEnv* env, Dict* kwargs) {
    RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    cfg->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    cfg->ring_reward = dict_get(kwargs, "ring_reward")->value;
    cfg->collision_penalty = dict_get(kwargs, "collision_penalty")->value;
    cfg->time_penalty = dict_get(kwargs, "time_penalty")->value;
    cfg->oob_penalty = dict_get(kwargs, "oob_penalty")->value;
    cfg->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    env->task_config = cfg;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    // Assign this env's task by its index using per-task env fractions. env->rng is
    // set to the env index before my_init (vecenv.h), so this is deterministic and
    // spreads tasks evenly across envs. Single-task = set one frac to 1.0.
    float hover_frac = dict_get(kwargs, "hover_frac")->value;
    float race_frac = dict_get(kwargs, "race_frac")->value;
    float total = hover_frac + race_frac;
    if (total <= 0.0f) {
        race_frac = 0.0f;
        total = 1.0f;
    }
    race_frac /= total;

    int idx = (int)env->rng;
    bool is_race = (int)floorf((idx + 1) * race_frac) > (int)floorf(idx * race_frac);
    if (is_race) {
        env->task = &TASK_RACE;
        race_config(env, kwargs);
    } else {
        env->task = &TASK_HOVER;
        hover_config(env, kwargs);
    }

    env->task->init(env);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    // Overall per-episode averages across both tasks. hover_* and race_* fields are
    // each already divided by the total episode count by the aggregator, and the two
    // tasks' episodes are disjoint, so summing gives the all-episode average. Keeps
    // top-level score/perf available (the sweep objective keys on env/score).
    dict_set(out, "score", log->hover_score + log->race_score);
    dict_set(out, "perf", log->hover_perf + log->race_perf);

    // Per-task averages: each key is divided by its own (aggregated) episode count,
    // which cancels the global-n division done by the vec aggregator. *_n is the
    // fraction of episodes that were this task.
    if (log->hover_n > 0.0f) {
        dict_set(out, "hover/frac", log->hover_n);
        dict_set(out, "hover/perf", log->hover_perf / log->hover_n);
        dict_set(out, "hover/score", log->hover_score / log->hover_n);
        dict_set(out, "hover/ema_dist", log->hover_keys[0] / log->hover_n);
        dict_set(out, "hover/ema_vel", log->hover_keys[1] / log->hover_n);
        dict_set(out, "hover/ema_omega", log->hover_keys[2] / log->hover_n);
        dict_set(out, "hover/oob", log->hover_keys[3] / log->hover_n);
    }
    if (log->race_n > 0.0f) {
        dict_set(out, "race/frac", log->race_n);
        dict_set(out, "race/perf", log->race_perf / log->race_n);
        dict_set(out, "race/score", log->race_score / log->race_n);
        dict_set(out, "race/rings_passed", log->race_keys[0] / log->race_n);
        dict_set(out, "race/ring_collisions", log->race_keys[1] / log->race_n);
        dict_set(out, "race/completed", log->race_keys[2] / log->race_n);
        dict_set(out, "race/oob", log->race_keys[3] / log->race_n);
    }
}