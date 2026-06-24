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
#include "task_sphere.h"

static void hover_config(DroneEnv* env, Dict* kwargs) {
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = dict_get(kwargs, "hover_target_dist")->value;
    cfg->hover_dist = dict_get(kwargs, "hover_dist")->value;
    cfg->hover_omega = dict_get(kwargs, "hover_omega")->value;
    cfg->hover_vel = dict_get(kwargs, "hover_vel")->value;
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    cfg->alpha_omega = dict_get(kwargs, "hover_alpha_omega")->value;
    env->task_config = cfg;
}

static void race_config(DroneEnv* env, Dict* kwargs) {
    RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    cfg->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    cfg->ring_reward = dict_get(kwargs, "ring_reward")->value;
    cfg->collision_penalty = dict_get(kwargs, "collision_penalty")->value;
    cfg->time_penalty = dict_get(kwargs, "time_penalty")->value;
    cfg->oob_penalty = dict_get(kwargs, "oob_penalty")->value;
    cfg->alpha_omega = dict_get(kwargs, "race_alpha_omega")->value;
    env->task_config = cfg;
}

static void sphere_config(DroneEnv* env, Dict* kwargs) {
    SphereConfig* cfg = (SphereConfig*)calloc(1, sizeof(SphereConfig));
    cfg->radius = dict_get(kwargs, "sphere_radius")->value;
    cfg->hover_dist = dict_get(kwargs, "hover_dist")->value;
    cfg->hover_omega = dict_get(kwargs, "hover_omega")->value;
    cfg->hover_vel = dict_get(kwargs, "hover_vel")->value;
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    cfg->alpha_omega = dict_get(kwargs, "hover_alpha_omega")->value;
    env->task_config = cfg;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    env->alpha_dist = dict_get(kwargs, "alpha_dist")->value;

    float hover_w = dict_get(kwargs, "hover_frac")->value;
    float race_w = dict_get(kwargs, "race_frac")->value;
    float sphere_w = dict_get(kwargs, "sphere_frac")->value;
    float total = hover_w + race_w + sphere_w;

    float c_hover = hover_w / total;
    float c_race = (hover_w + race_w) / total;

    int idx = (int)env->rng;
    bool is_hover = (int)floorf((idx + 1) * c_hover) > (int)floorf(idx * c_hover);
    bool is_race = !is_hover && (int)floorf((idx + 1) * c_race) > (int)floorf(idx * c_race);

    if (is_hover) {
        env->task = &TASK_HOVER;
        hover_config(env, kwargs);
    } else if (is_race) {
        env->task = &TASK_RACE;
        race_config(env, kwargs);
    } else {
        env->task = &TASK_SPHERE;
        sphere_config(env, kwargs);
    }

    env->task->init(env);
    init(env);
}

static inline float task_avg(float sum, float n) { return n > 0.0f ? sum / n : 0.0f; }

void my_log(Log* log, Dict* out) {
    float hn = log->hover_n, rn = log->race_n, sn = log->sphere_n;

    int active = (hn > 0.0f) + (rn > 0.0f) + (sn > 0.0f);
    dict_set(out, "perf",
        (task_avg(log->hover_perf, hn) + task_avg(log->race_perf, rn) + task_avg(log->sphere_perf, sn)) / active);
    dict_set(out, "score",
        (task_avg(log->hover_score, hn) + task_avg(log->race_score, rn) + task_avg(log->sphere_score, sn)) / active);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    dict_set(out, "hover/perf", task_avg(log->hover_perf, hn));
    dict_set(out, "hover/ema_dist", task_avg(log->hover_keys[0], hn));
    dict_set(out, "race/perf", task_avg(log->race_perf, rn));
    dict_set(out, "hover/ema_vel", task_avg(log->hover_keys[1], hn));
    dict_set(out, "hover/score", task_avg(log->hover_score, hn));
    dict_set(out, "hover/ema_omega", task_avg(log->hover_keys[2], hn));
    dict_set(out, "race/score", task_avg(log->race_score, rn));
    dict_set(out, "race/rings_passed", task_avg(log->race_keys[0], rn));
    dict_set(out, "hover/oob", task_avg(log->hover_keys[3], hn));
    dict_set(out, "race/ring_collisions", task_avg(log->race_keys[1], rn));
    dict_set(out, "race/oob", task_avg(log->race_keys[3], rn));
    dict_set(out, "race/completed", task_avg(log->race_keys[2], rn));
    dict_set(out, "sphere/perf", task_avg(log->sphere_perf, sn));
    dict_set(out, "sphere/ema_dist", task_avg(log->sphere_keys[0], sn));
    dict_set(out, "sphere/score", task_avg(log->sphere_score, sn));
    dict_set(out, "sphere/ema_vel", task_avg(log->sphere_keys[1], sn));
    dict_set(out, "sphere/oob", task_avg(log->sphere_keys[3], sn));
    dict_set(out, "sphere/ema_omega", task_avg(log->sphere_keys[2], sn));
    dict_set(out, "hover/episode_frac", hn);
    dict_set(out, "race/episode_frac", rn);
    dict_set(out, "sphere/episode_frac", sn);
}