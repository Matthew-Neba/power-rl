#include "drone.h"
#include "render.h"

#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"


static float task_fracs[NUM_TASKS];

static void hover_config(DroneEnv* env, Dict* kwargs) {
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = dict_get(kwargs, "hover_target_dist")->value;
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_dist = dict_get(kwargs, "hover_alpha_dist")->value;
    cfg->sphere_radius = dict_get(kwargs, "sphere_radius")->value;
    cfg->horizon = (int)dict_get(kwargs, "hover_horizon")->value;
    env->task_config = cfg;
}

static void race_config(DroneEnv* env, Dict* kwargs) {
    RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    cfg->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    cfg->ring_reward = dict_get(kwargs, "ring_reward")->value;
    cfg->alpha_dist = dict_get(kwargs, "race_alpha_dist")->value;
    cfg->horizon = (int)dict_get(kwargs, "race_horizon")->value;
    env->task_config = cfg;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    env->alpha_vel = dict_get(kwargs, "alpha_vel")->value;
    env->alpha_omega = dict_get(kwargs, "alpha_omega")->value;
    env->alpha_action = dict_get(kwargs, "alpha_action")->value;
    env->dr = dict_get(kwargs, "dr")->value;

    task_fracs[TASK_HOVER] = dict_get(kwargs, "hover_frac")->value;
    task_fracs[TASK_RACE] = dict_get(kwargs, "race_frac")->value;
    task_fracs[TASK_SPHERE] = dict_get(kwargs, "sphere_frac")->value;
    task_fracs[TASK_CUBE] = dict_get(kwargs, "cube_frac")->value;
    task_fracs[TASK_FLAG] = dict_get(kwargs, "flag_frac")->value;

    float total = 0.0f;
    for (int t = 0; t < NUM_TASKS; t++) {
        total += task_fracs[t];
    }

    int idx = (int)env->rng;
    float cum = 0.0f;
    env->task = TASK_HOVER;
    for (int t = 0; t < NUM_TASKS; t++) {
        cum += task_fracs[t] / total;
        if ((int)floorf((idx + 1) * cum) > (int)floorf(idx * cum)) {
            env->task = (TaskType)t;
            break;
        }
    }

    if (env->task == TASK_RACE) {
        race_config(env, kwargs);
    } else {
        hover_config(env, kwargs);
    }

    task_init(env);
    init(env);
}

static inline float task_avg(float sum, float n) { return n > 0.0f ? sum / n : 0.0f; }

void my_log(Log* log, Dict* out) {
    static int first = 1;

    float perf = 0.0f, score = 0.0f;
    int active = 0;
    for (int t = 0; t < NUM_TASKS; t++) {
        float n = log->task[t].n;
        if (n <= 0.0f) continue;
        perf += log->task[t].perf / n;
        score += log->task[t].score / n;
        active++;
    }
    dict_set(out, "perf", active > 0 ? perf / active : 0.0f);
    dict_set(out, "score", active > 0 ? score / active : 0.0f);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    if (log->task[TASK_HOVER].n > 0.0f || (first && task_fracs[TASK_HOVER] > 0.0f)) {
        TaskLog* h = &log->task[TASK_HOVER];
        dict_set(out, "hover/perf", task_avg(h->perf, h->n));
        dict_set(out, "hover/score", task_avg(h->score, h->n));
        dict_set(out, "hover/ema_dist", task_avg(h->keys[0], h->n));
        dict_set(out, "hover/ema_vel", task_avg(h->keys[1], h->n));
        dict_set(out, "hover/ema_omega", task_avg(h->keys[2], h->n));
        dict_set(out, "hover/oob", task_avg(h->keys[3], h->n));
        dict_set(out, "hover/episode_frac", h->n);
    }
    if (log->task[TASK_RACE].n > 0.0f || (first && task_fracs[TASK_RACE] > 0.0f)) {
        TaskLog* r = &log->task[TASK_RACE];
        dict_set(out, "race/perf", task_avg(r->perf, r->n));
        dict_set(out, "race/score", task_avg(r->score, r->n));
        dict_set(out, "race/rings_passed", task_avg(r->keys[0], r->n));
        dict_set(out, "race/ring_collisions", task_avg(r->keys[1], r->n));
        dict_set(out, "race/completed", task_avg(r->keys[2], r->n));
        dict_set(out, "race/oob", task_avg(r->keys[3], r->n));
        dict_set(out, "race/episode_frac", r->n);
    }
    if (log->task[TASK_SPHERE].n > 0.0f || (first && task_fracs[TASK_SPHERE] > 0.0f)) {
        TaskLog* s = &log->task[TASK_SPHERE];
        dict_set(out, "sphere/perf", task_avg(s->perf, s->n));
        dict_set(out, "sphere/score", task_avg(s->score, s->n));
        dict_set(out, "sphere/ema_dist", task_avg(s->keys[0], s->n));
        dict_set(out, "sphere/ema_vel", task_avg(s->keys[1], s->n));
        dict_set(out, "sphere/ema_omega", task_avg(s->keys[2], s->n));
        dict_set(out, "sphere/oob", task_avg(s->keys[3], s->n));
        dict_set(out, "sphere/episode_frac", s->n);
    }
    if (log->task[TASK_CUBE].n > 0.0f || (first && task_fracs[TASK_CUBE] > 0.0f)) {
        TaskLog* c = &log->task[TASK_CUBE];
        dict_set(out, "cube/perf", task_avg(c->perf, c->n));
        dict_set(out, "cube/score", task_avg(c->score, c->n));
        dict_set(out, "cube/ema_dist", task_avg(c->keys[0], c->n));
        dict_set(out, "cube/ema_vel", task_avg(c->keys[1], c->n));
        dict_set(out, "cube/ema_omega", task_avg(c->keys[2], c->n));
        dict_set(out, "cube/oob", task_avg(c->keys[3], c->n));
        dict_set(out, "cube/episode_frac", c->n);
    }
    if (log->task[TASK_FLAG].n > 0.0f || (first && task_fracs[TASK_FLAG] > 0.0f)) {
        TaskLog* f = &log->task[TASK_FLAG];
        dict_set(out, "flag/perf", task_avg(f->perf, f->n));
        dict_set(out, "flag/score", task_avg(f->score, f->n));
        dict_set(out, "flag/ema_dist", task_avg(f->keys[0], f->n));
        dict_set(out, "flag/ema_vel", task_avg(f->keys[1], f->n));
        dict_set(out, "flag/ema_omega", task_avg(f->keys[2], f->n));
        dict_set(out, "flag/oob", task_avg(f->keys[3], f->n));
        dict_set(out, "flag/episode_frac", f->n);
    }

    first = 0;
}