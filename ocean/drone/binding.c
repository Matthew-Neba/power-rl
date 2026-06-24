#include "drone.h"
#include "render.h"

#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"


static void hover_config(DroneEnv* env, Dict* kwargs) {
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = dict_get(kwargs, "hover_target_dist")->value;
    cfg->hover_dist = dict_get(kwargs, "hover_dist")->value;
    cfg->hover_omega = dict_get(kwargs, "hover_omega")->value;
    cfg->hover_vel = dict_get(kwargs, "hover_vel")->value;
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    cfg->alpha_omega = dict_get(kwargs, "hover_alpha_omega")->value;
    cfg->radius = dict_get(kwargs, "sphere_radius")->value;
    env->task_config = cfg;
}

static void race_config(DroneEnv* env, Dict* kwargs) {
    RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    cfg->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    cfg->ring_reward = dict_get(kwargs, "ring_reward")->value;
    cfg->alpha_omega = dict_get(kwargs, "race_alpha_omega")->value;
    env->task_config = cfg;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    env->alpha_dist = dict_get(kwargs, "alpha_dist")->value;

    float frac[NUM_TASKS];
    frac[TASK_HOVER] = dict_get(kwargs, "hover_frac")->value;
    frac[TASK_RACE] = dict_get(kwargs, "race_frac")->value;
    frac[TASK_SPHERE] = dict_get(kwargs, "sphere_frac")->value;
    frac[TASK_CUBE] = dict_get(kwargs, "cube_frac")->value;
    frac[TASK_FLAG] = dict_get(kwargs, "flag_frac")->value;

    float total = 0.0f;
    for (int t = 0; t < NUM_TASKS; t++) {
        total += frac[t];
    }

    int idx = (int)env->rng;
    float cum = 0.0f;
    env->task = TASK_HOVER;
    for (int t = 0; t < NUM_TASKS; t++) {
        cum += frac[t] / total;
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

void my_log(Log* log, Dict* out) {
    TaskLog* h = &log->task[TASK_HOVER];
    TaskLog* r = &log->task[TASK_RACE];
    TaskLog* s = &log->task[TASK_SPHERE];
    TaskLog* c = &log->task[TASK_CUBE];
    TaskLog* f = &log->task[TASK_FLAG];

    float perf = 0.0f, score = 0.0f;
    int active = 0;
    for (int t = 0; t < NUM_TASKS; t++) {
        float n = log->task[t].n;
        if (n <= 0.0f) continue;
        perf += log->task[t].perf / n;
        score += log->task[t].score / n;
        active++;
    }
    dict_set(out, "perf", perf / active);
    dict_set(out, "score", score / active);

    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    // per-task metrics where frac > 0
    if (h->n > 0.0f) {
        dict_set(out, "hover/perf", h->perf / h->n);
        dict_set(out, "hover/score", h->score / h->n);
        dict_set(out, "hover/ema_dist", h->keys[0] / h->n);
        dict_set(out, "hover/ema_vel", h->keys[1] / h->n);
        dict_set(out, "hover/ema_omega", h->keys[2] / h->n);
        dict_set(out, "hover/oob", h->keys[3] / h->n);
        dict_set(out, "hover/episode_frac", h->n);
    }
    if (r->n > 0.0f) {
        dict_set(out, "race/perf", r->perf / r->n);
        dict_set(out, "race/score", r->score / r->n);
        dict_set(out, "race/rings_passed", r->keys[0] / r->n);
        dict_set(out, "race/ring_collisions", r->keys[1] / r->n);
        dict_set(out, "race/completed", r->keys[2] / r->n);
        dict_set(out, "race/oob", r->keys[3] / r->n);
        dict_set(out, "race/episode_frac", r->n);
    }
    if (s->n > 0.0f) {
        dict_set(out, "sphere/perf", s->perf / s->n);
        dict_set(out, "sphere/score", s->score / s->n);
        dict_set(out, "sphere/ema_dist", s->keys[0] / s->n);
        dict_set(out, "sphere/ema_vel", s->keys[1] / s->n);
        dict_set(out, "sphere/ema_omega", s->keys[2] / s->n);
        dict_set(out, "sphere/oob", s->keys[3] / s->n);
        dict_set(out, "sphere/episode_frac", s->n);
    }
    if (c->n > 0.0f) {
        dict_set(out, "cube/perf", c->perf / c->n);
        dict_set(out, "cube/score", c->score / c->n);
        dict_set(out, "cube/ema_dist", c->keys[0] / c->n);
        dict_set(out, "cube/ema_vel", c->keys[1] / c->n);
        dict_set(out, "cube/ema_omega", c->keys[2] / c->n);
        dict_set(out, "cube/oob", c->keys[3] / c->n);
        dict_set(out, "cube/episode_frac", c->n);
    }
    if (f->n > 0.0f) {
        dict_set(out, "flag/perf", f->perf / f->n);
        dict_set(out, "flag/score", f->score / f->n);
        dict_set(out, "flag/ema_dist", f->keys[0] / f->n);
        dict_set(out, "flag/ema_vel", f->keys[1] / f->n);
        dict_set(out, "flag/ema_omega", f->keys[2] / f->n);
        dict_set(out, "flag/oob", f->keys[3] / f->n);
        dict_set(out, "flag/episode_frac", f->n);
    }
}