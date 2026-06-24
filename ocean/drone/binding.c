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
    cfg->collision_penalty = dict_get(kwargs, "collision_penalty")->value;
    cfg->time_penalty = dict_get(kwargs, "time_penalty")->value;
    cfg->oob_penalty = dict_get(kwargs, "oob_penalty")->value;
    cfg->alpha_omega = dict_get(kwargs, "race_alpha_omega")->value;
    env->task_config = cfg;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    env->alpha_dist = dict_get(kwargs, "alpha_dist")->value;

    float hover_w = dict_get(kwargs, "hover_frac")->value;
    float race_w = dict_get(kwargs, "race_frac")->value;
    float sphere_w = dict_get(kwargs, "sphere_frac")->value;
    float cube_w = dict_get(kwargs, "cube_frac")->value;
    float flag_w = dict_get(kwargs, "flag_frac")->value;
    float total = hover_w + race_w + sphere_w + cube_w + flag_w;

    float c_hover = hover_w / total;
    float c_race = (hover_w + race_w) / total;
    float c_sphere = (hover_w + race_w + sphere_w) / total;
    float c_cube = (hover_w + race_w + sphere_w + cube_w) / total;

    int idx = (int)env->rng;
    bool is_hover = (int)floorf((idx + 1) * c_hover) > (int)floorf(idx * c_hover);
    bool is_race = !is_hover && (int)floorf((idx + 1) * c_race) > (int)floorf(idx * c_race);
    bool is_sphere =
        !is_hover && !is_race && (int)floorf((idx + 1) * c_sphere) > (int)floorf(idx * c_sphere);
    bool is_cube = !is_hover && !is_race && !is_sphere &&
                   (int)floorf((idx + 1) * c_cube) > (int)floorf(idx * c_cube);

    if (is_hover) {
        env->task = TASK_HOVER;
        hover_config(env, kwargs);
    } else if (is_race) {
        env->task = TASK_RACE;
        race_config(env, kwargs);
    } else if (is_sphere) {
        env->task = TASK_SPHERE;
        hover_config(env, kwargs);
    } else if (is_cube) {
        env->task = TASK_CUBE;
        hover_config(env, kwargs);
    } else {
        env->task = TASK_FLAG;
        hover_config(env, kwargs);
    }

    task_init(env);
    init(env);
}

static inline float task_avg(float sum, float n) { return n > 0.0f ? sum / n : 0.0f; }

void my_log(Log* log, Dict* out) {
    TaskLog* h = &log->task[TASK_HOVER];
    TaskLog* r = &log->task[TASK_RACE];
    TaskLog* s = &log->task[TASK_SPHERE];
    TaskLog* c = &log->task[TASK_CUBE];
    TaskLog* f = &log->task[TASK_FLAG];
    float hn = h->n, rn = r->n, sn = s->n, cn = c->n, fn = f->n;

    int active = (hn > 0.0f) + (rn > 0.0f) + (sn > 0.0f) + (cn > 0.0f) + (fn > 0.0f);
    dict_set(out, "perf",
        (task_avg(h->perf, hn) + task_avg(r->perf, rn) + task_avg(s->perf, sn) + task_avg(c->perf, cn) + task_avg(f->perf, fn)) / active);
    dict_set(out, "score",
        (task_avg(h->score, hn) + task_avg(r->score, rn) + task_avg(s->score, sn) + task_avg(c->score, cn) + task_avg(f->score, fn)) / active);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    dict_set(out, "hover/perf", task_avg(h->perf, hn));
    dict_set(out, "hover/ema_dist", task_avg(h->keys[0], hn));
    dict_set(out, "race/perf", task_avg(r->perf, rn));
    dict_set(out, "hover/ema_vel", task_avg(h->keys[1], hn));
    dict_set(out, "hover/score", task_avg(h->score, hn));
    dict_set(out, "hover/ema_omega", task_avg(h->keys[2], hn));
    dict_set(out, "race/score", task_avg(r->score, rn));
    dict_set(out, "race/rings_passed", task_avg(r->keys[0], rn));
    dict_set(out, "hover/oob", task_avg(h->keys[3], hn));
    dict_set(out, "race/ring_collisions", task_avg(r->keys[1], rn));
    dict_set(out, "race/oob", task_avg(r->keys[3], rn));
    dict_set(out, "race/completed", task_avg(r->keys[2], rn));
    dict_set(out, "sphere/perf", task_avg(s->perf, sn));
    dict_set(out, "sphere/ema_dist", task_avg(s->keys[0], sn));
    dict_set(out, "sphere/score", task_avg(s->score, sn));
    dict_set(out, "sphere/ema_vel", task_avg(s->keys[1], sn));
    dict_set(out, "sphere/oob", task_avg(s->keys[3], sn));
    dict_set(out, "sphere/ema_omega", task_avg(s->keys[2], sn));
    dict_set(out, "cube/perf", task_avg(c->perf, cn));
    dict_set(out, "cube/ema_dist", task_avg(c->keys[0], cn));
    dict_set(out, "cube/score", task_avg(c->score, cn));
    dict_set(out, "cube/ema_vel", task_avg(c->keys[1], cn));
    dict_set(out, "cube/oob", task_avg(c->keys[3], cn));
    dict_set(out, "cube/ema_omega", task_avg(c->keys[2], cn));
    dict_set(out, "flag/perf", task_avg(f->perf, fn));
    dict_set(out, "flag/ema_dist", task_avg(f->keys[0], fn));
    dict_set(out, "flag/score", task_avg(f->score, fn));
    dict_set(out, "flag/ema_vel", task_avg(f->keys[1], fn));
    dict_set(out, "flag/oob", task_avg(f->keys[3], fn));
    dict_set(out, "flag/ema_omega", task_avg(f->keys[2], fn));
    dict_set(out, "hover/episode_frac", hn);
    dict_set(out, "race/episode_frac", rn);
    dict_set(out, "sphere/episode_frac", sn);
    dict_set(out, "cube/episode_frac", cn);
    dict_set(out, "flag/episode_frac", fn);
}