#pragma once

#include "drone.h"

// Each drone is assigned a fixed slot on a sphere surface (Fibonacci layout, by
// index) and must fly to it and hold; the sphere shape emerges from all drones
// reaching their slots. Per-drone this is "fly to your target and hold", so the
// reward/state mirror hover for now — the overlap is intentional and will be
// factored into a shared tasklib once the task is settled.

// types

#define SPHERE_SCORE_DIST_SCALE 0.01f
#define SPHERE_SCORE_VEL_SCALE 0.01f
#define SPHERE_SCORE_OMEGA_SCALE 0.1f

typedef struct {
    float radius;
    float hover_dist;
    float hover_omega;
    float hover_vel;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;
} SphereConfig;

typedef struct {
    float* prev_potential;
    float* score;
    float* perf;
    float* ema_dist;
    float* ema_vel;
    float* ema_omega;
} SphereState;

// lifecycle

static void sphere_init(DroneEnv* env) {
    SphereState* state = (SphereState*)calloc(1, sizeof(SphereState));
    state->prev_potential = (float*)calloc(env->num_agents, sizeof(float));
    state->score = (float*)calloc(env->num_agents, sizeof(float));
    state->perf = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_dist = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_vel = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_omega = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
}

static void sphere_close(DroneEnv* env) {
    SphereState* state = (SphereState*)env->task_state;
    if (state != NULL) {
        free(state->prev_potential);
        free(state->score);
        free(state->perf);
        free(state->ema_dist);
        free(state->ema_vel);
        free(state->ema_omega);
        free(state);
    }
    free(env->task_config);
}

// helpers

// Fibonacci-sphere slot for drone idx (z up), matching the original orbit task.
static inline Vec3 sphere_slot(int idx, int num_agents, float radius) {
    float phi = (float)M_PI * (sqrtf(5.0f) - 1.0f);
    float y = 1.0f - 2.0f * ((float)idx / (float)num_agents);
    float r = sqrtf(fmaxf(0.0f, 1.0f - y * y));
    float theta = phi * (float)idx;
    return (Vec3){radius * cosf(theta) * r, radius * sinf(theta) * r, radius * y};
}

static inline float sphere_potential(float dist, float vel, float omega, SphereConfig* cfg) {
    float d = 1.0f / (1.0f + dist / cfg->hover_dist);
    float v = 1.0f / (1.0f + vel / cfg->hover_vel);
    float w = 1.0f / (1.0f + omega / cfg->hover_omega);
    return d * (0.7f + 0.15f * v + 0.15f * w);
}

static inline float sphere_score(float dist, float vel, float omega) {
    float d = dist / SPHERE_SCORE_DIST_SCALE;
    float v = vel / SPHERE_SCORE_VEL_SCALE;
    float w = omega / SPHERE_SCORE_OMEGA_SCALE;
    float penalty = 0.7f * d + 0.15f * v + 0.15f * w;
    return 1.0f / (1.0f + 0.05f * penalty);
}

// callbacks

static void sphere_reset(DroneEnv* env, Drone* agent, int idx) {
    SphereConfig* cfg = (SphereConfig*)env->task_config;
    SphereState* state = (SphereState*)env->task_state;

    agent->state.pos = random_pos(&env->rng);
    agent->target->pos = sphere_slot(idx, env->num_agents, cfg->radius);
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->target->normal = (Vec3){0.0f, 0.0f, 0.0f};

    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    state->score[idx] = 0.0f;
    state->perf[idx] = sphere_score(dist, vel, omega);
    state->ema_dist[idx] = dist;
    state->ema_vel[idx] = vel;
    state->ema_omega[idx] = omega;
    state->prev_potential[idx] = sphere_potential(dist, vel, omega, cfg);
}

static float sphere_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    SphereConfig* cfg = (SphereConfig*)env->task_config;
    SphereState* state = (SphereState*)env->task_state;

    float curr = sphere_potential(cache->dist, cache->vel, cache->omega, cfg);
    float reward = cfg->alpha_hover * curr
                 + cfg->alpha_shaping * (curr - state->prev_potential[idx])
                 - cfg->alpha_omega * cache->omega;
    state->prev_potential[idx] = curr;

    float score = sphere_score(cache->dist, cache->vel, cache->omega);
    state->score[idx] += score;
    state->perf[idx] = 0.98f * state->perf[idx] + 0.02f * score;
    state->ema_dist[idx] = 0.99f * state->ema_dist[idx] + 0.01f * cache->dist;
    state->ema_vel[idx] = 0.99f * state->ema_vel[idx] + 0.01f * cache->vel;
    state->ema_omega[idx] = 0.99f * state->ema_omega[idx] + 0.01f * cache->omega;
    return reward;
}

static bool sphere_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    return out_of_bounds(agent->state.pos, 1.0f) || agent->episode_length >= HORIZON;
}

static void sphere_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    SphereState* state = (SphereState*)env->task_state;
    log->sphere_n += 1.0f;
    log->sphere_perf += state->perf[idx];
    log->sphere_score += state->score[idx];
    log->sphere_keys[0] += state->ema_dist[idx];
    log->sphere_keys[1] += state->ema_vel[idx];
    log->sphere_keys[2] += state->ema_omega[idx];
    log->sphere_keys[3] += out_of_bounds(agent->state.pos, 1.0f) ? 1.0f : 0.0f;
}

static void sphere_render(DroneEnv* env, Client* client) {
    SphereConfig* cfg = (SphereConfig*)env->task_config;
    for (int i = 0; i < env->num_agents; i++) {
        Vec3 p = sphere_slot(i, env->num_agents, cfg->radius);
        DrawSphere((Vector3){p.x, p.y, p.z}, 0.08f, (Color){0, 255, 255, 120});
    }
}

// definition

static const Task TASK_SPHERE = {
    .name = "sphere",
    .id = 2,
    .init = sphere_init,
    .close = sphere_close,
    .env_reset = NULL,
    .reset = sphere_reset,
    .reward = sphere_reward,
    .done = sphere_done,
    .log = sphere_log,
    .render = sphere_render,
};
