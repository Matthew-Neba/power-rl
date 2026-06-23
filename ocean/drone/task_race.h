#pragma once

#include "drone.h"

#define RACE_OOB_SCALE 2.0f

#define RACE_RING_MIN_DIST (2.0f * RING_RADIUS)
#define RACE_RING_MAX_DIST 8.0f

// types

typedef struct {
    int max_rings;
    float ring_reward;
    float collision_penalty;
    float time_penalty;
    float oob_penalty;
    float alpha_omega;
} RaceConfig;

typedef struct {
    Target* ring_buffer;
    int* ring_idx;
    int* rings_passed;
    float* collisions;
} RaceState;

// lifecycle

static void race_init(DroneEnv* env) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)calloc(1, sizeof(RaceState));
    state->ring_buffer = (Target*)calloc(cfg->max_rings, sizeof(Target));
    state->ring_idx = (int*)calloc(env->num_agents, sizeof(int));
    state->rings_passed = (int*)calloc(env->num_agents, sizeof(int));
    state->collisions = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
}

static void race_close(DroneEnv* env) {
    RaceState* state = (RaceState*)env->task_state;
    if (state != NULL) {
        free(state->ring_buffer);
        free(state->ring_idx);
        free(state->rings_passed);
        free(state->collisions);
        free(state);
    }
    free(env->task_config);
}

// helpers

static inline Target gen_next_ring(unsigned int* rng, const Target* current) {
    Target ring;
    float dist;
    do {
        ring = rndring(rng, RING_RADIUS);
        dist = norm3(sub3(ring.pos, current->pos));
    } while (dist < RACE_RING_MIN_DIST || dist > RACE_RING_MAX_DIST);
    return ring;
}

static inline int check_ring(Drone* drone, Target* ring) {
    float prev_dot = dot3(sub3(drone->prev_pos, ring->pos), ring->normal);
    float new_dot = dot3(sub3(drone->state.pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    if (valid_dir || invalid_dir) {
        Vec3 dir = sub3(drone->state.pos, drone->prev_pos);
        float denom = dot3(ring->normal, dir);
        if (fabsf(denom) < 1e-9f) return 0;

        float t = -prev_dot / denom;
        Vec3 intersection = add3(drone->prev_pos, scalmul3(dir, t));
        float d = norm3(sub3(intersection, ring->pos));

        // margins scale with radius
        float margin = 0.1f * ring->radius;
        if (d < (ring->radius - margin) && valid_dir) return 1;
        if (d < ring->radius + margin) return -1;
    }
    return 0;
}

// callbacks

static void race_env_reset(DroneEnv* env) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    state->ring_buffer[0] = rndring(&env->rng, RING_RADIUS);
    for (int i = 1; i < cfg->max_rings; i++)
        state->ring_buffer[i] = gen_next_ring(&env->rng, &state->ring_buffer[i - 1]);
}

static void race_reset(DroneEnv* env, Drone* agent, int idx) {
    RaceState* state = (RaceState*)env->task_state;

    do {
        agent->state.pos = random_pos(&env->rng);
    } while (norm3(sub3(agent->state.pos, state->ring_buffer[0].pos)) < 2.0f * RING_RADIUS);

    state->ring_idx[idx] = 0;
    state->rings_passed[idx] = 0;
    state->collisions[idx] = 0.0f;
    *agent->target = state->ring_buffer[0];
}

static float race_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;

    float reward = -cfg->alpha_omega * cache->omega;

    int result = check_ring(agent, &state->ring_buffer[state->ring_idx[idx]]);
    if (result == 1) {
        state->rings_passed[idx]++;
        state->ring_idx[idx]++;
        if (state->ring_idx[idx] < cfg->max_rings)
            *agent->target = state->ring_buffer[state->ring_idx[idx]];
        reward += cfg->ring_reward;
    } else if (result == -1) {
        state->collisions[idx] += 1.0f;
        reward -= cfg->collision_penalty;
    }

    if (out_of_bounds(agent->state.pos, RACE_OOB_SCALE)) reward -= cfg->oob_penalty;

    reward -= cfg->time_penalty;
    return reward;
}

static bool race_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    return state->rings_passed[idx] >= cfg->max_rings || out_of_bounds(agent->state.pos, RACE_OOB_SCALE) ||
           agent->episode_length >= HORIZON;
}

static void race_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    float completed = state->rings_passed[idx] >= cfg->max_rings ? 1.0f : 0.0f;
    log->race_n += 1.0f;
    log->race_perf += (float)state->rings_passed[idx] / (float)cfg->max_rings;
    log->race_score += (float)state->rings_passed[idx];
    log->race_keys[0] += (float)state->rings_passed[idx];
    log->race_keys[1] += state->collisions[idx];
    log->race_keys[2] += completed;
    log->race_keys[3] += out_of_bounds(agent->state.pos, RACE_OOB_SCALE) ? 1.0f : 0.0f;
}

static void race_render(DroneEnv* env, Client* client) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    for (int i = 0; i < cfg->max_rings; i++)
        DrawRing3D(state->ring_buffer[i], 0.2f, GREEN, BLUE);
}

// definition

static const Task TASK_RACE = {
    .name = "race",
    .id = 1,
    .init = race_init,
    .close = race_close,
    .env_reset = race_env_reset,
    .reset = race_reset,
    .reward = race_reward,
    .done = race_done,
    .log = race_log,
    .render = race_render,
};
