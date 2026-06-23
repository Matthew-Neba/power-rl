// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "dronelib.h"

#define HORIZON 2048

typedef struct {
    float dist;
    float prev_dist;
    float vel;
    float omega;
} StepCache;

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float n;

    float hover_n;
    float hover_perf;
    float hover_score;
    float hover_keys[4]; // ema_dist, ema_vel, ema_omega, oob

    float race_n;
    float race_perf;
    float race_score;
    float race_keys[4]; // rings_passed, ring_collisions, completed, oob
};

typedef struct DroneEnv DroneEnv;
typedef struct Client Client;

typedef struct {
    const char* name;
    int id; // task index, also the one-hot slot in the observation

    void (*init)(DroneEnv* env);
    void (*close)(DroneEnv* env);
    void (*env_reset)(DroneEnv* env);
    void (*reset)(DroneEnv* env, Drone* agent, int idx);
    float (*reward)(DroneEnv* env, Drone* agent, int idx, StepCache* cache);
    bool (*done)(DroneEnv* env, Drone* agent, int idx, StepCache* cache);

    void (*log)(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache);
    void (*render)(DroneEnv* env, Client* client);
} Task;

struct DroneEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int tick;
    Drone* agents;
    Log log;

    const Task* task;
    void* task_config;
    void* task_state;

    // shared reward primitive
    float alpha_dist;

    Client* client;
};

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++)
        compute_drone_observations(&env->agents[i], env->task->id,
                                   env->observations + i * DRONE_OBS_SIZE);
}

void reset_agent_base(Drone* agent, unsigned int* rng) {
    Target* target = agent->target;
    memset(agent, 0, sizeof(Drone));
    agent->target = target;
    init_drone(agent, rng, 0.05f);
}

void init(DroneEnv* env) {
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    for (int i = 0; i < env->num_agents; i++)
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneEnv* env, int idx, StepCache* cache) {
    Drone* agent = &env->agents[idx];
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.n += 1.0f;

    if (env->task->log) env->task->log(env, agent, idx, &env->log, cache);
}

void c_reset(DroneEnv* env) {
    if (env->task->env_reset) env->task->env_reset(env);

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent_base(agent, &env->rng);
        env->task->reset(env, agent, i);
        agent->prev_pos = agent->state.pos;
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        agent->prev_pos = agent->state.pos;
        move_drone(agent, &env->actions[4 * i]);
        agent->episode_length++;

        StepCache cache = {
            .prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos)),
            .dist = norm3(sub3(agent->target->pos, agent->state.pos)),
            .vel = norm3(agent->state.vel),
            .omega = norm3(agent->state.omega),
        };

        float reward = env->task->reward(env, agent, i, &cache);
        reward += env->alpha_dist * (cache.prev_dist - cache.dist);

        bool done = env->task->done(env, agent, i, &cache);

        agent->episode_return += reward;
        env->rewards[i] = reward;
        env->terminals[i] = done ? 1.0f : 0.0f;

        if (done) {
            add_log(env, i, &cache);
            reset_agent_base(agent, &env->rng);
            env->task->reset(env, agent, i);
            agent->prev_pos = agent->state.pos;
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    if (env->task != NULL && env->task->close != NULL) env->task->close(env);

    for (int i = 0; i < env->num_agents; i++)
        free(env->agents[i].target);
    free(env->agents);

    if (env->client != NULL) c_close_client(env->client);
}