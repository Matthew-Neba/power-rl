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
#include "physics.h"

#define HORIZON 2048

typedef enum {
    TASK_HOVER = 0,
    TASK_RACE = 1,
    TASK_SPHERE = 2,
    TASK_CUBE = 3,
    TASK_FLAG = 4,
} TaskType;

#define NUM_TASKS 5

typedef struct {
    float dist;
    float prev_dist;
    float vel;
    float omega;
} StepCache;

typedef struct {
    float n;
    float perf;
    float score;
    float keys[4];
} TaskLog;

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float n;
    TaskLog task[NUM_TASKS];
};

typedef struct DroneEnv DroneEnv;
typedef struct Client Client;

struct DroneEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int tick;
    Drone* agents;
    Paramsv* params_soa;
    Log log;

    TaskType task;
    void* task_config;
    void* task_state;

    // shared reward shaping
    float alpha_vel;
    float alpha_omega;
    float alpha_action;

    // domain randomisation
    float dr;

    Client* client;
};

#include "tasklib.h"

void compute_observations(DroneEnv* env) {
    bool is_race = (env->task == TASK_RACE);
    for (int i = 0; i < env->num_agents; i++)
        compute_drone_observations(&env->agents[i], env->observations + i * DRONE_OBS_SIZE, is_race);
}

void reset_agent_base(DroneEnv* env, int idx) {
    Drone* agent = &env->agents[idx];
    Target* target = agent->target;
    memset(agent, 0, sizeof(Drone));
    agent->target = target;

    init_drone(agent, &env->rng, env->dr);
    store_params(env->params_soa, idx, &agent->params);
}

void init(DroneEnv* env) {
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    for (int i = 0; i < env->num_agents; i++)
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));

    env->params_soa = init_physics(env->num_agents);

    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneEnv* env, int idx, StepCache* cache) {
    Drone* agent = &env->agents[idx];
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.n += 1.0f;

    task_log(env, agent, idx, &env->log, cache);
}

void c_reset(DroneEnv* env) {
    task_env_reset(env);

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent_base(env, i);
        task_reset(env, agent, i);
        agent->prev_pos = agent->state.pos;
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++)
        env->agents[i].prev_pos = env->agents[i].state.pos;

    move_drones(env->agents, env->actions, env->params_soa, env->num_agents);

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        agent->episode_length++;

        StepCache cache = {
            .prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos)),
            .dist = norm3(sub3(agent->target->pos, agent->state.pos)),
            .vel = norm3(agent->state.vel),
            .omega = norm3(agent->state.omega),
        };

        float reward = task_reward(env, agent, i, &cache);
        reward -= env->alpha_vel * cache.vel;
        reward -= env->alpha_omega * cache.omega;

        float* action = &env->actions[4 * i];
        if (agent->episode_length > 1) {
            float da = 0.0f;
            for (int k = 0; k < 4; k++) {
                float d = action[k] - agent->prev_action[k];
                da += d * d;
            }
            reward -= env->alpha_action * da;
        }
        for (int k = 0; k < 4; k++) agent->prev_action[k] = action[k];

        bool done = task_done(env, agent, i, &cache);

        agent->episode_return += reward;
        env->rewards[i] = reward;
        env->terminals[i] = done ? 1.0f : 0.0f;

        if (done) {
            add_log(env, i, &cache);
            reset_agent_base(env, i);
            task_reset(env, agent, i);
            agent->prev_pos = agent->state.pos;
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    task_close(env);

    for (int i = 0; i < env->num_agents; i++)
        free(env->agents[i].target);
    free(env->agents);
    free(env->params_soa);

    if (env->client != NULL) c_close_client(env->client);
}