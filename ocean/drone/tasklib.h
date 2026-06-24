#pragma once

#include "task_hover.h"
#include "task_race.h"

const char* task_name(TaskType task) {
    switch (task) {
        case TASK_HOVER: return "hover";
        case TASK_RACE: return "race";
        case TASK_SPHERE: return "sphere";
    }
    return "?";
}

void task_init(DroneEnv* env) {
    switch (env->task) {
        case TASK_HOVER: hover_init(env); break;
        case TASK_RACE: race_init(env); break;
        case TASK_SPHERE: hover_init(env); break;
    }
}

void task_close(DroneEnv* env) {
    switch (env->task) {
        case TASK_HOVER: hover_close(env); break;
        case TASK_RACE: race_close(env); break;
        case TASK_SPHERE: hover_close(env); break;
    }
}

void task_env_reset(DroneEnv* env) {
    switch (env->task) {
        case TASK_RACE: race_env_reset(env); break;
        default: break;
    }
}

void task_reset(DroneEnv* env, Drone* agent, int idx) {
    switch (env->task) {
        case TASK_HOVER: hover_reset(env, agent, idx); break;
        case TASK_RACE: race_reset(env, agent, idx); break;
        case TASK_SPHERE: sphere_reset(env, agent, idx); break;
    }
}

float task_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    switch (env->task) {
        case TASK_HOVER: return hover_reward(env, agent, idx, cache);
        case TASK_RACE: return race_reward(env, agent, idx, cache);
        case TASK_SPHERE: return hover_reward(env, agent, idx, cache);
    }
    return 0.0f;
}

bool task_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    switch (env->task) {
        case TASK_HOVER: return hover_done(env, agent, idx, cache);
        case TASK_RACE: return race_done(env, agent, idx, cache);
        case TASK_SPHERE: return hover_done(env, agent, idx, cache);
    }
    return false;
}

void task_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    switch (env->task) {
        case TASK_HOVER: hover_log(env, agent, idx, log, cache); break;
        case TASK_RACE: race_log(env, agent, idx, log, cache); break;
        case TASK_SPHERE: hover_log(env, agent, idx, log, cache); break;
    }
}
