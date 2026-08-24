#include <stdlib.h>
#include <time.h>

#include "power_grid_solver.c"
#include "power_grid.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "power_grid_policy.h"
#pragma clang diagnostic pop
#include "power_grid_user.h"

typedef struct
{
    Weights *weights;
    PufferNet *network;
} PowerGridRuntimePolicy;

static PowerGridRuntimePolicy power_grid_load_policy(void)
{
    PowerGridRuntimePolicy policy = {0};
    policy.weights = load_weights("resources/power_grid/policy.bin");
    if (policy.weights == NULL)
        return policy;
    int action_sizes[1] = {POWER_GRID_NUM_ACTIONS};
    policy.network = make_puffernet(policy.weights, 1, POWER_GRID_OBS_SIZE,
                                    POWER_GRID_POLICY_HIDDEN_SIZE,
                                    POWER_GRID_POLICY_NUM_LAYERS,
                                    action_sizes, 1);
    if (policy.network == NULL)
    {
        free(policy.weights);
        policy.weights = NULL;
    }
    return policy;
}

static void power_grid_free_policy(PowerGridRuntimePolicy *policy)
{
    if (policy->network == NULL)
        return;
    free_puffernet(policy->network);
    free(policy->weights);
    *policy = (PowerGridRuntimePolicy){0};
}

int main(void) {
    PowerGrid env = {0};
    /* The browser is a held-out AC validation demo. User clicks are the only
     * outage source, keeping the complete contingency at N-1 or N-2. */
    env.ac_power_flow = 1;
    env.offline_scenarios = 1;
    env.offline_scenario_validation = 1;
    env.random_events = 0;
    power_grid_allocate(&env);
    c_reset(&env);
    PowerGridUserSession user;
    power_grid_user_init(&user, 2);
    PowerGridRuntimePolicy policy = power_grid_load_policy();
    if (policy.network == NULL)
        return 1;
    PowerGridInferenceMode mode = POWER_GRID_INFERENCE_ARGMAX;
    int terminal_frames = 0;
    int terminal_failed = 0;
    srand((unsigned int)time(NULL));
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE))
            mode = POWER_GRID_INFERENCE_STOCHASTIC;
        if (IsKeyPressed(KEY_TWO))
            mode = POWER_GRID_INFERENCE_ARGMAX;
        const char *episode_status = terminal_frames == 0 ? "" :
            (terminal_failed ? " | GRID FAILURE - restarting" :
                               " | EPISODE COMPLETE - restarting");
        const char *user_status = user.last_status == POWER_GRID_SOLVE_OK ? "" :
            TextFormat(" | outage result: %s",
                       power_grid_solve_status_name(user.last_status));
        SetWindowTitle(TextFormat(
            "IEEE-14 AC | click up to 2 lines | %s (1 stochastic, 2 argmax) | outages %d/2%s%s",
            mode == POWER_GRID_INFERENCE_ARGMAX ? "ARGMAX" : "STOCHASTIC",
            power_grid_user_outage_count(&user),
            episode_status, user_status));

        if (terminal_frames > 0)
        {
            terminal_frames--;
            c_render(&env);
            if (terminal_frames == 0)
            {
                power_grid_free_policy(&policy);
                c_reset(&env);
                power_grid_user_init(&user, 2);
                policy = power_grid_load_policy();
                if (policy.network == NULL)
                    break;
            }
            continue;
        }
        power_grid_user_handle_input(&user, &env);
        power_grid_policy_action(policy.network, env.observations, env.actions, mode);
        c_step(&env);
        if (env.terminals[0] > 0.5f)
        {
            terminal_frames = 3;
            terminal_failed = env.solution.status != POWER_GRID_SOLVE_OK;
        }
        c_render(&env);
    }
    power_grid_free_policy(&policy);
    c_close(&env);
    return 0;
}
