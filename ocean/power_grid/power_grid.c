#include <stdlib.h>
#include <string.h>

#include "../../src/puffernet.h"
#include "power_grid_solver.c"
#include "power_grid.h"
#include "power_grid_user.h"

static void power_grid_policy_action(PufferNet *policy, PowerGrid *env)
{
    linear(policy->encoder, env->observations);
    mingru(policy->mingru, policy->encoder->output);
    linear(policy->decoder, policy->mingru->output);
    argmax_multidiscrete(policy->multidiscrete, policy->decoder->output,
                         env->actions);
}

int main(void) {
    PowerGrid env = {0};
    /* This checkpoint was trained with the DC environment. AC remains an
     * offline validation mode, not a silent deployment-time domain change. */
    env.ac_power_flow = 0;
    env.offline_scenarios = 1;
    env.offline_scenario_validation = 1;
    /* The live app is user-driven. Random outages remain a training option,
     * but must not compete with the one or two lines selected in the UI. */
    env.random_events = 0;
    power_grid_allocate(&env);
    env.max_episode_steps = 0;
    c_reset(&env);
    PowerGridUserSession user;
    power_grid_user_init(&user, 2);
    Weights *weights = load_weights("resources/power_grid/policy.bin");
    if (weights == NULL)
        return 1;
    int action_sizes[1] = {POWER_GRID_TERMINAL_ACTION_OFFSET};
    PufferNet *policy = make_puffernet(weights, 1, POWER_GRID_OBS_SIZE,
                                       512, 1, action_sizes, 1);
    c_render(&env);
    while (!WindowShouldClose()) {
        power_grid_user_handle_input(&user, &env);
        power_grid_policy_action(policy, &env);
        c_step(&env);
        if (env.episode_step > 0 &&
            env.episode_step % POWER_GRID_EPISODE_STEPS == 0)
            memset(policy->mingru->state, 0,
                   (size_t)policy->mingru->num_layers *
                   policy->mingru->hidden_size * sizeof(float));
        c_render(&env);
    }
    free_puffernet(policy);
    free(weights);
    c_close(&env);
    return 0;
}
