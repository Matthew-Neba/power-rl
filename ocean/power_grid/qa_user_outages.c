#define POWER_GRID_NO_RENDER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "power_grid_policy.h"
#pragma clang diagnostic pop
#include "power_grid_solver.c"
#include "power_grid.h"
#include "power_grid_user.h"

#define QA_MIN_SAFE_STEPS 0.95
#define QA_MIN_SERVED_LOAD 0.90

typedef struct
{
    int trials;
    int feasible;
    int survived;
    int handled;
    int secure_enough;
    int served_enough;
    int survived_secure;
    int survived_served;
    int one_step_secure;
    int feasible_survived;
    int feasible_handled;
    int thermal_trip_episodes;
    double safe_fraction;
    double peak_rho;
    double served_load_fraction;
    double load_shed_actions;
    double generator_trip_actions;
    int premature_emergency_trials;
    double premature_emergency_actions;
} UserOutageMetrics;

/* Diagnostic oracle only: determine whether the post-click AC state has any
 * unrestricted one-step action that is solved, within thermal/voltage limits,
 * and still serves the QA demand floor. It never chooses a policy action. */
static int has_one_step_secure_action(const PowerGrid *env)
{
    for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++)
    {
        PowerGrid probe = *env;
        power_grid_apply_action(&probe.topology, action);
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            if (!probe.line_available[line])
                probe.topology.line_closed[line] = 0;
        if (power_grid_solve_environment(&probe) == POWER_GRID_SOLVE_OK &&
            probe.solution.max_rho <= 1.0 &&
            probe.ac_solution.voltage_violation_count == 0 &&
            power_grid_served_load_fraction(&probe) >= QA_MIN_SERVED_LOAD)
            return 1;
    }
    return 0;
}

static void reset_policy(PufferNet *policy)
{
    memset(policy->mingru->state, 0,
           (size_t)policy->mingru->num_layers * policy->mingru->hidden_size *
               sizeof(float));
}

static void run_trial(PufferNet *policy, int first, int second, int context,
                      UserOutageMetrics *metrics)
{
    int pair = second >= 0;
    int case_id = first * POWER_GRID_NUM_BRANCHES + (pair ? second : first);
    PowerGrid env = {
        .rng = UINT32_C(0x51f15e5d) ^ (uint32_t)(context * 421 + case_id),
        .ac_power_flow = 1,
        .offline_scenarios = 1,
        .offline_scenario_validation = 1,
        .random_events = 0,
        .single_episode_evaluation = 1,
    };
    power_grid_allocate(&env);
    c_reset(&env);
    reset_policy(policy);

    PowerGridUserSession user;
    power_grid_user_init(&user, 2);
    int outage_step = POWER_GRID_STEPS_PER_PERIOD * (1 + (context + case_id) % 9);
    int post_steps = 0;
    int safe_steps = 0;
    int request_feasible = 1;
    int premature_emergency_actions = 0;
    int one_step_secure = 0;
    double peak_rho = 0.0;

    while (env.log.n == 0.0f)
    {
        if (env.episode_step == outage_step)
        {
            request_feasible &= power_grid_user_set_line_outage(
                &user, &env, context % 2 == 0 || !pair ? first : second, 1) ==
                POWER_GRID_SOLVE_OK;
            if (!pair)
                one_step_secure = has_one_step_secure_action(&env);
        }
        if (pair && env.episode_step == outage_step + 1)
        {
            request_feasible &= power_grid_user_set_line_outage(
                &user, &env, context % 2 == 0 ? second : first, 1) ==
                POWER_GRID_SOLVE_OK;
            one_step_secure = has_one_step_secure_action(&env);
        }

        int before_request = env.episode_step < outage_step;
        power_grid_policy_action(policy, env.observations, env.actions,
                                 POWER_GRID_INFERENCE_ARGMAX);
        c_step(&env);
        if (before_request &&
            (env.last_action_type == POWER_GRID_ACTION_LOAD_SHED ||
             env.last_action_type == POWER_GRID_ACTION_GENERATOR_TRIP))
            premature_emergency_actions++;
        if (env.episode_step > outage_step)
        {
            post_steps++;
            int secure = env.solution.status == POWER_GRID_SOLVE_OK &&
                         env.solution.max_rho <= 1.0 &&
                         env.ac_solution.voltage_violation_count == 0;
            safe_steps += secure;
            if (env.solution.status == POWER_GRID_SOLVE_OK)
                peak_rho = fmax(peak_rho, env.solution.max_rho);
        }
    }

    int survived = env.log.total_failure == 0.0f;
    double safe_fraction = post_steps > 0 ? (double)safe_steps / post_steps : 0.0;
    metrics->trials++;
    metrics->feasible += request_feasible;
    metrics->survived += survived;
    int secure_enough = safe_fraction >= QA_MIN_SAFE_STEPS;
    int served_enough = env.log.demand_fulfilled >= QA_MIN_SERVED_LOAD;
    metrics->secure_enough += secure_enough;
    metrics->served_enough += served_enough;
    metrics->survived_secure += survived && secure_enough;
    metrics->survived_served += survived && served_enough;
    metrics->one_step_secure += one_step_secure;
    int handled = survived && premature_emergency_actions == 0 &&
                  secure_enough && served_enough;
    metrics->handled += handled;
    metrics->feasible_survived += request_feasible && survived;
    metrics->feasible_handled +=
        request_feasible && handled;
    metrics->thermal_trip_episodes += env.log.thermal_trip_episode > 0.0f;
    metrics->safe_fraction += safe_fraction;
    metrics->peak_rho += peak_rho;
    metrics->served_load_fraction += env.log.demand_fulfilled;
    metrics->load_shed_actions += env.log.load_shed_actions;
    metrics->generator_trip_actions += env.log.generator_trip_actions;
    metrics->premature_emergency_trials += premature_emergency_actions > 0;
    metrics->premature_emergency_actions += premature_emergency_actions;
    c_close(&env);
}

static void print_metrics(const char *name, int combinations,
                          const UserOutageMetrics *metrics)
{
    double scale = metrics->trials > 0 ? 1.0 / metrics->trials : 0.0;
    double feasible_scale = metrics->feasible > 0 ? 1.0 / metrics->feasible : 0.0;
    printf("%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           name,
           combinations, metrics->trials, metrics->feasible * scale,
           metrics->survived * scale, metrics->handled * scale,
           metrics->feasible_survived * feasible_scale,
           metrics->feasible_handled * feasible_scale,
           metrics->safe_fraction * scale, metrics->peak_rho * scale,
           metrics->thermal_trip_episodes * scale,
           metrics->served_load_fraction * scale,
           metrics->load_shed_actions * scale,
           metrics->generator_trip_actions * scale,
           metrics->premature_emergency_trials * scale,
           metrics->premature_emergency_actions * scale,
           metrics->secure_enough * scale, metrics->served_enough * scale,
           metrics->survived_secure * scale,
           metrics->survived_served * scale,
           metrics->one_step_secure * scale);
}

int main(int argc, char **argv)
{
    const char *checkpoint = argc > 1 ? argv[1] : "resources/power_grid/policy.bin";
    int contexts = argc > 2 ? atoi(argv[2]) : 8;
    int context_offset = argc > 3 ? atoi(argv[3]) : 0;
    if (contexts < 2)
    {
        fprintf(stderr, "contexts must be at least 2 to cover both N-2 orders\n");
        return 2;
    }
    if (context_offset < 0)
    {
        fprintf(stderr, "context offset must be non-negative\n");
        return 2;
    }

    Weights *weights = load_weights(checkpoint);
    if (weights == NULL)
        return 1;
    int action_sizes[1] = {POWER_GRID_NUM_ACTIONS};
    PufferNet *policy = make_puffernet(
        weights, 1, POWER_GRID_OBS_SIZE, POWER_GRID_POLICY_HIDDEN_SIZE,
        POWER_GRID_POLICY_NUM_LAYERS, action_sizes, 1);
    if (policy == NULL)
    {
        free(weights);
        return 1;
    }

    UserOutageMetrics n1 = {0};
    UserOutageMetrics n2 = {0};
    for (int first = 0; first < POWER_GRID_NUM_BRANCHES; first++)
    {
        for (int context = context_offset;
             context < context_offset + contexts; context++)
            run_trial(policy, first, -1, context, &n1);
        for (int second = first + 1; second < POWER_GRID_NUM_BRANCHES; second++)
            for (int context = context_offset;
                 context < context_offset + contexts; context++)
                run_trial(policy, first, second, context, &n2);
    }

    puts("mode,combinations,trials,feasible_rate,survival_rate,handle_rate,"
         "feasible_survival_rate,feasible_handle_rate,safe_step_rate,"
         "mean_peak_rho,thermal_trip_rate,served_load_rate,mean_load_sheds,"
         "mean_generator_trips,premature_emergency_rate,"
         "mean_premature_emergency_actions,secure_enough_rate,"
         "served_enough_rate,survived_secure_rate,survived_served_rate,"
         "one_step_secure_rate");
    print_metrics("N-1", POWER_GRID_NUM_BRANCHES, &n1);
    print_metrics("N-2", POWER_GRID_NUM_BRANCHES * (POWER_GRID_NUM_BRANCHES - 1) / 2,
                  &n2);
    UserOutageMetrics all = {
        .trials = n1.trials + n2.trials,
        .feasible = n1.feasible + n2.feasible,
        .survived = n1.survived + n2.survived,
        .handled = n1.handled + n2.handled,
        .secure_enough = n1.secure_enough + n2.secure_enough,
        .served_enough = n1.served_enough + n2.served_enough,
        .survived_secure = n1.survived_secure + n2.survived_secure,
        .survived_served = n1.survived_served + n2.survived_served,
        .one_step_secure = n1.one_step_secure + n2.one_step_secure,
        .feasible_survived = n1.feasible_survived + n2.feasible_survived,
        .feasible_handled = n1.feasible_handled + n2.feasible_handled,
        .thermal_trip_episodes = n1.thermal_trip_episodes + n2.thermal_trip_episodes,
        .safe_fraction = n1.safe_fraction + n2.safe_fraction,
        .peak_rho = n1.peak_rho + n2.peak_rho,
        .served_load_fraction = n1.served_load_fraction + n2.served_load_fraction,
        .load_shed_actions = n1.load_shed_actions + n2.load_shed_actions,
        .generator_trip_actions =
            n1.generator_trip_actions + n2.generator_trip_actions,
        .premature_emergency_trials =
            n1.premature_emergency_trials + n2.premature_emergency_trials,
        .premature_emergency_actions =
            n1.premature_emergency_actions + n2.premature_emergency_actions,
    };
    print_metrics("all", POWER_GRID_NUM_BRANCHES * (POWER_GRID_NUM_BRANCHES + 1) / 2,
                  &all);

    free_puffernet(policy);
    free(weights);
    return 0;
}
