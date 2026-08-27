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
#ifdef POWER_GRID_POLICY_MLP
#include "power_grid_mlp_policy.h"
#define PufferNet PowerGridMlpPolicy
#define power_grid_policy_action power_grid_mlp_policy_action
#endif

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
    long pre_action_count[POWER_GRID_NUM_ACTIONS];
    long first_response_action_count[POWER_GRID_NUM_ACTIONS];
    long unsafe_first_response_action_count[POWER_GRID_NUM_ACTIONS];
    long post_action_count[POWER_GRID_NUM_ACTIONS];
    long unsafe_first_responses;
    long unsafe_first_responses_secured;
    long repeated_post_actions;
    long total_post_actions;
    long repeated_nonnoop_post_actions;
    long total_nonnoop_post_actions;
    long solved_post_steps;
    long thermal_unsafe_post_steps;
    long voltage_unsafe_post_steps;
    long voltage_only_unsafe_post_steps;
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
            probe.ac_solution.voltage_violation_count == 0)
            return 1;
    }
    return 0;
}

static void reset_policy(PufferNet *policy)
{
#ifndef POWER_GRID_POLICY_MLP
    memset(policy->mingru->state, 0,
           (size_t)policy->mingru->num_layers * policy->mingru->hidden_size *
               sizeof(float));
#else
    (void)policy;
#endif
}

static void run_trial(PufferNet *policy, int first, int second, int context,
                      int ac_power_flow, int forced_outage_step,
                      UserOutageMetrics *metrics)
{
    int pair = second >= 0;
    int case_id = first * POWER_GRID_NUM_BRANCHES + (pair ? second : first);
    PowerGrid env = {
        .rng = UINT32_C(0x51f15e5d) ^ (uint32_t)(context * 421 + case_id),
        .ac_power_flow = ac_power_flow,
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
    int outage_step = forced_outage_step >= 0 ? forced_outage_step :
        POWER_GRID_STEPS_PER_PERIOD * (1 + (context + case_id) % 9);
    int post_steps = 0;
    int safe_steps = 0;
    int request_feasible = 1;
    int one_step_secure = 0;
    double peak_rho = 0.0;
    int previous_post_action = -1;

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
        if (pair && env.episode_step == outage_step)
        {
            request_feasible &= power_grid_user_set_line_outage(
                &user, &env, context % 2 == 0 ? second : first, 1) ==
                POWER_GRID_SOLVE_OK;
            one_step_secure = has_one_step_secure_action(&env);
        }

        int before_request = env.episode_step < outage_step;
        int first_response_was_unsafe = 0;
        power_grid_policy_action(policy, env.observations, env.actions,
                                 POWER_GRID_INFERENCE_ARGMAX);
        int policy_action = (int)env.actions[0];
        if (before_request)
            metrics->pre_action_count[policy_action]++;
        else
        {
            metrics->post_action_count[policy_action]++;
            metrics->total_post_actions++;
            if (env.episode_step == outage_step)
            {
                metrics->first_response_action_count[policy_action]++;
                int unsafe = env.solution.status != POWER_GRID_SOLVE_OK ||
                    env.solution.max_rho > 1.0 ||
                    (env.ac_power_flow &&
                     env.ac_solution.voltage_violation_count > 0);
                if (unsafe)
                {
                    first_response_was_unsafe = 1;
                    metrics->unsafe_first_response_action_count[policy_action]++;
                    metrics->unsafe_first_responses++;
                }
            }
            if (policy_action == previous_post_action)
                metrics->repeated_post_actions++;
            if (policy_action != POWER_GRID_ACTION_NONE)
            {
                metrics->total_nonnoop_post_actions++;
                if (policy_action == previous_post_action)
                    metrics->repeated_nonnoop_post_actions++;
            }
            previous_post_action = policy_action;
        }
        c_step(&env);
        if (!before_request && env.episode_step == outage_step + 1 &&
            env.solution.status == POWER_GRID_SOLVE_OK &&
            env.solution.max_rho <= 1.0 &&
            (!env.ac_power_flow ||
             env.ac_solution.voltage_violation_count == 0))
            metrics->unsafe_first_responses_secured +=
                first_response_was_unsafe;
        if (env.episode_step > outage_step)
        {
            post_steps++;
            if (env.solution.status == POWER_GRID_SOLVE_OK)
            {
                int thermal_unsafe = env.solution.max_rho > 1.0;
                int voltage_unsafe = env.ac_power_flow &&
                    env.ac_solution.voltage_violation_count > 0;
                metrics->solved_post_steps++;
                metrics->thermal_unsafe_post_steps += thermal_unsafe;
                metrics->voltage_unsafe_post_steps += voltage_unsafe;
                metrics->voltage_only_unsafe_post_steps +=
                    voltage_unsafe && !thermal_unsafe;
            }
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
    int handled = survived && secure_enough && served_enough;
    metrics->handled += handled;
    metrics->feasible_survived += request_feasible && survived;
    metrics->feasible_handled +=
        request_feasible && handled;
    metrics->thermal_trip_episodes += env.log.thermal_trip_episode > 0.0f;
    metrics->safe_fraction += safe_fraction;
    metrics->peak_rho += peak_rho;
    metrics->served_load_fraction += env.log.demand_fulfilled;
    c_close(&env);
}

static void print_action_counts(const char *phase, const long *counts)
{
    for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++)
        if (counts[action] > 0)
            printf("action,%s,%d,%ld\n", phase, action, counts[action]);
}

static void print_metrics(const char *name, int combinations,
                          const UserOutageMetrics *metrics)
{
    double scale = metrics->trials > 0 ? 1.0 / metrics->trials : 0.0;
    double feasible_scale = metrics->feasible > 0 ? 1.0 / metrics->feasible : 0.0;
    printf("%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           name,
           combinations, metrics->trials, metrics->feasible * scale,
           metrics->survived * scale, metrics->handled * scale,
           metrics->feasible_survived * feasible_scale,
           metrics->feasible_handled * feasible_scale,
           metrics->safe_fraction * scale, metrics->peak_rho * scale,
           metrics->thermal_trip_episodes * scale,
           metrics->served_load_fraction * scale,
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
    int ac_power_flow = argc > 4 ? atoi(argv[4]) : 1;
    int forced_outage_step = argc > 5 ? atoi(argv[5]) : -1;
    int print_diagnostics = argc > 6 ? atoi(argv[6]) : 0;
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
    PufferNet *policy =
#ifdef POWER_GRID_POLICY_MLP
        power_grid_make_mlp_policy(weights);
#else
        make_puffernet(
        weights, 1, POWER_GRID_OBS_SIZE, POWER_GRID_POLICY_HIDDEN_SIZE,
        POWER_GRID_POLICY_NUM_LAYERS, action_sizes, 1);
#endif
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
            run_trial(policy, first, -1, context, ac_power_flow,
                      forced_outage_step, &n1);
        for (int second = first + 1; second < POWER_GRID_NUM_BRANCHES; second++)
            for (int context = context_offset;
                 context < context_offset + contexts; context++)
                run_trial(policy, first, second, context, ac_power_flow,
                          forced_outage_step, &n2);
    }

    puts("mode,combinations,trials,feasible_rate,survival_rate,handle_rate,"
         "feasible_survival_rate,feasible_handle_rate,safe_step_rate,"
         "mean_peak_rho,thermal_trip_rate,served_load_rate,secure_enough_rate,"
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
        .solved_post_steps = n1.solved_post_steps + n2.solved_post_steps,
        .thermal_unsafe_post_steps = n1.thermal_unsafe_post_steps +
                                     n2.thermal_unsafe_post_steps,
        .voltage_unsafe_post_steps = n1.voltage_unsafe_post_steps +
                                     n2.voltage_unsafe_post_steps,
        .voltage_only_unsafe_post_steps = n1.voltage_only_unsafe_post_steps +
                                          n2.voltage_only_unsafe_post_steps,
    };
    for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++)
    {
        all.pre_action_count[action] =
            n1.pre_action_count[action] + n2.pre_action_count[action];
        all.first_response_action_count[action] =
            n1.first_response_action_count[action] +
            n2.first_response_action_count[action];
        all.unsafe_first_response_action_count[action] =
            n1.unsafe_first_response_action_count[action] +
            n2.unsafe_first_response_action_count[action];
        all.post_action_count[action] =
            n1.post_action_count[action] + n2.post_action_count[action];
    }
    all.repeated_post_actions =
        n1.repeated_post_actions + n2.repeated_post_actions;
    all.total_post_actions = n1.total_post_actions + n2.total_post_actions;
    all.repeated_nonnoop_post_actions =
        n1.repeated_nonnoop_post_actions + n2.repeated_nonnoop_post_actions;
    all.total_nonnoop_post_actions =
        n1.total_nonnoop_post_actions + n2.total_nonnoop_post_actions;
    all.unsafe_first_responses =
        n1.unsafe_first_responses + n2.unsafe_first_responses;
    all.unsafe_first_responses_secured =
        n1.unsafe_first_responses_secured + n2.unsafe_first_responses_secured;
    print_metrics("all", POWER_GRID_NUM_BRANCHES * (POWER_GRID_NUM_BRANCHES + 1) / 2,
                  &all);
    if (print_diagnostics)
    {
        print_action_counts("n1_unsafe_first_response",
                            n1.unsafe_first_response_action_count);
        printf("action,n1_unsafe_first_secured_fraction,-1,%.6f\n",
               n1.unsafe_first_responses > 0 ?
                   (double)n1.unsafe_first_responses_secured /
                       n1.unsafe_first_responses : 0.0);
        print_action_counts("pre", all.pre_action_count);
        print_action_counts("first_response", all.first_response_action_count);
        print_action_counts("unsafe_first_response",
                            all.unsafe_first_response_action_count);
        print_action_counts("post", all.post_action_count);
        printf("action,repeated_post_fraction,-1,%.6f\n",
               all.total_post_actions > 0 ?
                   (double)all.repeated_post_actions / all.total_post_actions : 0.0);
        printf("action,repeated_nonnoop_post_fraction,-1,%.6f\n",
               all.total_nonnoop_post_actions > 0 ?
                   (double)all.repeated_nonnoop_post_actions /
                       all.total_nonnoop_post_actions : 0.0);
        printf("action,unsafe_first_secured_fraction,-1,%.6f\n",
               all.unsafe_first_responses > 0 ?
                   (double)all.unsafe_first_responses_secured /
                       all.unsafe_first_responses : 0.0);
        printf("diagnostic,thermal_unsafe_solved_post_fraction,-1,%.6f\n",
               all.solved_post_steps > 0 ?
                   (double)all.thermal_unsafe_post_steps /
                       all.solved_post_steps : 0.0);
        printf("diagnostic,voltage_unsafe_solved_post_fraction,-1,%.6f\n",
               all.solved_post_steps > 0 ?
                   (double)all.voltage_unsafe_post_steps /
                       all.solved_post_steps : 0.0);
        printf("diagnostic,voltage_only_unsafe_solved_post_fraction,-1,%.6f\n",
               all.solved_post_steps > 0 ?
                   (double)all.voltage_only_unsafe_post_steps /
                       all.solved_post_steps : 0.0);
    }

#ifdef POWER_GRID_POLICY_MLP
    free(policy);
#else
    free_puffernet(policy);
#endif
    free(weights);
    return 0;
}
