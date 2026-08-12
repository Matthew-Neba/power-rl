#define POWER_GRID_NO_RENDER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "power_grid_policy.h"
#pragma clang diagnostic pop
#include "power_grid_solver.c"
#include "power_grid.h"

typedef enum {
    BASELINE_NO_ACTION,
    BASELINE_RANDOM,
    BASELINE_RANDOM_LINES,
    BASELINE_SAFE_RANDOM,
    BASELINE_GREEDY_LINES,
    BASELINE_GREEDY_ALL,
    BASELINE_LOOKAHEAD_LINES,
    BASELINE_COUNT,
    POLICY_PPO = BASELINE_COUNT,
    POLICY_PPO_STOCHASTIC,
    POLICY_COUNT,
} BaselinePolicy;

typedef struct {
    double perf;
    double score;
    double total_failure;
    double event_failure;
    double total_switches;
    double line_switches;
    double busbar_switches;
    double coupler_switches;
    double random_events;
    double demand_fulfilled;
    double outage_completion;
    double all_outages_survived;
    double thermal_trips;
    double thermal_trip_episode;
    double peak_thermal_stress;
    double peak_line_loading;
    double overloaded_line_fraction;
} BenchmarkMetrics;

static const char *const BASELINE_NAMES[POLICY_COUNT] = {
    "No action",
    "Uniform random",
    "Random lines",
    "Safe random",
    "Greedy lines",
    "Greedy all",
    "Lookahead lines",
    "PPO deterministic",
    "PPO stochastic",
};

static int ppo_action(PufferNet *policy, const float *observations, int deterministic)
{
    float action = 0.0f;
    power_grid_policy_action(policy, observations, &action,
        deterministic ? POWER_GRID_INFERENCE_ARGMAX :
                        POWER_GRID_INFERENCE_STOCHASTIC);
    return (int)action;
}

static uint32_t baseline_random(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state ^ (*state >> 16);
}

static double immediate_action_value(const PowerGrid *env, int action)
{
    PowerGridTopology topology = env->topology;
    PowerGridActionType type = power_grid_apply_action(&topology, action);
    if (type == POWER_GRID_ACTION_LINE)
    {
        int line = action - POWER_GRID_LINE_ACTION_OFFSET;
        if (!env->line_available[line])
            topology.line_closed[line] = 0;
    }

    PowerGridSolveResult solution;
    PowerGridACSolveResult ac_solution;
    if (env->ac_power_flow)
    {
        power_grid_ac_solve(&topology, &env->operating_point, &ac_solution);
        power_grid_ac_to_compatible(&ac_solution, &solution);
    }
    else if (power_grid_solve(&topology, &env->operating_point, &solution) !=
             POWER_GRID_SOLVE_OK)
        return -INFINITY;
    if (solution.status != POWER_GRID_SOLVE_OK)
        return -INFINITY;

    double congestion_cost = 0.0;
    double max_rho = 0.0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        double rating = (env->ac_power_flow ? power_grid_ac_branch_rating_mva(line) :
                         POWER_GRID_BRANCHES[line].thermal_limit_mw) *
                        env->branch_rating_scale;
        double loading = env->ac_power_flow ?
            fmax(ac_solution.branch_from_mva[line], ac_solution.branch_to_mva[line]) :
            fabs(solution.branch_flow_mw[line]);
        double rho = loading / rating;
        max_rho = fmax(max_rho, rho);
        if (rho > 1.0)
            congestion_cost += (rho - 1.0) * (rho - 1.0);
    }
    int switched = memcmp(&topology, &env->topology, sizeof(topology)) != 0;
    return env->safe_step_reward * (max_rho <= 1.0) -
           env->congestion_cost_weight * congestion_cost -
           env->congestion_progress_weight *
               (congestion_cost - env->solution.congestion_cost) -
           (env->solution.max_rho <= 1.0 ? env->secure_switch_penalty :
                                          env->switch_penalty) * switched;
}

static int baseline_action(const PowerGrid *env, BaselinePolicy policy, uint32_t *rng)
{
    if (policy == BASELINE_NO_ACTION)
        return POWER_GRID_ACTION_NONE;
    if (policy == BASELINE_RANDOM)
        return (int)(baseline_random(rng) % POWER_GRID_NUM_ACTIONS);
    if (policy == BASELINE_RANDOM_LINES)
        return (int)(baseline_random(rng) % POWER_GRID_TERMINAL_ACTION_OFFSET);

    if ((policy == BASELINE_GREEDY_LINES || policy == BASELINE_GREEDY_ALL ||
         policy == BASELINE_LOOKAHEAD_LINES) &&
        env->solution.max_rho <= 1.0)
        return POWER_GRID_ACTION_NONE;

    int action_limit = policy == BASELINE_GREEDY_LINES ?
                       POWER_GRID_TERMINAL_ACTION_OFFSET : POWER_GRID_NUM_ACTIONS;
    if (policy == BASELINE_LOOKAHEAD_LINES)
    {
        int best_action = POWER_GRID_ACTION_NONE;
        double best_value = 2.0 * immediate_action_value(env, best_action);
        for (int first = POWER_GRID_LINE_ACTION_OFFSET;
             first < POWER_GRID_TERMINAL_ACTION_OFFSET; first++)
        {
            PowerGrid candidate = *env;
            power_grid_apply_action(&candidate.topology, first);
            int line = first - POWER_GRID_LINE_ACTION_OFFSET;
            if (!candidate.line_available[line])
                candidate.topology.line_closed[line] = 0;
            if (power_grid_solve_scaled(&candidate.topology, &candidate.operating_point,
                                        &candidate.solution,
                                        candidate.branch_rating_scale) != POWER_GRID_SOLVE_OK ||
                candidate.solution.congestion_cost >= env->solution.congestion_cost - 1e-12)
                continue;
            double first_value = immediate_action_value(env, first);
            double second_value = immediate_action_value(&candidate, POWER_GRID_ACTION_NONE);
            for (int second = POWER_GRID_LINE_ACTION_OFFSET;
                 second < POWER_GRID_TERMINAL_ACTION_OFFSET; second++)
                second_value = fmax(second_value,
                                    immediate_action_value(&candidate, second));
            double value = first_value + second_value;
            if (value > best_value + 1e-12)
            {
                best_value = value;
                best_action = first;
            }
        }
        return best_action;
    }
    if (policy == BASELINE_SAFE_RANDOM)
    {
        int valid[POWER_GRID_NUM_ACTIONS];
        int count = 0;
        for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++)
        {
            if (isfinite(immediate_action_value(env, action)))
                valid[count++] = action;
        }
        return valid[baseline_random(rng) % (unsigned int)count];
    }

    int best_action = POWER_GRID_ACTION_NONE;
    double best_value = immediate_action_value(env, best_action);
    for (int action = 1; action < action_limit; action++)
    {
        double value = immediate_action_value(env, action);
        if (value > best_value + 1e-12)
        {
            best_value = value;
            best_action = action;
        }
    }
    return best_action;
}

static BenchmarkMetrics run_baseline(
    BaselinePolicy policy, int episodes, int ac_power_flow,
    double random_event_probability, int random_outage_count, int seed_offset,
    const char *checkpoint)
{
    BenchmarkMetrics total = {0};
    Weights *weights = NULL;
    PufferNet *learned = NULL;
    if (policy == POLICY_PPO || policy == POLICY_PPO_STOCHASTIC)
    {
        weights = load_weights(checkpoint);
        if (weights == NULL)
            exit(1);
        int action_sizes[1] = {POWER_GRID_NUM_ACTIONS};
        learned = make_puffernet(weights, 1, POWER_GRID_OBS_SIZE,
                                 POWER_GRID_POLICY_HIDDEN_SIZE,
                                 POWER_GRID_POLICY_NUM_LAYERS, action_sizes, 1);
        if (learned == NULL)
        {
            free(weights);
            exit(1);
        }
    }
    for (int episode_index = 0; episode_index < episodes; episode_index++)
    {
        int episode = seed_offset + episode_index;
        PowerGrid env = {
            .rng = (unsigned int)episode,
            .ac_power_flow = ac_power_flow,
            .offline_scenarios = 1,
            .offline_scenario_validation = 1,
            .offline_scenario_probability = 0.65,
            .random_events = random_event_probability > 0.0,
            .random_event_probability = random_event_probability,
            .random_outage_count = random_outage_count,
        };
        uint32_t action_rng = UINT32_C(0x9e3779b9) ^ (uint32_t)episode;
        power_grid_allocate(&env);
        c_reset(&env);
        srand((unsigned int)episode);
        if (learned != NULL)
            memset(learned->mingru->state, 0,
                   (size_t)learned->mingru->num_layers * learned->mingru->hidden_size *
                       sizeof(float));
        while (env.log.n == 0.0f)
        {
            env.actions[0] = learned == NULL ?
                (float)baseline_action(&env, policy, &action_rng) :
                (float)ppo_action(learned, env.observations,
                                  policy == POLICY_PPO);
            c_step(&env);
        }
        total.perf += env.log.perf;
        total.score += env.log.score;
        total.total_failure += env.log.total_failure;
        total.event_failure += env.log.event_failure;
        total.total_switches += env.log.total_switches;
        total.line_switches += env.log.line_switches;
        total.busbar_switches += env.log.busbar_switches;
        total.coupler_switches += env.log.coupler_switches;
        total.random_events += env.log.random_events;
        total.demand_fulfilled += env.log.demand_fulfilled;
        total.outage_completion += env.log.outage_completion;
        total.all_outages_survived += env.log.all_outages_survived;
        total.thermal_trips += env.log.thermal_trips;
        total.thermal_trip_episode += env.log.thermal_trip_episode;
        total.peak_thermal_stress += env.log.peak_thermal_stress;
        total.peak_line_loading += env.log.peak_line_loading;
        total.overloaded_line_fraction += env.log.overloaded_line_fraction;
        c_close(&env);
    }
    if (learned != NULL)
    {
        free_puffernet(learned);
        free(weights);
    }
    double scale = 1.0 / episodes;
    total.perf *= scale;
    total.score *= scale;
    total.total_failure *= scale;
    total.event_failure *= scale;
    total.total_switches *= scale;
    total.line_switches *= scale;
    total.busbar_switches *= scale;
    total.coupler_switches *= scale;
    total.random_events *= scale;
    total.demand_fulfilled *= scale;
    total.outage_completion *= scale;
    total.all_outages_survived *= scale;
    total.thermal_trips *= scale;
    total.thermal_trip_episode *= scale;
    total.peak_thermal_stress *= scale;
    total.peak_line_loading *= scale;
    total.overloaded_line_fraction *= scale;
    return total;
}

int main(int argc, char **argv)
{
    int episodes = argc > 1 ? atoi(argv[1]) : 1024;
    int ac_power_flow = argc > 2 ? atoi(argv[2]) : 0;
    double random_event_probability = argc > 3 ? atof(argv[3]) : 0.25;
    int random_outage_count = argc > 4 ? atoi(argv[4]) : 3;
    int selected_policy = argc > 5 ? atoi(argv[5]) : -1;
    int seed_offset = argc > 6 ? atoi(argv[6]) : 0;
    const char *checkpoint = argc > 7 ? argv[7] : NULL;
    if (episodes <= 0)
    {
        fprintf(stderr, "episodes must be positive\n");
        return 1;
    }
    if (selected_policy < -1 || selected_policy >= POLICY_COUNT)
    {
        fprintf(stderr, "policy must be -1 or between 0 and %d\n", POLICY_COUNT - 1);
        return 1;
    }
    if ((selected_policy == POLICY_PPO || selected_policy == POLICY_PPO_STOCHASTIC) &&
        checkpoint == NULL)
    {
        fprintf(stderr, "deterministic PPO policy requires a checkpoint path\n");
        return 1;
    }
    if (seed_offset < 0)
    {
        fprintf(stderr, "seed offset must be non-negative\n");
        return 1;
    }

    printf("policy,episodes,perf,score,total_failure,event_failure,"
           "total_switches,line_switches,busbar_switches,coupler_switches,"
           "random_events,demand_fulfilled,outage_completion,"
           "all_outages_survived,thermal_trips,thermal_trip_episode,"
           "peak_thermal_stress,peak_line_loading,overloaded_line_fraction\n");
    int first_policy = selected_policy < 0 ? 0 : selected_policy;
    int final_policy = selected_policy < 0 ? BASELINE_COUNT : selected_policy + 1;
    for (int policy = first_policy; policy < final_policy; policy++)
    {
        BenchmarkMetrics metrics = run_baseline(
            (BaselinePolicy)policy, episodes, ac_power_flow,
            random_event_probability, random_outage_count, seed_offset, checkpoint);
        printf("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
               "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               BASELINE_NAMES[policy], episodes, metrics.perf, metrics.score,
               metrics.total_failure, metrics.event_failure,
               metrics.total_switches, metrics.line_switches,
               metrics.busbar_switches, metrics.coupler_switches,
               metrics.random_events,
               metrics.demand_fulfilled, metrics.outage_completion,
               metrics.all_outages_survived, metrics.thermal_trips,
               metrics.thermal_trip_episode, metrics.peak_thermal_stress,
               metrics.peak_line_loading, metrics.overloaded_line_fraction);
        fflush(stdout);
    }
    return 0;
}
