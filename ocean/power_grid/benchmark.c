#define POWER_GRID_NO_RENDER

#include "power_grid_solver.c"
#include "power_grid_ac.c"
#include "power_grid.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
    BASELINE_NO_ACTION,
    BASELINE_RANDOM,
    BASELINE_RANDOM_LINES,
    BASELINE_SAFE_RANDOM,
    BASELINE_GREEDY_LINES,
    BASELINE_GREEDY_ALL,
    BASELINE_COUNT,
} BaselinePolicy;

typedef struct {
    double perf;
    double score;
    double total_failure;
    double event_failure;
    double total_switches;
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

static const char *const BASELINE_NAMES[BASELINE_COUNT] = {
    "No action",
    "Uniform random",
    "Random lines",
    "Safe random",
    "Greedy lines",
    "Greedy all",
};

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
    if (power_grid_solve(&topology, &env->operating_point, &solution) !=
        POWER_GRID_SOLVE_OK)
    {
        return -INFINITY;
    }
    if (solution.status != POWER_GRID_SOLVE_OK)
        return -INFINITY;

    double congestion_cost = 0.0;
    double max_rho = 0.0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        double rating = POWER_GRID_BRANCHES[line].thermal_limit_mw *
                        env->branch_rating_scale;
        double loading = fabs(solution.branch_flow_mw[line]);
        double rho = loading / rating;
        max_rho = fmax(max_rho, rho);
        if (rho > 1.0)
            congestion_cost += (rho - 1.0) * (rho - 1.0);
    }
    int switched = memcmp(&topology, &env->topology, sizeof(topology)) != 0;
    return POWER_GRID_SAFE_STEP_REWARD * (max_rho <= 1.0) -
           POWER_GRID_CONGESTION_COST_WEIGHT * congestion_cost -
           POWER_GRID_SWITCH_PENALTY * switched;
}

static int baseline_action(const PowerGrid *env, BaselinePolicy policy, uint32_t *rng)
{
    if (policy == BASELINE_NO_ACTION)
        return POWER_GRID_ACTION_NONE;
    if (policy == BASELINE_RANDOM)
        return (int)(baseline_random(rng) % POWER_GRID_NUM_ACTIONS);
    if (policy == BASELINE_RANDOM_LINES)
        return (int)(baseline_random(rng) % POWER_GRID_TERMINAL_ACTION_OFFSET);

    if ((policy == BASELINE_GREEDY_LINES || policy == BASELINE_GREEDY_ALL) &&
        env->solution.max_rho <= 1.0)
        return POWER_GRID_ACTION_NONE;

    int action_limit = policy == BASELINE_GREEDY_LINES ?
                       POWER_GRID_TERMINAL_ACTION_OFFSET : POWER_GRID_NUM_ACTIONS;
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
    double random_event_probability, int random_outage_count)
{
    BenchmarkMetrics total = {0};
    for (int episode = 0; episode < episodes; episode++)
    {
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
        while (env.log.n == 0.0f)
        {
            env.actions[0] = (float)baseline_action(&env, policy, &action_rng);
            c_step(&env);
        }
        total.perf += env.log.perf;
        total.score += env.log.score;
        total.total_failure += env.log.total_failure;
        total.event_failure += env.log.event_failure;
        total.total_switches += env.log.total_switches;
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
    double scale = 1.0 / episodes;
    total.perf *= scale;
    total.score *= scale;
    total.total_failure *= scale;
    total.event_failure *= scale;
    total.total_switches *= scale;
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
    if (episodes <= 0)
    {
        fprintf(stderr, "episodes must be positive\n");
        return 1;
    }

    printf("policy,episodes,perf,score,total_failure,event_failure,"
           "total_switches,random_events,demand_fulfilled,outage_completion,"
           "all_outages_survived,thermal_trips,thermal_trip_episode,"
           "peak_thermal_stress,peak_line_loading,overloaded_line_fraction\n");
    for (int policy = 0; policy < BASELINE_COUNT; policy++)
    {
        BenchmarkMetrics metrics = run_baseline(
            (BaselinePolicy)policy, episodes, ac_power_flow,
            random_event_probability, random_outage_count);
        printf("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
               "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               BASELINE_NAMES[policy], episodes, metrics.perf, metrics.score,
               metrics.total_failure, metrics.event_failure,
               metrics.total_switches, metrics.random_events,
               metrics.demand_fulfilled, metrics.outage_completion,
               metrics.all_outages_survived, metrics.thermal_trips,
               metrics.thermal_trip_episode, metrics.peak_thermal_stress,
               metrics.peak_line_loading, metrics.overloaded_line_fraction);
        fflush(stdout);
    }
    return 0;
}
