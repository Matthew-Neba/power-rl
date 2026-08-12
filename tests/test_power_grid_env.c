#define POWER_GRID_NO_RENDER

#include "power_grid_solver.c"
#include "power_grid.h"

#include <math.h>
#include <stdio.h>

static int failures;

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)
#define CHECK_CLOSE(actual, expected) \
    CHECK(fabs((double)(actual) - (double)(expected)) < 1e-6)

static void test_contract_and_reward(void)
{
    CHECK(POWER_GRID_OBS_SIZE == 221);
    CHECK(POWER_GRID_NUM_ACTIONS == 91);
    PowerGrid env = {
        .failure_reward = POWER_GRID_DEFAULT_FAILURE_REWARD,
        .safe_step_reward = POWER_GRID_DEFAULT_SAFE_STEP_REWARD,
        .recovery_reward = POWER_GRID_DEFAULT_RECOVERY_REWARD,
        .switch_penalty = POWER_GRID_DEFAULT_SWITCH_PENALTY,
        .congestion_cost_weight = POWER_GRID_DEFAULT_CONGESTION_COST_WEIGHT,
    };
    CHECK_CLOSE(calculate_reward(&env, POWER_GRID_INVALID_TOPOLOGY, 0.0, 0, 0),
                POWER_GRID_DEFAULT_FAILURE_REWARD);
    CHECK_CLOSE(calculate_reward(&env, POWER_GRID_SOLVE_OK, 0.0, 0, 1), 0.01);
    CHECK_CLOSE(calculate_reward(&env, POWER_GRID_SOLVE_OK, 0.0, 1, 1), 0.0);
    CHECK_CLOSE(calculate_reward(&env, POWER_GRID_SOLVE_OK, 0.2, 0, 0), -0.2);
    CHECK(POWER_GRID_EPISODE_STEPS * POWER_GRID_DEFAULT_SAFE_STEP_REWARD <
          -POWER_GRID_DEFAULT_FAILURE_REWARD);

    power_grid_topology_normal(&env.topology);
    memset(env.line_available, 1, sizeof(env.line_available));
    PowerGridAppliedAction no_op = apply_agent_action(&env, 0.0f);
    CHECK(no_op.type == POWER_GRID_ACTION_NONE && !no_op.switched);
    PowerGridAppliedAction line = apply_agent_action(&env, 1.0f);
    CHECK(line.type == POWER_GRID_ACTION_LINE && line.switched);
    PowerGridAppliedAction invalid = apply_agent_action(&env, NAN);
    CHECK(invalid.type == POWER_GRID_ACTION_INVALID && !invalid.switched);
}

static void test_scenario_cache(void)
{
    CHECK(POWER_GRID_OFFLINE_TRAIN_COUNT == 365);
    CHECK(POWER_GRID_OFFLINE_VALIDATION_COUNT == 366);
    CHECK(POWER_GRID_OFFLINE_SCENARIO_COUNT == 731);
    CHECK(POWER_GRID_OFFLINE_SCENARIOS[0].date_yyyymmdd == 20190101u);
    CHECK(POWER_GRID_OFFLINE_SCENARIOS[730].date_yyyymmdd == 20201231u);

    int derated = 0, uprated = 0;
    double peak_loading = 0.0;
    PowerGrid env = {.offline_scenarios = 1};
    power_grid_allocate(&env);
    power_grid_topology_normal(&env.topology);
    memset(env.line_available, 1, sizeof(env.line_available));
    for (int index = 0; index < POWER_GRID_OFFLINE_SCENARIO_COUNT; index += 17)
    {
        env.offline_scenario = &POWER_GRID_OFFLINE_SCENARIOS[index];
        for (int period = 0; period < POWER_GRID_NUM_PERIODS; period++)
        {
            env.ac_power_flow = 0;
            power_grid_set_operating_period(&env, period);
            CHECK(power_grid_solve_environment(&env) == POWER_GRID_SOLVE_OK);
            derated += env.branch_rating_scale < 0.95;
            uprated += env.branch_rating_scale > 1.05;
            peak_loading = fmax(peak_loading, env.solution.max_rho);
        }
        env.ac_power_flow = 1;
        power_grid_set_operating_period(&env, 8);
        CHECK(power_grid_solve_environment(&env) == POWER_GRID_SOLVE_OK);
        CHECK(env.ac_solution.status == POWER_GRID_AC_OK);
    }
    CHECK(derated > 0);
    CHECK(uprated > 0);
    /* Historical days should approach the limit; deliberately overloaded
     * cases remain in the synthetic curriculum and random contingencies. */
    CHECK(peak_loading > 0.95);
    c_close(&env);
}

static void test_deterministic_sampling_and_observations(void)
{
    PowerGrid first = {
        .rng = 12345u, .offline_scenarios = 1, .offline_scenario_probability = 1.0,
    };
    PowerGrid second = first;
    power_grid_allocate(&first);
    power_grid_allocate(&second);
    c_reset(&first);
    c_reset(&second);
    CHECK(first.offline_scenario != NULL && second.offline_scenario != NULL);
    CHECK(first.offline_scenario->date_yyyymmdd == second.offline_scenario->date_yyyymmdd);
    CHECK(first.offline_scenario->date_yyyymmdd / 10000u ==
          POWER_GRID_OFFLINE_TRAIN_YEAR);
    CHECK_CLOSE(first.branch_rating_scale, second.branch_rating_scale);
    CHECK_CLOSE(first.observations[POWER_GRID_RATING_SCALE_OBS_OFFSET],
                first.branch_rating_scale);
    CHECK_CLOSE(first.observations[POWER_GRID_WEATHER_OBS_OFFSET],
                first.ambient_temperature_c / 50.0);
    CHECK(first.solution.status == POWER_GRID_SOLVE_OK);

    PowerGrid validation = {
        .rng = 12345u, .offline_scenarios = 1, .offline_scenario_validation = 1,
    };
    power_grid_allocate(&validation);
    c_reset(&validation);
    CHECK(validation.offline_scenario->date_yyyymmdd / 10000u ==
          POWER_GRID_OFFLINE_VALIDATION_YEAR);
    CHECK(validation.solution.status == POWER_GRID_SOLVE_OK);
    c_close(&validation);
    c_close(&second);
    c_close(&first);
}

static void test_no_op_dc_loading_diagnostics(void)
{
    PowerGrid env = {0};
    power_grid_allocate(&env);
    c_reset(&env);

    /* The no-op path deliberately reuses this cached solution. Injecting a
       known overload makes the accounting regression deterministic across
       future profile and branch-rating changes. */
    env.solution.branch_rho[0] = 1.25;
    env.solution.max_rho = fmax(env.solution.max_rho, 1.25);

    double expected_peak = env.solution.max_rho;
    int expected_overloaded = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        expected_overloaded += env.solution.branch_rho[line] > 1.0;
    CHECK(expected_overloaded > 0);
    env.solution.overloaded_branch_count = expected_overloaded;

    env.actions[0] = POWER_GRID_ACTION_NONE;
    c_step(&env);
    CHECK_CLOSE(env.episode.peak_line_loading, expected_peak);
    CHECK(env.episode.overloaded_line_steps == expected_overloaded);
    c_close(&env);
}

static void test_invalid_agent_switch_is_rejected(void)
{
    int bridge = -1;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        if (!power_grid_random_event_eligible(line))
        {
            bridge = line;
            break;
        }
    CHECK(bridge >= 0);

    PowerGrid env = {0};
    power_grid_allocate(&env);
    c_reset(&env);
    env.line_available[0] = 0;
    env.topology.line_closed[0] = 0;
    env.solution.max_rho = 1.1;
    env.solution.congestion_cost = 0.01;
    PowerGridSolveResult before = env.solution;
    env.actions[0] = (float)(POWER_GRID_LINE_ACTION_OFFSET + bridge);
    c_step(&env);

    CHECK(env.topology.line_closed[bridge]);
    CHECK(env.solution.status == POWER_GRID_SOLVE_OK);
    CHECK_CLOSE(env.solution.max_rho, before.max_rho);
    CHECK_CLOSE(env.rewards[0], -env.switch_penalty);
    CHECK(env.terminals[0] == 0.0f);
    CHECK(env.episode.switches[0] == 0);
    c_close(&env);
}

static void test_secure_grid_ignores_switch_commands(void)
{
    PowerGrid env = {0};
    power_grid_allocate(&env);
    c_reset(&env);
    CHECK(env.solution.max_rho <= 1.0);
    PowerGridTopology before = env.topology;

    env.actions[0] = POWER_GRID_LINE_ACTION_OFFSET;
    c_step(&env);

    CHECK(memcmp(&env.topology, &before, sizeof(before)) == 0);
    CHECK(env.last_action == POWER_GRID_ACTION_NONE);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 0);
    c_close(&env);
}

static void test_scaled_random_outages(void)
{
    for (int ac = 0; ac <= 1; ac++)
    {
        PowerGrid env = {
            .rng = 17u,
            .ac_power_flow = ac,
            .random_events = 1,
            .random_event_probability = 1.0,
            .random_outage_count = 3,
        };
        power_grid_allocate(&env);
        c_reset(&env);
        CHECK(env.scheduled_random_event_count == 3);
        PowerGridTopology combined;
        power_grid_topology_normal(&combined);
        for (int event = 0; event < 3; event++)
        {
            int line = env.scheduled_random_event_line[event];
            int period = env.scheduled_random_event_period[event];
            CHECK(power_grid_random_event_eligible(line));
            CHECK(period > 0 && period < POWER_GRID_NUM_PERIODS);
            combined.line_closed[line] = 0;
            for (int prior = 0; prior < event; prior++)
            {
                CHECK(line != env.scheduled_random_event_line[prior]);
                CHECK(period != env.scheduled_random_event_period[prior]);
            }
        }
        CHECK(power_grid_validate_topology(&combined, NULL, NULL) == POWER_GRID_SOLVE_OK);
        while (env.log.n == 0.0f)
        {
            env.actions[0] = POWER_GRID_ACTION_NONE;
            c_step(&env);
        }
        CHECK(env.log.random_events >= 1.0f && env.log.random_events <= 3.0f);
        CHECK(env.log.outage_completion > 0.0f && env.log.outage_completion <= 1.0f);
        c_close(&env);
    }
}

static void test_bridge_failure_and_single_episode_stop(void)
{
    int bridge = -1;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        if (!power_grid_random_event_eligible(line))
        {
            bridge = line;
            break;
        }
    CHECK(bridge >= 0);
    PowerGrid env = {.single_episode_evaluation = 1};
    power_grid_allocate(&env);
    c_reset(&env);
    env.scheduled_random_event_count = 1;
    env.scheduled_random_event_period[0] = 1;
    env.scheduled_random_event_line[0] = bridge;
    for (int step = 0; step < POWER_GRID_STEPS_PER_PERIOD; step++)
    {
        env.actions[0] = POWER_GRID_ACTION_NONE;
        c_step(&env);
    }
    CHECK_CLOSE(env.log.total_failure, 1.0);
    CHECK_CLOSE(env.log.connectivity_failure, 1.0);
    CHECK_CLOSE(env.log.event_failure, 1.0);
    env.actions[0] = POWER_GRID_ACTION_NONE;
    c_step(&env);
    CHECK(env.terminals[0] == 1.0f);
    CHECK_CLOSE(env.log.n, 1.0);
    c_close(&env);
}

static void test_second_outage_rolls_back_invalid_recovery_topology(void)
{
    int recovery_line = -1;
    int outage_line = -1;
    for (int first = 0; first < POWER_GRID_NUM_BRANCHES && recovery_line < 0; first++)
        for (int second = first + 1; second < POWER_GRID_NUM_BRANCHES; second++)
        {
            if (!power_grid_random_event_eligible(first) ||
                !power_grid_random_event_eligible(second))
                continue;
            PowerGridTopology topology;
            power_grid_topology_normal(&topology);
            topology.line_closed[first] = 0;
            topology.line_closed[second] = 0;
            if (power_grid_validate_topology(&topology, NULL, NULL) != POWER_GRID_SOLVE_OK)
            {
                recovery_line = first;
                outage_line = second;
                break;
            }
        }
    CHECK(recovery_line >= 0 && outage_line >= 0);

    PowerGrid env = {.single_episode_evaluation = 1};
    power_grid_allocate(&env);
    c_reset(&env);
    env.topology.line_closed[recovery_line] = 0;
    CHECK(power_grid_solve_environment(&env) == POWER_GRID_SOLVE_OK);
    env.scheduled_random_event_count = 1;
    env.scheduled_random_event_period[0] = 1;
    env.scheduled_random_event_line[0] = outage_line;
    for (int step = 0; step < POWER_GRID_STEPS_PER_PERIOD; step++)
    {
        env.actions[0] = POWER_GRID_ACTION_NONE;
        c_step(&env);
    }
    CHECK(env.solution.status == POWER_GRID_SOLVE_OK);
    CHECK(env.topology.line_closed[recovery_line]);
    CHECK(!env.topology.line_closed[outage_line]);
    CHECK(!env.line_available[outage_line]);
    CHECK_CLOSE(env.log.event_failure, 0.0);
    c_close(&env);
}

static void test_randomized_lifecycle(void)
{
    PowerGrid env = {
        .rng = 99u, .offline_scenarios = 1, .offline_scenario_probability = 1.0,
        .random_events = 1, .random_event_probability = 1.0, .random_outage_count = 3,
    };
    power_grid_allocate(&env);
    c_reset(&env);
    unsigned int random = 7u;
    for (int step = 0; step < 256; step++)
    {
        random = random * 1664525u + 1013904223u;
        env.actions[0] = (float)(random % POWER_GRID_NUM_ACTIONS);
        c_step(&env);
        CHECK(isfinite(env.rewards[0]));
        float optimized_observations[POWER_GRID_OBS_SIZE];
        memcpy(optimized_observations, env.observations,
               sizeof(optimized_observations));
        power_grid_compute_observations(&env);
        for (int observation = 0; observation < POWER_GRID_OBS_SIZE; observation++)
        {
            CHECK(isfinite(env.observations[observation]));
            CHECK_CLOSE(optimized_observations[observation],
                        env.observations[observation]);
        }
    }
    CHECK(env.log.n > 0.0f);
    c_close(&env);
}

int main(void)
{
    test_contract_and_reward();
    test_scenario_cache();
    test_deterministic_sampling_and_observations();
    test_no_op_dc_loading_diagnostics();
    test_invalid_agent_switch_is_rejected();
    test_secure_grid_ignores_switch_commands();
    test_scaled_random_outages();
    test_bridge_failure_and_single_episode_stop();
    test_second_outage_rolls_back_invalid_recovery_topology();
    test_randomized_lifecycle();
    if (failures)
        return 1;
    puts("IEEE-14 environment tests passed");
    return 0;
}
