#define POWER_GRID_NO_RENDER

#include "power_grid_solver.c"
#include "power_grid_ac.c"
#include "power_grid.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)

#define CHECK_CLOSE(actual, expected) \
    CHECK(fabs((double)(actual) - (double)(expected)) < 1e-6)

static void test_action_application_and_bookkeeping(void)
{
    PowerGrid env = {0};
    power_grid_topology_normal(&env.topology);
    memset(env.line_available, 1, sizeof(env.line_available));

    PowerGridAppliedAction no_op = apply_agent_action(&env, 0.0f);
    CHECK(no_op.value == 0);
    CHECK(no_op.type == POWER_GRID_ACTION_NONE);
    CHECK(!no_op.switched);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 0);

    PowerGridAppliedAction line = apply_agent_action(&env, 1.0f);
    CHECK(line.type == POWER_GRID_ACTION_LINE);
    CHECK(line.switched);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 1);
    CHECK(env.episode.switches[POWER_GRID_ACTION_LINE] == 1);

    PowerGridAppliedAction invalid = apply_agent_action(&env, NAN);
    CHECK(invalid.type == POWER_GRID_ACTION_INVALID);
    CHECK(!invalid.switched);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 1);

    env.topology.line_closed[0] = 0;
    env.line_available[0] = 0;
    PowerGridAppliedAction blocked = apply_agent_action(&env, 1.0f);
    CHECK(blocked.type == POWER_GRID_ACTION_LINE);
    CHECK(!blocked.switched);
    CHECK(env.topology.line_closed[0] == 0);
    CHECK(env.episode.switches[0] == 1);
}

static void test_reward(void)
{
    CHECK_CLOSE(calculate_reward(POWER_GRID_INVALID_TOPOLOGY, 0.0, 0, 0),
                POWER_GRID_FAILURE_REWARD);
    CHECK_CLOSE(calculate_reward(POWER_GRID_SOLVE_OK, 0.0, 0, 1), 0.2);
    CHECK_CLOSE(calculate_reward(POWER_GRID_SOLVE_OK, 0.0, 1, 1), 0.199);
    CHECK_CLOSE(calculate_reward(POWER_GRID_SOLVE_OK, 0.04, 0, 0), -0.04);
}

static void test_episode_metrics(void)
{
    PowerGrid env = {0};
    power_grid_allocate(&env);
    env.rendering = 1;
    env.episode_step = 10;
    env.episode.safe_steps = 7;
    env.episode.no_op_actions = 8;
    env.episode.demand_served_steps = 10;
    env.episode.overloaded_line_steps = 3;
    env.episode.thermal_trips = 2;
    env.episode.peak_thermal_stress = 1.25;
    env.episode.peak_line_loading = 1.5;
    env.scheduled_random_event_period = 1;
    env.episode.random_events = 1;
    env.solution.status = POWER_GRID_SOLVE_OK;
    power_grid_finish_episode(&env);
    CHECK_CLOSE(env.log.perf, 0.7);
    CHECK_CLOSE(env.log.score, 0.8);
    CHECK_CLOSE(env.log.total_failure, 0.0);
    CHECK_CLOSE(env.log.connectivity_failure, 0.0);
    CHECK_CLOSE(env.log.solver_failure, 0.0);
    CHECK_CLOSE(env.log.demand_fulfilled, 10.0 / POWER_GRID_EPISODE_STEPS);
    CHECK_CLOSE(env.log.outage_completion, 1.0);
    CHECK_CLOSE(env.log.all_outages_survived, 1.0);
    CHECK_CLOSE(env.log.thermal_trips, 2.0);
    CHECK_CLOSE(env.log.thermal_trip_episode, 1.0);
    CHECK_CLOSE(env.log.peak_thermal_stress, 1.25);
    CHECK_CLOSE(env.log.peak_line_loading, 1.5);
    CHECK_CLOSE(env.log.overloaded_line_fraction,
                3.0 / (POWER_GRID_EPISODE_STEPS * POWER_GRID_NUM_BRANCHES));

    memset(&env.log, 0, sizeof(env.log));
    env.pending_reset = 0;
    env.solution.status = POWER_GRID_ISLANDED;
    power_grid_finish_episode(&env);
    CHECK_CLOSE(env.log.perf, 0.0);
    CHECK_CLOSE(env.log.score, 0.8);
    CHECK_CLOSE(env.log.total_failure, 1.0);
    CHECK_CLOSE(env.log.connectivity_failure, 1.0);
    CHECK_CLOSE(env.log.solver_failure, 0.0);

    memset(&env.log, 0, sizeof(env.log));
    env.pending_reset = 0;
    env.solution.status = POWER_GRID_SINGULAR;
    power_grid_finish_episode(&env);
    CHECK_CLOSE(env.log.total_failure, 1.0);
    CHECK_CLOSE(env.log.connectivity_failure, 0.0);
    CHECK_CLOSE(env.log.solver_failure, 1.0);

    memset(&env.log, 0, sizeof(env.log));
    env.pending_reset = 0;
    env.solution.status = POWER_GRID_INVALID_INPUT;
    power_grid_finish_episode(&env);
    CHECK_CLOSE(env.log.total_failure, 1.0);
    CHECK_CLOSE(env.log.connectivity_failure, 0.0);
    CHECK_CLOSE(env.log.solver_failure, 1.0);
    c_close(&env);
}

static void test_offline_scenario_cache(void)
{
    CHECK(POWER_GRID_OFFLINE_TRAIN_COUNT == 365);
    CHECK(POWER_GRID_OFFLINE_VALIDATION_COUNT == 366);
    CHECK(POWER_GRID_OFFLINE_SCENARIO_COUNT == 731);
    int weather_derated_periods = 0;
    int weather_uprated_periods = 0;
    int overloaded_periods = 0;
    int ac_solve_failures = 0;
    int ac_voltage_infeasible = 0;
    unsigned int previous_date = 0;
    PowerGridTopology topology;
    power_grid_topology_normal(&topology);
    for (int index = 0; index < POWER_GRID_OFFLINE_SCENARIO_COUNT; index++)
    {
        const PowerGridOfflineScenario *scenario = &POWER_GRID_OFFLINE_SCENARIOS[index];
        CHECK(scenario->date_yyyymmdd > previous_date);
        previous_date = scenario->date_yyyymmdd;
        int expected_year = index < POWER_GRID_OFFLINE_TRAIN_COUNT ?
                            POWER_GRID_OFFLINE_TRAIN_YEAR :
                            POWER_GRID_OFFLINE_VALIDATION_YEAR;
        CHECK((int)(scenario->date_yyyymmdd / 10000u) == expected_year);
        for (int period = 0; period < POWER_GRID_NUM_PERIODS; period++)
        {
            CHECK(isfinite(scenario->load_scale[period]));
            CHECK(scenario->load_scale[period] >= 0.65f);
            CHECK(scenario->load_scale[period] <= 1.35f);
            CHECK(isfinite(scenario->solar_mw[period]));
            CHECK(scenario->solar_mw[period] >= 0.0f);
            CHECK(scenario->solar_mw[period] <= 80.0f);
            CHECK(isfinite(scenario->wind_mw[period]));
            CHECK(scenario->wind_mw[period] >= 0.0f);
            CHECK(scenario->wind_mw[period] <= 70.0f);
            CHECK(isfinite(scenario->ambient_temperature_c[period]));
            CHECK(isfinite(scenario->wind_speed_mps[period]));
            CHECK(scenario->wind_speed_mps[period] >= 0.0f);
            CHECK(isfinite(scenario->solar_irradiance_wm2[period]));
            CHECK(scenario->solar_irradiance_wm2[period] >= 0.0f);

            PowerGridOperatingPoint point;
            PowerGridSolveResult solution;
            power_grid_operating_point_nominal(&point);
            for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
                point.load_mw[load] *= scenario->load_scale[period];
            point.generator_mw[2] = scenario->solar_mw[period];
            point.generator_mw[3] = scenario->wind_mw[period];
            CHECK(power_grid_solve(&topology, &point, &solution) == POWER_GRID_SOLVE_OK);
            for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            {
                if ((POWER_GRID_RANDOM_EVENT_LINE_MASK & (1u << line)) == 0)
                    continue;
                PowerGridTopology event_topology = topology;
                PowerGridSolveResult event_solution;
                event_topology.line_closed[line] = 0;
                CHECK(power_grid_solve(&event_topology, &point, &event_solution) ==
                      POWER_GRID_SOLVE_OK);
            }
            PowerGridACSolveResult ac_solution;
            PowerGridACStatus ac_status = power_grid_ac_solve(
                &topology, &point, &ac_solution);
            ac_solve_failures += ac_status != POWER_GRID_AC_OK;
            if (ac_status == POWER_GRID_AC_OK)
                ac_voltage_infeasible += ac_solution.voltage_violation_count > 0;

            double scale = power_grid_weather_rating_scale(
                scenario->ambient_temperature_c[period],
                scenario->wind_speed_mps[period],
                scenario->solar_irradiance_wm2[period]);
            weather_derated_periods += scale < 0.95;
            weather_uprated_periods += scale > 1.05;
            if (solution.max_rho / scale > 1.0)
                overloaded_periods++;

        }
    }
    CHECK(weather_derated_periods > 0);
    CHECK(weather_uprated_periods > 0);
    CHECK(overloaded_periods > 0);
    CHECK(ac_solve_failures == 0);
    CHECK(ac_voltage_infeasible == 0);

    for (int profile = 0; profile < POWER_GRID_NUM_PROFILES; profile++)
    {
        PowerGridOperatingPoint point;
        power_grid_operating_point_profile(&point, (PowerGridProfile)profile);
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        {
            if ((POWER_GRID_RANDOM_EVENT_LINE_MASK & (1u << line)) == 0)
                continue;
            PowerGridTopology event_topology = topology;
            PowerGridSolveResult event_solution;
            event_topology.line_closed[line] = 0;
            CHECK(power_grid_solve(&event_topology, &point, &event_solution) ==
                  POWER_GRID_SOLVE_OK);
        }
    }

    double cold_windy = power_grid_weather_rating_scale(-20.0, 8.0, 0.0);
    double hot_calm = power_grid_weather_rating_scale(35.0, 0.0, 1000.0);
    CHECK_CLOSE(power_grid_weather_rating_scale(3.6, 3.13, 7.0), 1.0);
    CHECK(cold_windy > hot_calm);
    CHECK(cold_windy <= 1.35);
    CHECK(hot_calm >= 0.90);
}

static void test_offline_episode_sampling(void)
{
    PowerGrid first = {
        .rng = 12345u, .offline_scenarios = 1, .offline_scenario_probability = 1.0,
    };
    PowerGrid second = {
        .rng = 12345u, .offline_scenarios = 1, .offline_scenario_probability = 1.0,
    };
    power_grid_allocate(&first);
    power_grid_allocate(&second);
    c_reset(&first);
    c_reset(&second);
    CHECK(first.offline_scenario != NULL);
    CHECK(second.offline_scenario != NULL);
    CHECK(first.offline_scenario->date_yyyymmdd == second.offline_scenario->date_yyyymmdd);
    CHECK(first.offline_scenario->date_yyyymmdd / 10000u ==
          POWER_GRID_OFFLINE_TRAIN_YEAR);
    CHECK_CLOSE(first.operating_point.load_mw[0], second.operating_point.load_mw[0]);
    CHECK_CLOSE(first.operating_point.generator_mw[2], second.operating_point.generator_mw[2]);
    CHECK_CLOSE(first.branch_rating_scale, second.branch_rating_scale);
    CHECK_CLOSE(first.observations[3], 1.0);
    CHECK_CLOSE(first.observations[4], 0.0);
    CHECK_CLOSE(first.observations[POWER_GRID_RATING_SCALE_OBS_OFFSET],
                first.branch_rating_scale);
    CHECK_CLOSE(first.observations[POWER_GRID_WEATHER_OBS_OFFSET],
                first.ambient_temperature_c / 50.0);
    CHECK_CLOSE(first.observations[POWER_GRID_WEATHER_OBS_OFFSET + 1],
                first.wind_speed_mps / 10.0);
    CHECK_CLOSE(first.observations[POWER_GRID_WEATHER_OBS_OFFSET + 2],
                first.solar_irradiance_wm2 / 1000.0);
    CHECK(first.solution.status == POWER_GRID_SOLVE_OK);

    PowerGridSolveResult static_rating_solution;
    CHECK(power_grid_solve(&first.topology, &first.operating_point,
                           &static_rating_solution) == POWER_GRID_SOLVE_OK);
    CHECK_CLOSE(first.solution.branch_rho[0],
                static_rating_solution.branch_rho[0] / first.branch_rating_scale);

    unsigned int first_date = first.offline_scenario->date_yyyymmdd;
    int found_another_day = 0;
    for (int episode = 0; episode < 16; episode++)
    {
        c_reset(&first);
        found_another_day |= first.offline_scenario->date_yyyymmdd != first_date;
    }
    CHECK(found_another_day);

    PowerGrid curriculum = {
        .rng = 12345u,
        .offline_scenarios = 1,
        .offline_scenario_probability = 0.0,
    };
    power_grid_allocate(&curriculum);
    c_reset(&curriculum);
    CHECK(curriculum.offline_scenario == NULL);
    CHECK(curriculum.synthetic_profile > POWER_GRID_PROFILE_P0_NOMINAL);
    for (int profile = 1; profile < POWER_GRID_NUM_PROFILES; profile++)
    {
        curriculum.synthetic_profile = (PowerGridProfile)profile;
        power_grid_set_operating_period(&curriculum, 0);
        PowerGridOperatingPoint previous = curriculum.operating_point;
        for (int period = 1; period < POWER_GRID_NUM_PERIODS; period++)
        {
            power_grid_set_operating_period(&curriculum, period);
            double largest_change = 0.0;
            for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
                largest_change = fmax(largest_change,
                    fabs(curriculum.operating_point.load_mw[load] - previous.load_mw[load]));
            for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
                largest_change = fmax(largest_change,
                    fabs(curriculum.operating_point.generator_mw[generator] -
                         previous.generator_mw[generator]));
            CHECK(largest_change <= 34.0 + 1e-6);
            previous = curriculum.operating_point;
        }
    }

    PowerGrid validation = {
        .rng = 12345u,
        .offline_scenarios = 1,
        .offline_scenario_validation = 1,
        .offline_scenario_probability = 0.0,
    };
    power_grid_allocate(&validation);
    c_reset(&validation);
    CHECK(validation.offline_scenario != NULL);
    CHECK(validation.offline_scenario->date_yyyymmdd / 10000u ==
          POWER_GRID_OFFLINE_VALIDATION_YEAR);
    CHECK(validation.solution.status == POWER_GRID_SOLVE_OK);

    PowerGrid ac = {
        .rng = 12345u, .offline_scenarios = 1, .offline_scenario_probability = 1.0,
        .ac_power_flow = 1,
    };
    power_grid_allocate(&ac);
    c_reset(&ac);
    CHECK(ac.solution.status == POWER_GRID_SOLVE_OK);
    CHECK(ac.ac_solution.status == POWER_GRID_AC_OK);
    CHECK_CLOSE(ac.solution.branch_rho[0], ac.ac_solution.branch_rho[0]);
    CHECK_CLOSE(power_grid_branch_rating(&ac, 0),
                power_grid_ac_branch_rating_mva(0) * ac.branch_rating_scale);
    CHECK_CLOSE(ac.observations[POWER_GRID_VOLTAGE_OBS_OFFSET],
                ac.ac_solution.node_voltage_pu[0]);
    CHECK_CLOSE(ac.observations[POWER_GRID_GENERATOR_Q_OBS_OFFSET],
                ac.ac_solution.generator_q_mvar[0] / POWER_GRID_BASE_MVA);

    c_close(&ac);
    c_close(&validation);
    c_close(&curriculum);
    c_close(&second);
    c_close(&first);
}

static void test_period_transition_failure_reward(void)
{
    PowerGrid env = {
        .rng = 12345u,
        .offline_scenarios = 1,
        .offline_scenario_probability = 1.0,
    };
    power_grid_allocate(&env);
    c_reset(&env);
    PowerGridOfflineScenario invalid = *env.offline_scenario;
    invalid.load_scale[1] = 0.65f;
    invalid.solar_mw[1] = 80.0f;
    invalid.wind_mw[1] = 70.0f;
    env.offline_scenario = &invalid;
    env.episode_step = POWER_GRID_STEPS_PER_PERIOD - 1;
    env.actions[0] = POWER_GRID_ACTION_NONE;
    c_step(&env);

    CHECK_CLOSE(env.rewards[0], POWER_GRID_FAILURE_REWARD);
    CHECK(env.terminals[0] == 1.0f);
    CHECK_CLOSE(env.log.episode_return, POWER_GRID_FAILURE_REWARD);
    CHECK_CLOSE(env.log.total_failure, 1.0f);
    c_close(&env);
}

static void test_topology_persists_across_periods(void)
{
    PowerGrid env = {
        .rng = 12345u,
        .offline_scenarios = 1,
        .offline_scenario_probability = 1.0,
    };
    power_grid_allocate(&env);
    c_reset(&env);
    env.topology.line_closed[16] = 0; /* 9-14 has a connected parallel path. */
    env.episode_step = POWER_GRID_STEPS_PER_PERIOD - 1;
    env.actions[0] = POWER_GRID_ACTION_NONE;
    c_step(&env);

    CHECK(env.terminals[0] == 0.0f);
    CHECK(env.current_period == 1);
    CHECK(env.topology.line_closed[16] == 0);
    c_close(&env);
}

static void test_solvable_random_event_lifecycle(void)
{
    for (int ac_power_flow = 0; ac_power_flow <= 1; ac_power_flow++)
    {
        PowerGrid env = {
            .rng = 12345u,
            .ac_power_flow = ac_power_flow,
            .random_events = 1,
            .random_event_probability = 1.0,
        };
        power_grid_allocate(&env);
        c_reset(&env);
        CHECK(env.scheduled_random_event_period > 0);
        CHECK(env.scheduled_random_event_period < POWER_GRID_NUM_PERIODS);
        CHECK(env.scheduled_random_event_line >= 0);
        CHECK(env.scheduled_random_event_line < POWER_GRID_NUM_BRANCHES);
        env.synthetic_profile = POWER_GRID_PROFILE_P0_NOMINAL;

        while (env.current_period < env.scheduled_random_event_period)
        {
            env.actions[0] = POWER_GRID_ACTION_NONE;
            c_step(&env);
            CHECK(env.terminals[0] == 0.0f);
        }
        int outage_line = env.scheduled_random_event_line;
        CHECK(env.active_random_event_line == outage_line);
        CHECK(env.episode.random_events == 1);
        CHECK(env.line_available[outage_line] == 0);
        CHECK(env.topology.line_closed[outage_line] == 0);
        CHECK_CLOSE(env.observations[POWER_GRID_LINE_OBS_OFFSET +
                                     outage_line * POWER_GRID_LINE_OBS_FEATURES + 3], 0.0);
        CHECK(env.solution.status == POWER_GRID_SOLVE_OK);

        env.actions[0] = NAN;
        c_step(&env);
        CHECK_CLOSE(env.log.solver_failure, 1.0);
        CHECK_CLOSE(env.log.event_failure, 0.0);

        c_close(&env);
    }
}

static void test_immediate_random_event_failure(void)
{
    PowerGrid env = {0};
    power_grid_allocate(&env);
    c_reset(&env);
    env.scheduled_random_event_period = 1;
    env.scheduled_random_event_line = 13; /* Radial 7-8 branch, excluded from normal events. */

    for (int step = 0; step < POWER_GRID_STEPS_PER_PERIOD; step++)
    {
        env.actions[0] = POWER_GRID_ACTION_NONE;
        c_step(&env);
    }

    CHECK_CLOSE(env.log.random_events, 1.0);
    CHECK_CLOSE(env.log.event_failure, 1.0);
    CHECK_CLOSE(env.log.connectivity_failure, 1.0);
    CHECK_CLOSE(env.log.solver_failure, 0.0);
    CHECK_CLOSE(env.log.total_failure, 1.0);
    c_close(&env);
}

static void test_random_event_applies_after_switching(void)
{
    PowerGrid env = {0};
    power_grid_allocate(&env);
    c_reset(&env);
    env.synthetic_profile = POWER_GRID_PROFILE_P0_NOMINAL;
    env.topology.line_closed[0] = 0;
    env.scheduled_random_event_period = 1;
    env.scheduled_random_event_line = 16;

    for (int step = 0; step < POWER_GRID_STEPS_PER_PERIOD; step++)
    {
        env.actions[0] = POWER_GRID_ACTION_NONE;
        c_step(&env);
    }

    CHECK(env.episode.random_events == 1);
    CHECK(env.active_random_event_line == 16);
    CHECK(env.line_available[16] == 0);
    CHECK(env.topology.line_closed[16] == 0);
    c_close(&env);
}

static void test_randomized_environment_lifecycle(void)
{
    const PowerGrid configurations[] = {
        {.rng = 7u},
        {.rng = 11u, .offline_scenarios = 1, .offline_scenario_probability = 1.0},
        {.rng = 13u, .offline_scenarios = 1, .offline_scenario_validation = 1},
        {.rng = 19u, .offline_scenarios = 1, .offline_scenario_probability = 1.0,
         .ac_power_flow = 1, .random_events = 1, .random_event_probability = 1.0},
    };
    for (unsigned int config = 0;
         config < sizeof(configurations) / sizeof(configurations[0]); config++)
    {
        PowerGrid env = configurations[config];
        power_grid_allocate(&env);
        c_reset(&env);
        unsigned int action_rng = 100u + config;
        for (int step = 0; step < 256; step++)
        {
            action_rng = action_rng * 1664525u + 1013904223u;
            env.actions[0] = (float)(action_rng % POWER_GRID_NUM_ACTIONS);
            c_step(&env);
            CHECK(isfinite(env.rewards[0]));
            CHECK(env.terminals[0] == 0.0f || env.terminals[0] == 1.0f);
            CHECK(env.solution.status >= POWER_GRID_SOLVE_OK);
            CHECK(env.solution.status <= POWER_GRID_INVALID_INPUT);
            for (int observation = 0; observation < POWER_GRID_OBS_SIZE; observation++)
                CHECK(isfinite(env.observations[observation]));
        }
        CHECK(env.log.n > 0.0f);
        c_close(&env);
    }
}

int main(void)
{
    test_action_application_and_bookkeeping();
    test_reward();
    test_episode_metrics();
    test_offline_scenario_cache();
    test_offline_episode_sampling();
    test_period_transition_failure_reward();
    test_topology_persists_across_periods();
    test_solvable_random_event_lifecycle();
    test_immediate_random_event_failure();
    test_random_event_applies_after_switching();
    test_randomized_environment_lifecycle();
    if (failures)
        return 1;
    puts("power-grid environment tests passed");
    return 0;
}
