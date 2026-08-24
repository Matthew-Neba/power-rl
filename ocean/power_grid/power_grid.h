#ifndef POWER_GRID_H
#define POWER_GRID_H

#include "power_grid_solver.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef POWER_GRID_NO_RENDER
#include "raylib.h"
#endif

#define POWER_GRID_OBS_SIZE 236
#define POWER_GRID_EPISODE_STEPS 72
#define POWER_GRID_STEPS_PER_PERIOD 6
#define POWER_GRID_NUM_PERIODS (POWER_GRID_EPISODE_STEPS / POWER_GRID_STEPS_PER_PERIOD)
#define POWER_GRID_MAX_RANDOM_OUTAGES 8
#define POWER_GRID_EMERGENCY_RECOVERY_STEPS 4

typedef struct {
    unsigned int date_yyyymmdd;
    float load_scale[POWER_GRID_NUM_PERIODS];
    float solar_mw[POWER_GRID_NUM_PERIODS];
    float wind_mw[POWER_GRID_NUM_PERIODS];
    float ambient_temperature_c[POWER_GRID_NUM_PERIODS];
    float wind_speed_mps[POWER_GRID_NUM_PERIODS];
    float solar_irradiance_wm2[POWER_GRID_NUM_PERIODS];
} PowerGridOfflineScenario;

#include "power_grid_scenarios_data.h"

/* Steady-state IEEE 738 ampacity normalized to median 2019 ERA5 conditions.
 * IEEE-14 has no physical routes, so all lines use the published 26/7 Drake
 * ACSR example parameters and perpendicular wind. Measured irradiance is
 * treated as effective incident flux. Synthetic ratings remain capped at
 * 0.90x-1.35x until real conductor and route data are available. */
static inline double power_grid_weather_rating_scale(
    double ambient_temperature_c, double wind_speed_mps, double solar_irradiance_wm2)
{
    const double conductor_temperature_c = 100.0;
    const double conductor_diameter_m = 0.02814;
    const double conductor_emissivity = 0.8;
    const double conductor_solar_absorptivity = 0.8;
    const double elevation_m = 645.0;
    const double resistance_25_ohm_m = 7.283e-5;
    const double resistance_75_ohm_m = 8.688e-5;
    const double resistance_100_ohm_m = resistance_25_ohm_m +
        (resistance_75_ohm_m - resistance_25_ohm_m) * 1.5;
    const double ambient[2] = {ambient_temperature_c, 3.6};
    const double wind[2] = {wind_speed_mps, 3.13};
    const double irradiance[2] = {solar_irradiance_wm2, 7.0};
    double ampacity[2] = {0.0, 0.0};

    for (int condition = 0; condition < 2; condition++)
    {
        double temperature_difference = fmax(0.0,
            conductor_temperature_c - ambient[condition]);
        double film_temperature_c = 0.5 *
            (conductor_temperature_c + ambient[condition]);
        double air_viscosity = 1.458e-6 * pow(film_temperature_c + 273.0, 1.5) /
                               (film_temperature_c + 383.4);
        double air_density = (1.293 - 1.525e-4 * elevation_m +
                              6.379e-9 * elevation_m * elevation_m) /
                             (1.0 + 0.00367 * film_temperature_c);
        double air_conductivity = 2.424e-2 + 7.477e-5 * film_temperature_c -
                                  4.407e-9 * film_temperature_c * film_temperature_c;
        double reynolds = conductor_diameter_m * air_density * wind[condition] /
                          air_viscosity;
        double natural_convection = 3.645 * sqrt(air_density) *
            pow(conductor_diameter_m, 0.75) * pow(temperature_difference, 1.25);
        /* K_angle is 1.0 for wind perpendicular to the conductor axis. */
        double forced_convection_1 = (1.01 + 1.35 * pow(reynolds, 0.52)) *
                                     air_conductivity * temperature_difference;
        double forced_convection_2 = 0.754 * pow(reynolds, 0.6) *
                                     air_conductivity * temperature_difference;
        double convection = fmax(natural_convection,
                                 fmax(forced_convection_1, forced_convection_2));
        double radiation = 17.8 * conductor_diameter_m * conductor_emissivity *
            (pow((conductor_temperature_c + 273.0) / 100.0, 4.0) -
             pow((ambient[condition] + 273.0) / 100.0, 4.0));
        double solar = conductor_solar_absorptivity * irradiance[condition] *
                       conductor_diameter_m;
        ampacity[condition] = sqrt(fmax(1.0, convection + radiation - solar) /
                                   resistance_100_ohm_m);
    }
    return fmin(1.35, fmax(0.90, ampacity[0] / ampacity[1]));
}

/* Reward configuration. */
#define POWER_GRID_DEFAULT_FAILURE_REWARD (-1.0f)
#define POWER_GRID_DEFAULT_SAFE_STEP_REWARD 1.0f
#define POWER_GRID_DEFAULT_RECOVERY_REWARD 0.50f
#define POWER_GRID_DEFAULT_SWITCH_PENALTY 0.002f
#define POWER_GRID_DEFAULT_SECURE_SWITCH_PENALTY 0.10f
#define POWER_GRID_DEFAULT_CONGESTION_COST_WEIGHT 0.01f
#define POWER_GRID_DEFAULT_CONGESTION_PROGRESS_WEIGHT 1.0f
#define POWER_GRID_DEFAULT_UNSERVED_LOAD_COST_WEIGHT 5.0f
#define POWER_GRID_VALID_REWARD_MARGIN 0.05f

#define POWER_GRID_LINE_OBS_FEATURES 5
#define POWER_GRID_LINE_OBS_OFFSET 0
#define POWER_GRID_TERMINAL_OBS_OFFSET \
    (POWER_GRID_NUM_BRANCHES * POWER_GRID_LINE_OBS_FEATURES)
#define POWER_GRID_COUPLER_OBS_OFFSET \
    (POWER_GRID_TERMINAL_OBS_OFFSET + POWER_GRID_NUM_TERMINALS)
#define POWER_GRID_INJECTION_OBS_OFFSET \
    (POWER_GRID_COUPLER_OBS_OFFSET + POWER_GRID_NUM_SUBSTATIONS)
#define POWER_GRID_RATING_SCALE_OBS_OFFSET \
    (POWER_GRID_INJECTION_OBS_OFFSET + POWER_GRID_NUM_SUBSTATIONS)
#define POWER_GRID_WEATHER_OBS_OFFSET (POWER_GRID_RATING_SCALE_OBS_OFFSET + 1)
#define POWER_GRID_VOLTAGE_OBS_OFFSET (POWER_GRID_WEATHER_OBS_OFFSET + 3)
#define POWER_GRID_GENERATOR_Q_OBS_OFFSET \
    (POWER_GRID_VOLTAGE_OBS_OFFSET + POWER_GRID_NUM_NODES)
#define POWER_GRID_EMERGENCY_OBS_OFFSET \
    (POWER_GRID_GENERATOR_Q_OBS_OFFSET + POWER_GRID_NUM_GENERATORS)
_Static_assert(POWER_GRID_EMERGENCY_OBS_OFFSET + POWER_GRID_NUM_LOADS +
                   POWER_GRID_NUM_GENERATORS - 1 ==
                   POWER_GRID_OBS_SIZE,
               "power-grid observation layout must total 236 floats");
_Static_assert(POWER_GRID_EPISODE_STEPS % POWER_GRID_STEPS_PER_PERIOD == 0,
               "power-grid periods must divide the episode evenly");
_Static_assert(POWER_GRID_ACTION_NONE == 0 && POWER_GRID_ACTION_LINE == 1 &&
                   POWER_GRID_ACTION_TERMINAL == 2 && POWER_GRID_ACTION_COUPLER == 3 &&
                   POWER_GRID_ACTION_LOAD_SHED == 4 &&
                   POWER_GRID_ACTION_GENERATOR_TRIP == 5,
               "power-grid action types must index episode switch counters");

typedef struct
{
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float total_failure;
    float connectivity_failure;
    float solver_failure;
    float event_failure;
    float total_switches;
    float line_switches;
    float busbar_switches;
    float coupler_switches;
    float load_shed_actions;
    float generator_trip_actions;
    float overload_free_steps;
    float random_events;
    float demand_fulfilled;
    float outage_completion;
    float all_outages_survived;
    float thermal_trips;
    float thermal_trip_episode;
    float peak_thermal_stress;
    float peak_line_loading;
    float overloaded_line_fraction;
    float n;
} Log;

typedef struct
{
    int switches[6]; /* total followed by each physical control type */
    int no_op_actions;
    int safe_steps;
    int thermal_trips;
    int random_events;
    int recoveries_rewarded;
    double served_load_fraction_sum;
    int overloaded_line_steps;
    double peak_thermal_stress;
    double peak_line_loading;
} PowerGridEpisodeStats;

typedef struct
{
    int value;
    PowerGridActionType type;
    int switched;
    int electrical_change;
    int observation_only_terminal;
} PowerGridAppliedAction;

typedef struct
{
    Log log;
    int rendering;
    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int num_agents;
    unsigned int rng;
    int owns_buffers;

    PowerGridTopology topology;
    PowerGridOperatingPoint operating_point;
    PowerGridSolveResult solution;
    PowerGridACSolveResult ac_solution;
    const PowerGridOfflineScenario *offline_scenario;
    PowerGridProfile synthetic_profile;
    int episode_step;
    int current_period;
    int operating_period;
    int operating_period_offset;
    PowerGridEpisodeStats episode;
    unsigned char line_available[POWER_GRID_NUM_BRANCHES];
    double line_thermal_stress[POWER_GRID_NUM_BRANCHES];
    int scheduled_random_event_count;
    int scheduled_random_event_period[POWER_GRID_MAX_RANDOM_OUTAGES];
    int scheduled_random_event_line[POWER_GRID_MAX_RANDOM_OUTAGES];
    int active_random_event_line;
    int last_failure_was_event;
    int emergency_recovery_steps;
    float episode_return;
    int pending_reset;
    int last_action;
    PowerGridActionType last_action_type;
    int ac_power_flow;
    int offline_scenarios;
    int offline_scenario_validation;
    double offline_scenario_probability;
    int random_events;
    double random_event_probability;
    int random_outage_count;
    int random_outage_count_min;
    int random_outages_at_reset;
    double reset_outage_probability;
    int randomize_reset_operating_period;
    int initial_outage_requires_overload;
    int initial_outage_requires_one_step_recovery;
    int end_episode_on_recovery;
    int single_episode_evaluation;
    /* Zero means unbounded operation. Training uses the default 72-step limit. */
    int max_episode_steps;
    float failure_reward;
    float safe_step_reward;
    float recovery_reward;
    float switch_penalty;
    float secure_switch_penalty;
    float congestion_cost_weight;
    float congestion_progress_weight;
    float unserved_load_cost_weight;
    double branch_rating_scale;
    double ambient_temperature_c;
    double wind_speed_mps;
    double solar_irradiance_wm2;
} PowerGrid;

void c_reset(PowerGrid *env);

static int power_grid_generator_index_at_bus(int bus)
{
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
        if (POWER_GRID_GENERATOR_BUSES[generator] == bus)
            return generator;
    return -1;
}

/* Preserve IEEE-14's online conventional participation while treating the
 * Alberta-derived wind and solar traces as exogenous. Dispatchable headroom
 * attempts to cover 91% of demand; the slack balances any remaining shortfall
 * once the much smaller IEEE-14 generator fleet reaches its limits. */
static void power_grid_redispatch(PowerGridOperatingPoint *point)
{
    int solar = power_grid_generator_index_at_bus(POWER_GRID_SOLAR_GENERATOR_BUS);
    int wind = power_grid_generator_index_at_bus(POWER_GRID_WIND_GENERATOR_BUS);
    double total_load = 0.0, current = 0.0;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        total_load += point->load_mw[load];
    for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
        current += point->generator_mw[generator];
    double required = 0.91 * total_load;
    double difference = required - current;
    if (difference > 0.0)
    {
        for (int pass = 0; pass < 3 && difference > 1e-8; pass++)
        {
            double headroom = 0.0;
            for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
                if (generator != solar && generator != wind &&
                    POWER_GRID_GENERATOR_P_NOMINAL[generator] > 0.0)
                    headroom += fmax(0.0, POWER_GRID_GENERATOR_P_MAX[generator] -
                                           point->generator_mw[generator]);
            if (headroom <= 0.0)
                break;
            double applied = fmin(difference, headroom);
            for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
            {
                if (generator == solar || generator == wind ||
                    POWER_GRID_GENERATOR_P_NOMINAL[generator] <= 0.0)
                    continue;
                double available = fmax(0.0, POWER_GRID_GENERATOR_P_MAX[generator] -
                                              point->generator_mw[generator]);
                point->generator_mw[generator] += applied * available / headroom;
            }
            difference -= applied;
        }
    }
    else if (difference < 0.0)
    {
        double dispatchable = 0.0;
        for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
            if (generator != solar && generator != wind)
                dispatchable += point->generator_mw[generator];
        double reduction = fmin(-difference, dispatchable);
        if (dispatchable > 0.0)
            for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
                if (generator != solar && generator != wind)
                    point->generator_mw[generator] *= 1.0 - reduction / dispatchable;
    }
}

static void power_grid_set_operating_period(PowerGrid *env, int period)
{
    env->operating_period = period;
    env->branch_rating_scale = 1.0;
    env->ambient_temperature_c = 3.6;
    env->wind_speed_mps = 3.13;
    env->solar_irradiance_wm2 = 7.0;
    if (env->offline_scenarios && env->offline_scenario != NULL)
    {
        const PowerGridOfflineScenario *scenario = env->offline_scenario;
        power_grid_operating_point_nominal(&env->operating_point);
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            env->operating_point.load_mw[load] *= scenario->load_scale[period];
        for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
            env->operating_point.generator_mw[generator] *= scenario->load_scale[period];
        int solar = power_grid_generator_index_at_bus(POWER_GRID_SOLAR_GENERATOR_BUS);
        int wind = power_grid_generator_index_at_bus(POWER_GRID_WIND_GENERATOR_BUS);
        env->operating_point.generator_mw[solar] = scenario->solar_mw[period] *
            POWER_GRID_SOLAR_NAMEPLATE_MW /
            POWER_GRID_SCENARIO_SOLAR_SOURCE_NAMEPLATE_MW;
        env->operating_point.generator_mw[wind] = scenario->wind_mw[period] *
            POWER_GRID_WIND_NAMEPLATE_MW /
            POWER_GRID_SCENARIO_WIND_SOURCE_NAMEPLATE_MW;
        power_grid_redispatch(&env->operating_point);
        env->ambient_temperature_c = scenario->ambient_temperature_c[period];
        env->wind_speed_mps = scenario->wind_speed_mps[period];
        env->solar_irradiance_wm2 = scenario->solar_irradiance_wm2[period];
        env->branch_rating_scale = power_grid_weather_rating_scale(
            env->ambient_temperature_c, env->wind_speed_mps,
            env->solar_irradiance_wm2);
    }
    else
    {
        static const double stress_scale[POWER_GRID_NUM_PERIODS] = {
            0.0, 0.20, 0.40, 0.60, 0.80, 1.00,
            1.0, 0.80, 0.60, 0.40, 0.20, 0.00,
        };
        PowerGridOperatingPoint target;
        power_grid_operating_point_nominal(&env->operating_point);
        power_grid_operating_point_profile(&target, env->synthetic_profile);
        double scale = stress_scale[period];
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        {
            env->operating_point.load_mw[load] +=
                scale * (target.load_mw[load] - env->operating_point.load_mw[load]);
        }
        for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
        {
            env->operating_point.generator_mw[generator] +=
                scale * (target.generator_mw[generator] -
                         env->operating_point.generator_mw[generator]);
        }
        power_grid_redispatch(&env->operating_point);
    }
}

static void power_grid_apply_dynamic_ratings(PowerGrid *env)
{
    if (env->offline_scenario == NULL || env->solution.status != POWER_GRID_SOLVE_OK)
        return;

    env->solution.max_rho = 0.0;
    env->solution.congestion_cost = 0.0;
    env->solution.overloaded_branch_count = 0;
    if (env->ac_power_flow)
    {
        env->ac_solution.max_rho = 0.0;
        env->ac_solution.congestion_cost = 0.0;
    }
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        double base_rating = env->ac_power_flow ? power_grid_ac_branch_rating_mva(line) :
                                                 POWER_GRID_BRANCHES[line].thermal_limit_mw;
        double rating = base_rating * env->branch_rating_scale;
        double loading = fabs(env->solution.branch_flow_mw[line]);
        if (env->ac_power_flow)
            loading = fmax(env->ac_solution.branch_from_mva[line],
                           env->ac_solution.branch_to_mva[line]);
        double rho = loading / rating;
        env->solution.branch_rho[line] = rho;
        if (env->ac_power_flow)
            env->ac_solution.branch_rho[line] = rho;
        if (rho > env->solution.max_rho)
            env->solution.max_rho = rho;
        if (env->ac_power_flow && rho > env->ac_solution.max_rho)
            env->ac_solution.max_rho = rho;
        if (rho > 1.0)
        {
            double overload = rho - 1.0;
            env->solution.congestion_cost += overload * overload;
            env->solution.overloaded_branch_count++;
            if (env->ac_power_flow)
                env->ac_solution.congestion_cost += overload * overload;
        }
    }
}

static unsigned int next_random(PowerGrid *env)
{
    env->rng = env->rng * 1664525u + 1013904223u;
    return env->rng ^ (env->rng >> 16);
}

static PowerGridSolveStatus power_grid_solve_environment(PowerGrid *env)
{
    if (!env->ac_power_flow)
    {
        return power_grid_solve_scaled(
            &env->topology, &env->operating_point, &env->solution,
            env->branch_rating_scale);
    }
    power_grid_ac_solve(&env->topology, &env->operating_point, &env->ac_solution);
    power_grid_ac_to_compatible(&env->ac_solution, &env->solution);
    power_grid_apply_dynamic_ratings(env);
    return env->solution.status;
}

static double power_grid_branch_rating(const PowerGrid *env, int line)
{
    double base_rating = env->ac_power_flow ? power_grid_ac_branch_rating_mva(line) :
                                             POWER_GRID_BRANCHES[line].thermal_limit_mw;
    return base_rating * env->branch_rating_scale;
}

/* AC-mode inverse-time protection. Stress is an intentionally simple
 * thermal proxy: 200% trips in one step, 150% in four, and 120% in about 25.
 * A tripped line is locked out for the rest of the episode. */
static int power_grid_update_ac_protection(PowerGrid *env)
{
    if (!env->ac_power_flow || env->solution.status != POWER_GRID_SOLVE_OK)
        return 0;
    int new_trips = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        if (!env->line_available[line])
            continue;
        if (!env->topology.line_closed[line])
        {
            env->line_thermal_stress[line] = power_grid_ac_thermal_step(
                env->line_thermal_stress[line], 0.0);
            continue;
        }
        env->line_thermal_stress[line] = power_grid_ac_thermal_step(
            env->line_thermal_stress[line], env->solution.branch_rho[line]);
        env->episode.peak_thermal_stress = fmax(
            env->episode.peak_thermal_stress, env->line_thermal_stress[line]);
        if (env->line_thermal_stress[line] >= POWER_GRID_THERMAL_TRIP_THRESHOLD)
        {
            env->line_available[line] = 0;
            env->topology.line_closed[line] = 0;
            new_trips++;
        }
    }
    env->episode.thermal_trips += new_trips;
    if (new_trips)
        power_grid_solve_environment(env);
    return new_trips;
}

static void power_grid_compute_observations(PowerGrid *env)
{
    int index = POWER_GRID_LINE_OBS_OFFSET;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        float signed_loading = (float)(env->solution.branch_flow_mw[line] /
                                       power_grid_branch_rating(env, line));
        float rho = (float)env->solution.branch_rho[line];
        /* Natural per-unit values retain overload severity: 1.0 is exactly the line limit. */
        env->observations[index++] = signed_loading;
        env->observations[index++] = rho;
        env->observations[index++] = env->topology.line_closed[line] ? 1.0f : 0.0f;
        env->observations[index++] = env->line_available[line] ? 1.0f : 0.0f;
        env->observations[index++] = (float)env->line_thermal_stress[line];
    }
    index = POWER_GRID_TERMINAL_OBS_OFFSET;
    /* A bit is the non-redundant one-hot encoding for each two-state busbar category. */
    for (int terminal = 0; terminal < POWER_GRID_NUM_TERMINALS; terminal++)
    {
        env->observations[index++] = env->topology.terminal_busbar[terminal] ? 1.0f : 0.0f;
    }
    index = POWER_GRID_COUPLER_OBS_OFFSET;
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        env->observations[index++] = env->topology.coupler_closed[bus] ? 1.0f : 0.0f;
    }
    index = POWER_GRID_INJECTION_OBS_OFFSET;
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        /* Normalize MW by the IEEE case's 100 MVA system base without clipping. */
        env->observations[index++] = (float)(env->solution.substation_injection_mw[bus] /
                                             POWER_GRID_BASE_MVA);
    }
    env->observations[POWER_GRID_RATING_SCALE_OBS_OFFSET] =
        (float)env->branch_rating_scale;
    env->observations[POWER_GRID_WEATHER_OBS_OFFSET] =
        (float)(env->ambient_temperature_c / 50.0);
    env->observations[POWER_GRID_WEATHER_OBS_OFFSET + 1] =
        (float)(env->wind_speed_mps / 10.0);
    env->observations[POWER_GRID_WEATHER_OBS_OFFSET + 2] =
        (float)(env->solar_irradiance_wm2 / 1000.0);
    unsigned char active_node[POWER_GRID_NUM_NODES] = {0};
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        if (!env->topology.line_closed[line])
            continue;
        active_node[power_grid_terminal_node(&env->topology,
            POWER_GRID_LINE_TERMINAL(line, 0), POWER_GRID_BRANCHES[line].from_bus)] = 1;
        active_node[power_grid_terminal_node(&env->topology,
            POWER_GRID_LINE_TERMINAL(line, 1), POWER_GRID_BRANCHES[line].to_bus)] = 1;
    }
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
        if (env->topology.generator_connected[generator])
            active_node[power_grid_terminal_node(&env->topology,
                POWER_GRID_GENERATOR_TERMINAL(generator),
                POWER_GRID_GENERATOR_BUSES[generator])] = 1;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        if (env->topology.load_connected[load])
            active_node[power_grid_terminal_node(&env->topology,
                POWER_GRID_LOAD_TERMINAL(load), POWER_GRID_LOAD_BUSES[load])] = 1;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
        env->observations[POWER_GRID_VOLTAGE_OBS_OFFSET + node] = env->ac_power_flow ?
            (float)env->ac_solution.node_voltage_pu[node] : (float)active_node[node];
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
        env->observations[POWER_GRID_GENERATOR_Q_OBS_OFFSET + generator] =
            env->ac_power_flow ?
                (float)(env->ac_solution.generator_q_mvar[generator] / POWER_GRID_BASE_MVA) :
                0.0f;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        env->observations[POWER_GRID_EMERGENCY_OBS_OFFSET + load] =
            env->topology.load_connected[load] ? 1.0f : 0.0f;
    for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
        env->observations[POWER_GRID_EMERGENCY_OBS_OFFSET + POWER_GRID_NUM_LOADS +
                          generator - 1] =
            env->topology.generator_connected[generator] ? 1.0f : 0.0f;
}

/* Agent training-step pipeline. Keep reward and metric calculations pure so their
 * signatures expose every input that can affect learning or evaluation. */
static PowerGridAppliedAction apply_agent_action(PowerGrid *env, float raw_action)
{
    PowerGridAppliedAction action = {
        .value = isfinite(raw_action) ? (int)raw_action : -1,
        .electrical_change = 0,
        .observation_only_terminal = -1,
    };
    PowerGridTopology previous = env->topology;
    action.type = power_grid_apply_action(&env->topology, action.value);
    action.switched = action.type > POWER_GRID_ACTION_NONE;
    env->last_action = action.value;
    env->last_action_type = action.type;

    if (action.type == POWER_GRID_ACTION_LINE)
    {
        int line = action.value - POWER_GRID_LINE_ACTION_OFFSET;
        if (!env->line_available[line])
            env->topology.line_closed[line] = 0;
        action.switched = env->topology.line_closed[line] !=
                          previous.line_closed[line];
        action.electrical_change = action.switched;
    }
    else if (action.type == POWER_GRID_ACTION_TERMINAL)
    {
        int terminal = action.value - POWER_GRID_TERMINAL_ACTION_OFFSET;
        int substation;
        if (terminal < 2 * POWER_GRID_NUM_BRANCHES)
        {
            int line = terminal / 2;
            substation = terminal % 2 == 0 ? POWER_GRID_BRANCHES[line].from_bus :
                                              POWER_GRID_BRANCHES[line].to_bus;
        }
        else if (terminal < 2 * POWER_GRID_NUM_BRANCHES + POWER_GRID_NUM_GENERATORS)
        {
            substation = POWER_GRID_GENERATOR_BUSES[
                terminal - 2 * POWER_GRID_NUM_BRANCHES];
        }
        else
        {
            substation = POWER_GRID_LOAD_BUSES[
                terminal - 2 * POWER_GRID_NUM_BRANCHES - POWER_GRID_NUM_GENERATORS];
        }
        /* With a closed coupler both busbars are one electrical node. The
         * terminal bit remains observable and matters if the coupler opens,
         * but it cannot change this step's power-flow solution. */
        action.electrical_change = !env->topology.coupler_closed[substation];
        /* An open line endpoint has no incident branch, so moving that
         * endpoint between open busbars cannot alter injections or flows. */
        if (action.electrical_change && terminal < 2 * POWER_GRID_NUM_BRANCHES &&
            !env->topology.line_closed[terminal / 2])
            action.electrical_change = 0;
        if (!action.electrical_change)
            action.observation_only_terminal = terminal;
    }
    else if (action.type == POWER_GRID_ACTION_COUPLER)
    {
        action.electrical_change = action.switched;
    }
    else if (action.type == POWER_GRID_ACTION_LOAD_SHED)
    {
        int load = action.value - POWER_GRID_LOAD_SHED_ACTION_OFFSET;
        action.switched = previous.load_connected[load] !=
                          env->topology.load_connected[load];
        action.electrical_change = action.switched;
    }
    else if (action.type == POWER_GRID_ACTION_GENERATOR_TRIP)
    {
        int generator = 1 + action.value - POWER_GRID_GENERATOR_TRIP_ACTION_OFFSET;
        action.switched = previous.generator_connected[generator] !=
                          env->topology.generator_connected[generator];
        action.electrical_change = action.switched;
    }
    env->episode.no_op_actions += action.type == POWER_GRID_ACTION_NONE;
    env->episode.switches[0] += action.switched;
    if (action.switched)
        env->episode.switches[action.type]++;
    return action;
}

static double power_grid_served_load_fraction(const PowerGrid *env)
{
    double total = 0.0, served = 0.0;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
    {
        total += env->operating_point.load_mw[load];
        if (env->topology.load_connected[load])
            served += env->operating_point.load_mw[load];
    }
    return total > 0.0 ? served / total : 1.0;
}

static float calculate_reward(const PowerGrid *env,
                              PowerGridSolveStatus solve_status,
                              double constraint_cost, double congestion_progress,
                              int switched, int electrical_change,
                              int previously_safe, int safe,
                              double served_load_fraction)
{
    if (solve_status != POWER_GRID_SOLVE_OK)
        return env->failure_reward;
    /* The main term is exactly the evaluation target: one unit for a secure
     * valid step and zero for an overloaded valid step. Failure remains worse
     * than continuing to seek recovery. */
    /* The trainer clips rewards to [-1, 1]. Preserve the intended ordering
     * after that clip for the passive choice: remaining in a solved overloaded
     * state must be strictly better than an action that destroys connectivity.
     * Bound only the state cost so the signed progress potential remains
     * exactly reversible for topology changes. */
    double state_reward = env->safe_step_reward * safe -
        env->unserved_load_cost_weight * (1.0 - served_load_fraction);
    double maximum_constraint_cost = fmax(
        0.0, state_reward - env->failure_reward -
                 POWER_GRID_VALID_REWARD_MARGIN);
    double bounded_constraint_cost = fmin(constraint_cost,
                                          maximum_constraint_cost);
    return (float)(state_reward - bounded_constraint_cost +
                   env->congestion_progress_weight * congestion_progress -
                   ((!electrical_change || previously_safe) ?
                        env->secure_switch_penalty : env->switch_penalty) * switched);
}

static void power_grid_finish_episode(PowerGrid *env)
{
    int failed = env->solution.status != POWER_GRID_SOLVE_OK;
    float steps = env->episode_step > 0 ? (float)env->episode_step : 1.0f;
    float perf = failed ? 0.0f : (float)env->episode.safe_steps / steps;
    float score = (float)env->episode.no_op_actions / steps;
    /* perf measures grid security; score measures how often switching was avoided. */
    env->log.perf += perf;
    env->log.overload_free_steps += (float)env->episode.safe_steps;
    env->log.random_events += (float)env->episode.random_events;
    env->log.demand_fulfilled +=
        (float)(env->episode.served_load_fraction_sum / steps);
    if (env->scheduled_random_event_count > 0)
    {
        env->log.outage_completion += (float)env->episode.random_events /
                                      env->scheduled_random_event_count;
        env->log.all_outages_survived +=
            !failed && env->episode.random_events == env->scheduled_random_event_count;
    }
    env->log.thermal_trips += (float)env->episode.thermal_trips;
    env->log.thermal_trip_episode += env->episode.thermal_trips > 0;
    env->log.peak_thermal_stress += (float)env->episode.peak_thermal_stress;
    env->log.peak_line_loading += (float)env->episode.peak_line_loading;
    env->log.overloaded_line_fraction +=
        (float)env->episode.overloaded_line_steps /
        (POWER_GRID_EPISODE_STEPS * POWER_GRID_NUM_BRANCHES);
    env->log.score += score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_step;
    int connectivity_failure = env->solution.status == POWER_GRID_INVALID_TOPOLOGY ||
                               env->solution.status == POWER_GRID_DISCONNECTED_LOAD ||
                               env->solution.status == POWER_GRID_DISCONNECTED_GENERATOR ||
                               env->solution.status == POWER_GRID_ISLANDED;
    int solver_failure = env->solution.status == POWER_GRID_SINGULAR ||
                         env->solution.status == POWER_GRID_NONFINITE ||
                         env->solution.status == POWER_GRID_INVALID_INPUT;
    env->log.connectivity_failure += connectivity_failure;
    env->log.solver_failure += solver_failure;
    env->log.total_failure += connectivity_failure + solver_failure;
    env->log.total_switches += (float)env->episode.switches[0];
    env->log.line_switches += (float)env->episode.switches[POWER_GRID_ACTION_LINE];
    env->log.busbar_switches += (float)env->episode.switches[POWER_GRID_ACTION_TERMINAL];
    env->log.coupler_switches += (float)env->episode.switches[POWER_GRID_ACTION_COUPLER];
    env->log.load_shed_actions +=
        (float)env->episode.switches[POWER_GRID_ACTION_LOAD_SHED];
    env->log.generator_trip_actions +=
        (float)env->episode.switches[POWER_GRID_ACTION_GENERATOR_TRIP];
    env->log.n += 1.0f;
    if (env->rendering)
    {
        env->pending_reset = 1;
        power_grid_compute_observations(env);
    }
    else if (!env->single_episode_evaluation)
    {
        c_reset(env);
    }
}

static uint32_t power_grid_one_step_recovery_lines
    [POWER_GRID_OFFLINE_TRAIN_COUNT][POWER_GRID_NUM_PERIODS];
static int power_grid_one_step_recovery_cache_ready;

static int power_grid_has_one_step_recovery(
    const PowerGrid *env, const PowerGridTopology *outage_topology)
{
    if (power_grid_one_step_recovery_cache_ready &&
        env->offline_scenario >= POWER_GRID_OFFLINE_SCENARIOS &&
        env->offline_scenario < POWER_GRID_OFFLINE_SCENARIOS +
                                POWER_GRID_OFFLINE_TRAIN_COUNT)
    {
        int outage_line = -1;
        int outage_count = 0;
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            if (!outage_topology->line_closed[line])
            {
                outage_line = line;
                outage_count++;
            }
        if (outage_count == 1)
        {
            ptrdiff_t scenario =
                env->offline_scenario - POWER_GRID_OFFLINE_SCENARIOS;
            return (power_grid_one_step_recovery_lines
                        [scenario][env->operating_period] &
                    (UINT32_C(1) << outage_line)) != 0;
        }
    }
    for (int action = 1; action < POWER_GRID_NUM_ACTIONS; action++)
    {
        PowerGridTopology topology = *outage_topology;
        power_grid_apply_action(&topology, action);
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            if (!env->line_available[line])
                topology.line_closed[line] = 0;
        if (memcmp(&topology, outage_topology, sizeof(topology)) == 0)
            continue;
        PowerGridSolveResult solution;
        if (power_grid_solve_scaled(&topology, &env->operating_point, &solution,
                                    env->branch_rating_scale) == POWER_GRID_SOLVE_OK &&
            solution.max_rho <= 1.0)
            return 1;
    }
    return 0;
}

static inline void power_grid_prepare_one_step_recovery_cache(void)
{
    if (power_grid_one_step_recovery_cache_ready)
        return;
    PowerGrid probe = {.offline_scenarios = 1};
    memset(probe.line_available, 1, sizeof(probe.line_available));
    for (int scenario = 0; scenario < POWER_GRID_OFFLINE_TRAIN_COUNT; scenario++)
    {
        probe.offline_scenario = &POWER_GRID_OFFLINE_SCENARIOS[scenario];
        for (int period = 0; period < POWER_GRID_NUM_PERIODS; period++)
        {
            power_grid_set_operating_period(&probe, period);
            for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            {
                if (!power_grid_random_event_eligible(line))
                    continue;
                PowerGridTopology topology;
                PowerGridSolveResult solution;
                power_grid_topology_normal(&topology);
                topology.line_closed[line] = 0;
                PowerGridSolveStatus outage_status = power_grid_solve_scaled(
                    &topology, &probe.operating_point, &solution,
                    probe.branch_rating_scale);
                if (outage_status == POWER_GRID_SOLVE_OK &&
                    solution.max_rho <= 1.0)
                    continue;
                probe.line_available[line] = 0;
                int recoverable = 0;
                for (int action = 1; action < POWER_GRID_NUM_ACTIONS; action++)
                {
                    PowerGridTopology recovered = topology;
                    power_grid_apply_action(&recovered, action);
                    recovered.line_closed[line] = 0;
                    if (memcmp(&recovered, &topology, sizeof(topology)) == 0)
                        continue;
                    if (power_grid_solve_scaled(
                            &recovered, &probe.operating_point, &solution,
                            probe.branch_rating_scale) == POWER_GRID_SOLVE_OK &&
                        solution.max_rho <= 1.0)
                    {
                        recoverable = 1;
                        break;
                    }
                }
                if (recoverable)
                    power_grid_one_step_recovery_lines[scenario][period] |=
                        UINT32_C(1) << line;
                probe.line_available[line] = 1;
            }
        }
    }
    power_grid_one_step_recovery_cache_ready = 1;
}

static int power_grid_sample_one_step_outage(PowerGrid *env)
{
    if (!power_grid_one_step_recovery_cache_ready ||
        env->offline_scenario < POWER_GRID_OFFLINE_SCENARIOS ||
        env->offline_scenario >= POWER_GRID_OFFLINE_SCENARIOS +
                                 POWER_GRID_OFFLINE_TRAIN_COUNT)
        return -1;
    ptrdiff_t scenario = env->offline_scenario - POWER_GRID_OFFLINE_SCENARIOS;
    uint32_t recoverable = power_grid_one_step_recovery_lines
        [scenario][env->operating_period];
    int line_count = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        line_count += (recoverable >> line) & 1u;
    if (line_count == 0)
        return -1;
    int selected = (int)(next_random(env) % (unsigned int)line_count);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        if ((recoverable >> line) & 1u)
            if (selected-- == 0)
                return line;
    return -1;
}

void c_reset(PowerGrid *env)
{
    power_grid_topology_normal(&env->topology);
    env->episode_step = 0;
    memset(&env->episode, 0, sizeof(env->episode));
    memset(env->line_available, 1, sizeof(env->line_available));
    memset(env->line_thermal_stress, 0, sizeof(env->line_thermal_stress));
    env->pending_reset = 0;
    env->last_action = POWER_GRID_ACTION_NONE;
    env->last_action_type = POWER_GRID_ACTION_NONE;
    env->episode_return = 0.0f;
    env->active_random_event_line = -1;
    env->last_failure_was_event = 0;
    env->emergency_recovery_steps = 0;
    int reset_outages_now = env->random_outages_at_reset;
    if (reset_outages_now && env->reset_outage_probability < 1.0)
    {
        double reset_sample = (double)(next_random(env) >> 8) / 16777216.0;
        reset_outages_now = reset_sample < env->reset_outage_probability;
    }
    env->offline_scenario = NULL;
    env->branch_rating_scale = 1.0;
    if (env->offline_scenarios)
    {
        /* One draw chooses the curriculum branch and, when selected, a complete
         * correlated cached day. No file or network access occurs in the hot path. */
        unsigned int sample = next_random(env);
        double unit_sample = (double)(sample >> 8) / 16777216.0;
        if (env->offline_scenario_validation ||
            unit_sample < env->offline_scenario_probability)
        {
            unsigned int offset = env->offline_scenario_validation ?
                                  POWER_GRID_OFFLINE_VALIDATION_OFFSET : 0u;
            unsigned int count = env->offline_scenario_validation ?
                                 POWER_GRID_OFFLINE_VALIDATION_COUNT :
                                 POWER_GRID_OFFLINE_TRAIN_COUNT;
            env->offline_scenario = &POWER_GRID_OFFLINE_SCENARIOS[
                offset + sample % count];
        }
    }
    env->synthetic_profile = POWER_GRID_PROFILE_P0_NOMINAL;
    if (env->offline_scenario == NULL)
    {
        /* One stress family per episode produces a coherent daily ramp. */
        env->synthetic_profile = (PowerGridProfile)(
            1 + next_random(env) % (POWER_GRID_NUM_PROFILES - 1));
    }
    env->current_period = 0;
    int reset_operating_period = env->random_outages_at_reset &&
        env->randomize_reset_operating_period ?
        (int)(next_random(env) % POWER_GRID_NUM_PERIODS) : 0;
    env->operating_period_offset = reset_operating_period;
    power_grid_set_operating_period(env, reset_operating_period);
    env->scheduled_random_event_count = 0;
    for (int event = 0; event < POWER_GRID_MAX_RANDOM_OUTAGES; event++)
    {
        env->scheduled_random_event_period[event] = -1;
        env->scheduled_random_event_line[event] = -1;
    }
    if (env->random_events)
    {
        double unit_sample = (double)(next_random(env) >> 8) / 16777216.0;
        if (unit_sample < env->random_event_probability)
        {
            int requested = env->random_outage_count > 0 ? env->random_outage_count : 1;
            int minimum = env->random_outage_count_min > 0 ?
                          env->random_outage_count_min : requested;
            if (minimum < requested)
                requested = minimum + (int)(next_random(env) %
                    (unsigned int)(requested - minimum + 1));
            env->scheduled_random_event_count =
                requested < POWER_GRID_MAX_RANDOM_OUTAGES ? requested :
                                                            POWER_GRID_MAX_RANDOM_OUTAGES;
            if (env->random_outages_at_reset && !reset_outages_now)
                env->scheduled_random_event_count = 0;
            int schedule_accepted;
            int schedule_attempts = 0;
            do
            {
                for (int event = 0; event < env->scheduled_random_event_count; event++)
                {
                    int period = 0;
                    if (!reset_outages_now)
                    {
                        int unique_period;
                        do
                        {
                            period = 1 +
                                (int)(next_random(env) % (POWER_GRID_NUM_PERIODS - 1));
                            unique_period = 1;
                            for (int prior = 0; prior < event; prior++)
                                unique_period &=
                                    env->scheduled_random_event_period[prior] != period;
                        }
                        while (!unique_period);
                    }
                    env->scheduled_random_event_period[event] = period;

                    int line = -1;
                    int unique_line = 0;
                    if (reset_outages_now &&
                        env->initial_outage_requires_one_step_recovery &&
                        env->scheduled_random_event_count == 1)
                    {
                        line = power_grid_sample_one_step_outage(env);
                        unique_line = line >= 0;
                    }
                    if (!unique_line)
                    {
                        do
                        {
                            line = (int)(next_random(env) % POWER_GRID_NUM_BRANCHES);
                            unique_line = power_grid_random_event_eligible(line);
                            for (int prior = 0; prior < event; prior++)
                                unique_line &=
                                    env->scheduled_random_event_line[prior] != line;
                        }
                        while (!unique_line);
                    }
                    env->scheduled_random_event_line[event] = line;
                }

                schedule_accepted = 1;
                if (reset_outages_now &&
                    env->initial_outage_requires_overload)
                {
                    PowerGridTopology candidate;
                    PowerGridSolveResult candidate_solution;
                    power_grid_topology_normal(&candidate);
                    for (int event = 0; event < env->scheduled_random_event_count; event++)
                        candidate.line_closed[env->scheduled_random_event_line[event]] = 0;
                    schedule_accepted = power_grid_solve_scaled(
                        &candidate, &env->operating_point, &candidate_solution,
                        env->branch_rating_scale) == POWER_GRID_SOLVE_OK &&
                        candidate_solution.max_rho > 1.0;
                    if (schedule_accepted &&
                        env->initial_outage_requires_one_step_recovery)
                    {
                        unsigned char availability[POWER_GRID_NUM_BRANCHES];
                        memcpy(availability, env->line_available,
                               sizeof(availability));
                        for (int event = 0;
                             event < env->scheduled_random_event_count; event++)
                            env->line_available[
                                env->scheduled_random_event_line[event]] = 0;
                        schedule_accepted =
                            power_grid_has_one_step_recovery(env, &candidate);
                        memcpy(env->line_available, availability,
                               sizeof(availability));
                    }
                }
                schedule_attempts++;
            }
            while (!schedule_accepted && schedule_attempts < 256);
        }
    }
    for (int event = 0; event < env->scheduled_random_event_count; event++)
    {
        if (!reset_outages_now || env->scheduled_random_event_period[event] != 0)
            continue;
        int line = env->scheduled_random_event_line[event];
        env->line_available[line] = 0;
        env->topology.line_closed[line] = 0;
        env->active_random_event_line = line;
        env->episode.random_events++;
    }
    power_grid_solve_environment(env);
    if (env->solution.status != POWER_GRID_SOLVE_OK && reset_outages_now)
        env->emergency_recovery_steps = POWER_GRID_EMERGENCY_RECOVERY_STEPS;
    power_grid_compute_observations(env);
}

void c_step(PowerGrid *env)
{
    if (env->single_episode_evaluation && env->log.n > 0.0f)
    {
        env->rewards[0] = 0.0f;
        env->terminals[0] = 1.0f;
        return;
    }
    if (env->rendering && env->pending_reset)
    {
        c_reset(env);
        return;
    }
    int previously_safe = env->solution.status == POWER_GRID_SOLVE_OK &&
                          env->solution.max_rho <= 1.0;
    int failure_pending_from_event = env->last_failure_was_event &&
                                     env->solution.status != POWER_GRID_SOLVE_OK;
    double previous_overload = env->solution.status == POWER_GRID_SOLVE_OK ?
        fmax(env->solution.max_rho - 1.0, 0.0) : 0.0;
    PowerGridAppliedAction action = apply_agent_action(env, env->actions[0]);
    env->terminals[0] = 0.0f;
    /* In DC mode, a no-op before a period transition leaves every value in the
     * observation vector unchanged. Track whether a later state mutation
     * actually requires rebuilding the complete observation. */
    int observations_dirty = action.switched || env->ac_power_flow;
    int observation_only_terminal = action.observation_only_terminal;

    PowerGridSolveStatus status;
    if (action.type == POWER_GRID_ACTION_INVALID)
    {
        status = env->solution.status = POWER_GRID_INVALID_INPUT;
    }
    else if (!action.electrical_change && !env->ac_power_flow)
    {
        /* In DC mode, a no-op cannot change topology, injections, or ratings;
         * the previous solved state is exactly the current solved state. */
        status = env->solution.status;
    }
    else
    {
        status = power_grid_solve_environment(env);
        if (status == POWER_GRID_SOLVE_OK && env->ac_power_flow)
        {
            for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            {
                double rho = env->solution.branch_rho[line];
                env->episode.peak_line_loading =
                    fmax(env->episode.peak_line_loading, rho);
                env->episode.overloaded_line_steps += rho > 1.0;
            }
            power_grid_update_ac_protection(env);
        }
        status = env->solution.status;
    }

    env->episode_step++;

    if (status != POWER_GRID_SOLVE_OK && env->emergency_recovery_steps > 0)
    {
        env->emergency_recovery_steps--;
        env->rewards[0] = calculate_reward(
            env, status, 0.0, 0.0, action.switched, action.electrical_change,
            previously_safe, 0, power_grid_served_load_fraction(env));
        env->episode_return += env->rewards[0];
        power_grid_compute_observations(env);
        return;
    }
    if (status == POWER_GRID_SOLVE_OK)
        env->emergency_recovery_steps = 0;

    int metric_safe = 0;
    int reward_safe = 0;
    double served_load_fraction = power_grid_served_load_fraction(env);
    double constraint_cost = 0.0;
    double congestion_progress = 0.0;
    if (status == POWER_GRID_SOLVE_OK)
    {
        /* DC may reuse an unchanged solution, but loading diagnostics describe
         * every served step and must still be accumulated on cache/no-op paths. */
        if (!env->ac_power_flow)
        {
            env->episode.peak_line_loading = fmax(
                env->episode.peak_line_loading, env->solution.max_rho);
            env->episode.overloaded_line_steps +=
                env->solution.overloaded_branch_count;
        }
        /* Scheduled generator-P violations are exogenous data faults, not agent actions. */
        reward_safe = env->solution.max_rho <= 1.0;
        metric_safe = reward_safe &&
                      (!env->ac_power_flow || env->ac_solution.voltage_violation_count == 0);
        constraint_cost = env->congestion_cost_weight * env->solution.congestion_cost;
        double overload = fmax(env->solution.max_rho - 1.0, 0.0);
        /* Signed potential shaping: undoing and repeating a topology change
         * has zero net progress, so oscillations cannot farm reward. */
        congestion_progress = previous_overload - overload;
        env->episode.safe_steps += metric_safe;
        env->episode.served_load_fraction_sum += served_load_fraction;
    }

    env->rewards[0] = calculate_reward(
        env, status, constraint_cost, congestion_progress,
        action.switched, action.electrical_change, previously_safe, reward_safe,
        served_load_fraction);
    if (action.electrical_change && env->episode.random_events > 0 &&
        env->episode.recoveries_rewarded < env->episode.random_events &&
        !previously_safe && reward_safe)
    {
        env->rewards[0] += env->recovery_reward;
        env->episode.recoveries_rewarded++;
    }
    int recovery_completed = env->end_episode_on_recovery &&
                             env->episode.random_events > 0 &&
                             env->episode.random_events ==
                                 env->scheduled_random_event_count &&
                             reward_safe;
    env->terminals[0] = status != POWER_GRID_SOLVE_OK || recovery_completed ||
                        (env->max_episode_steps > 0 &&
                         env->episode_step >= env->max_episode_steps);

    if (env->terminals[0])
    {
        env->log.event_failure += failure_pending_from_event;
        env->last_failure_was_event = failure_pending_from_event;
        env->episode_return += env->rewards[0];
        power_grid_finish_episode(env);
        return;
    }
    env->last_failure_was_event = 0;

    int random_event_applied = 0;
    int next_period = (env->episode_step / POWER_GRID_STEPS_PER_PERIOD) %
                      POWER_GRID_NUM_PERIODS;
    if (next_period != env->current_period)
    {
        observations_dirty = 1;
        observation_only_terminal = -1;
        env->current_period = next_period;
        power_grid_set_operating_period(
            env, (env->operating_period_offset + next_period) %
                     POWER_GRID_NUM_PERIODS);
        for (int event = 0; event < env->scheduled_random_event_count; event++)
        {
            if (next_period != env->scheduled_random_event_period[event])
                continue;
            int line = env->scheduled_random_event_line[event];
            env->line_available[line] = 0;
            env->topology.line_closed[line] = 0;
            env->active_random_event_line = line;
            env->episode.random_events++;
            random_event_applied = 1;
        }
        status = power_grid_solve_environment(env);
    }
    if (status != POWER_GRID_SOLVE_OK)
    {
        /* A new outage can itself disconnect equipment. Expose that state and
         * give the policy one step to use an explicit emergency action. */
        if (random_event_applied)
        {
            env->last_failure_was_event = 1;
            env->emergency_recovery_steps = POWER_GRID_EMERGENCY_RECOVERY_STEPS;
            env->rewards[0] = calculate_reward(
                env, status, 0.0, 0.0, 0, 0, 0, 0,
                power_grid_served_load_fraction(env));
            env->episode_return += env->rewards[0];
            power_grid_compute_observations(env);
            return;
        }
        /* New injections can expose an infeasible state without an outage. */
        env->log.event_failure += random_event_applied;
        env->last_failure_was_event = random_event_applied;
        env->rewards[0] = calculate_reward(
            env, status, 0.0, 0.0, 0, 0, 0, 0,
            power_grid_served_load_fraction(env));
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        power_grid_finish_episode(env);
        return;
    }
    env->episode_return += env->rewards[0];
    if (observations_dirty)
    {
        if (observation_only_terminal >= 0)
            env->observations[POWER_GRID_TERMINAL_OBS_OFFSET +
                              observation_only_terminal] =
                env->topology.terminal_busbar[observation_only_terminal] ? 1.0f : 0.0f;
        else
            power_grid_compute_observations(env);
    }
}

void power_grid_allocate(PowerGrid *env)
{
    env->num_agents = 1;
    env->observations = calloc(POWER_GRID_OBS_SIZE, sizeof(float));
    env->actions = calloc(1, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(float));
    env->owns_buffers = 1;
    env->max_episode_steps = POWER_GRID_EPISODE_STEPS;
    env->failure_reward = POWER_GRID_DEFAULT_FAILURE_REWARD;
    env->safe_step_reward = POWER_GRID_DEFAULT_SAFE_STEP_REWARD;
    env->recovery_reward = POWER_GRID_DEFAULT_RECOVERY_REWARD;
    env->switch_penalty = POWER_GRID_DEFAULT_SWITCH_PENALTY;
    env->secure_switch_penalty = POWER_GRID_DEFAULT_SECURE_SWITCH_PENALTY;
    env->congestion_cost_weight = POWER_GRID_DEFAULT_CONGESTION_COST_WEIGHT;
    env->congestion_progress_weight = POWER_GRID_DEFAULT_CONGESTION_PROGRESS_WEIGHT;
    env->unserved_load_cost_weight = POWER_GRID_DEFAULT_UNSERVED_LOAD_COST_WEIGHT;
    if (env->reset_outage_probability == 0.0)
        env->reset_outage_probability = 1.0;
    env->branch_rating_scale = 1.0;
}

/* Standalone and evaluation renderer. Training builds define POWER_GRID_NO_RENDER. */
#ifndef POWER_GRID_NO_RENDER
#define POWER_GRID_RENDER_WIDTH 1800
#define POWER_GRID_RENDER_HEIGHT 1000
#define POWER_GRID_MAP_WIDTH 1400
#define POWER_GRID_RENDER_FPS 1

static const Color POWER_GRID_BG = {9, 14, 24, 255};
static const Color POWER_GRID_PANEL = {20, 29, 43, 255};
static const Color POWER_GRID_PANEL_EDGE = {54, 70, 91, 255};
static const Color POWER_GRID_SAFE = {55, 205, 145, 255};
static const Color POWER_GRID_WARN = {250, 185, 55, 255};
static const Color POWER_GRID_OVERLOAD = {245, 68, 75, 255};
static const Color POWER_GRID_OPEN = {78, 87, 105, 255};
static const Color POWER_GRID_BB1 = {75, 185, 255, 255};
static const Color POWER_GRID_BB2 = {190, 105, 255, 255};
static const Color POWER_GRID_TEXT = {225, 232, 242, 255};
static const Color POWER_GRID_MUTED = {145, 158, 178, 255};

static const Vector2 POWER_GRID_STATION_POSITIONS[POWER_GRID_NUM_SUBSTATIONS] = {
    {100, 285}, {315, 205}, {570, 135}, {575, 355}, {305, 415},
    {455, 620}, {785, 350}, {1020, 205}, {1020, 485}, {1250, 570},
    {870, 635}, {530, 835}, {870, 845}, {1235, 810},
};

static const Vector2 POWER_GRID_LABEL_OFFSETS[POWER_GRID_NUM_BRANCHES] = {
    {30, 25}, {-34, 18}, {30, 0}, {-5, 30}, {-30, 22},
    {24, 0}, {0, 24}, {-24, -27}, {25, 10}, {-28, 5},
    {-34, -22}, {-32, 14}, {25, 16}, {0, -25}, {24, 10},
    {18, -23}, {30, 18}, {-27, 16}, {0, -23}, {20, 11},
};

static Color power_grid_line_color(const PowerGrid *env, int line)
{
    if (!env->line_available[line])
        return POWER_GRID_OVERLOAD;
    if (!env->topology.line_closed[line])
        return POWER_GRID_OPEN;
    double rho = env->solution.branch_rho[line];
    if (rho > 1.0)
        return POWER_GRID_OVERLOAD;
    if (rho > 0.8)
        return POWER_GRID_WARN;
    return POWER_GRID_SAFE;
}

static Vector2 power_grid_branch_endpoint(Vector2 station, Vector2 other, int busbar)
{
    float dx = other.x - station.x;
    float x = station.x;
    if (fabsf(dx) > 12.0f)
        x += dx > 0.0f ? 54.0f : -54.0f;
    return (Vector2){x, station.y + (busbar ? 8.0f : -8.0f)};
}

static void power_grid_draw_dashed_line(Vector2 from, Vector2 to, float dash, float gap,
                                        float thickness, Color color)
{
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float length = sqrtf(dx * dx + dy * dy);
    if (length <= 0.0f)
        return;
    dx /= length;
    dy /= length;
    for (float start = 0.0f; start < length; start += dash + gap)
    {
        float end = fminf(start + dash, length);
        DrawLineEx((Vector2){from.x + dx * start, from.y + dy * start},
                   (Vector2){from.x + dx * end, from.y + dy * end}, thickness, color);
    }
}

static void power_grid_draw_flow_arrow(Vector2 from, Vector2 to, double flow, float phase,
                                       Color color)
{
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float length = sqrtf(dx * dx + dy * dy);
    if (length < 1.0f || fabs(flow) < 0.05)
        return;

    dx /= length;
    dy /= length;
    if (flow < 0.0)
    {
        Vector2 tmp = from;
        from = to;
        to = tmp;
        dx = (to.x - from.x) / length;
        dy = (to.y - from.y) / length;
    }

    float progress = fmodf((float)(GetTime() * 0.40) + phase, 1.0f);
    float head = fminf(7.0f, length * 0.12f);
    float tail = fminf(6.0f, length * 0.10f);
    for (int marker = 0; marker < 2; marker++)
    {
        float fraction = marker == 0 ? 0.10f + 0.22f * progress :
                                       0.68f + 0.22f * progress;
        float distance = fminf(fmaxf(length * fraction, tail), length - head);
        Vector2 tip = {from.x + dx * (distance + head),
                       from.y + dy * (distance + head)};
        Vector2 left = {from.x + dx * (distance - tail) - dy * tail,
                        from.y + dy * (distance - tail) + dx * tail};
        Vector2 right = {from.x + dx * (distance - tail) + dy * tail,
                         from.y + dy * (distance - tail) - dx * tail};
        DrawLineEx(left, tip, 5.0f, Fade(BLACK, 0.75f));
        DrawLineEx(right, tip, 5.0f, Fade(BLACK, 0.75f));
        DrawLineEx(left, tip, 2.5f, color);
        DrawLineEx(right, tip, 2.5f, color);
    }
}

static void power_grid_draw_label(Vector2 position, const char *text, Color border)
{
    int width = MeasureText(text, 16) + 8;
    Rectangle box = {position.x - width * 0.5f, position.y - 12.0f,
                     (float)width, 24.0f};
    DrawRectangleRounded(box, 0.3f, 4, Fade(POWER_GRID_BG, 0.94f));
    DrawRectangleRoundedLinesEx(box, 0.3f, 4, 1.0f, Fade(border, 0.75f));
    DrawText(text, (int)(box.x + 4), (int)(box.y + 4), 16, POWER_GRID_TEXT);
}

static void power_grid_draw_branches(const PowerGrid *env)
{
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        const PowerGridBranch *branch = &POWER_GRID_BRANCHES[line];
        Vector2 from_station = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to_station = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        int from_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 0)];
        int to_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 1)];
        Vector2 from = power_grid_branch_endpoint(from_station, to_station, from_bar);
        Vector2 to = power_grid_branch_endpoint(to_station, from_station, to_bar);
        Color color = power_grid_line_color(env, line);
        DrawLineEx(from, to, 7.0f, Fade(BLACK, 0.65f));
        if (env->topology.line_closed[line])
        {
            float width = 2.5f + 2.5f * fminf((float)env->solution.branch_rho[line], 1.2f);
            DrawLineEx(from, to, width, color);
        }
        else
        {
            power_grid_draw_dashed_line(from, to, 9.0f, 7.0f, 2.5f, POWER_GRID_OPEN);
            DrawCircleLines((int)from.x, (int)from.y, 5.0f, POWER_GRID_OPEN);
            DrawCircleLines((int)to.x, (int)to.y, 5.0f, POWER_GRID_OPEN);
        }
    }

    /* Labels are drawn over the conductors so crossings cannot hide them. */
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        const PowerGridBranch *branch = &POWER_GRID_BRANCHES[line];
        Vector2 from = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        Vector2 label = {(from.x + to.x) * 0.5f + POWER_GRID_LABEL_OFFSETS[line].x,
                         (from.y + to.y) * 0.5f + POWER_GRID_LABEL_OFFSETS[line].y};
        char text[80];
        if (env->topology.line_closed[line])
        {
            if (env->ac_power_flow)
            {
                double active_power_mw = env->ac_solution.branch_from_p_mw[line];
                snprintf(text, sizeof(text), "%s%s  %+.0fMW %.0f/%.0fMVA %.0f%%",
                         branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                         active_power_mw,
                         fmax(env->ac_solution.branch_from_mva[line],
                              env->ac_solution.branch_to_mva[line]),
                         power_grid_branch_rating(env, line),
                         100.0 * env->solution.branch_rho[line]);
            }
            else
            {
                snprintf(text, sizeof(text), "%s%s  %+.0f/%.0fMW %.0f%%",
                         branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                         env->solution.branch_flow_mw[line],
                         power_grid_branch_rating(env, line),
                         100.0 * env->solution.branch_rho[line]);
            }
        }
        else
        {
            snprintf(text, sizeof(text), "%s  %s", POWER_GRID_BRANCH_NAMES[line],
                     !env->line_available[line] ? "OUTAGE" : "OPEN");
        }
        power_grid_draw_label(label, text, power_grid_line_color(env, line));
    }

    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        if (!env->topology.line_closed[line])
            continue;
        const PowerGridBranch *branch = &POWER_GRID_BRANCHES[line];
        Vector2 from_station = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to_station = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        int from_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 0)];
        int to_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 1)];
        Vector2 from = power_grid_branch_endpoint(from_station, to_station, from_bar);
        Vector2 to = power_grid_branch_endpoint(to_station, from_station, to_bar);
        power_grid_draw_flow_arrow(from, to, env->solution.branch_flow_mw[line],
                                   0.13f * line, RAYWHITE);
    }
}

static void power_grid_draw_generator(const PowerGrid *env, int generator)
{
    int bus = POWER_GRID_GENERATOR_BUSES[generator];
    int terminal = POWER_GRID_GENERATOR_TERMINAL(generator);
    int bar = env->topology.terminal_busbar[terminal];
    Vector2 station = POWER_GRID_STATION_POSITIONS[bus];
    Vector2 busbar = {station.x - 19.0f, station.y + (bar ? 8.0f : -8.0f)};
    Vector2 symbol = {station.x - 19.0f, station.y - 49.0f};
    int connected = env->topology.generator_connected[generator];
    Color terminal_color = connected ? (bar ? POWER_GRID_BB2 : POWER_GRID_BB1) :
                                       POWER_GRID_OPEN;
    DrawLineEx(busbar, (Vector2){symbol.x, symbol.y + 13.0f}, 2.0f, terminal_color);
    DrawCircleV(symbol, 17.0f, connected ? (Color){34, 78, 100, 255} :
                                             (Color){68, 72, 80, 255});
    DrawCircleLines((int)symbol.x, (int)symbol.y, 17.0f, POWER_GRID_BB1);
    DrawText("G", (int)symbol.x - 7, (int)symbol.y - 11, 21, RAYWHITE);
    double output = generator == 0 ? env->solution.slack_generation_mw :
                                     env->operating_point.generator_mw[generator];
    const char *label = !connected ? "TRIPPED" : env->ac_power_flow ?
        TextFormat("%.0f MW  %+.0f MVAr", output,
                   env->ac_solution.generator_q_mvar[generator]) :
        TextFormat("%.1f MW", output);
    int font_size = env->ac_power_flow ? 16 : 18;
    int label_x = (int)symbol.x + 22;
    int label_width = MeasureText(label, font_size);
    if (label_x + label_width > POWER_GRID_MAP_WIDTH - 8)
        label_x = (int)symbol.x - 22 - label_width;
    DrawText(label, label_x, (int)symbol.y - 10, font_size, POWER_GRID_TEXT);
}

static void power_grid_draw_load(const PowerGrid *env, int load)
{
    int bus = POWER_GRID_LOAD_BUSES[load];
    int terminal = POWER_GRID_LOAD_TERMINAL(load);
    int bar = env->topology.terminal_busbar[terminal];
    Vector2 station = POWER_GRID_STATION_POSITIONS[bus];
    Vector2 busbar = {station.x + 19.0f, station.y + (bar ? 8.0f : -8.0f)};
    Vector2 symbol = {station.x + 19.0f, station.y + 49.0f};
    int connected = env->topology.load_connected[load];
    DrawLineEx(busbar, (Vector2){symbol.x, symbol.y - 12.0f}, 2.0f,
               connected ? (bar ? POWER_GRID_BB2 : POWER_GRID_BB1) :
                           POWER_GRID_OPEN);
    Rectangle load_box = {symbol.x - 15.0f, symbol.y - 13.0f, 30.0f, 26.0f};
    DrawRectangleRounded(load_box, 0.25f, 4,
                         connected ? (Color){178, 77, 50, 255} :
                                     (Color){68, 72, 80, 255});
    DrawRectangleRoundedLinesEx(load_box, 0.25f, 4, 1.0f,
                                (Color){255, 154, 105, 255});
    DrawText("L", (int)symbol.x - 6, (int)symbol.y - 10, 20, RAYWHITE);
    const char *label = !connected ? "SHED" : env->ac_power_flow ?
        TextFormat("%.1f MW  %+.1f MVAr", env->operating_point.load_mw[load],
                   power_grid_ac_load_q_mvar(&env->operating_point, load)) :
        TextFormat("%.1f MW", env->operating_point.load_mw[load]);
    int font_size = env->ac_power_flow ? 16 : 18;
    int label_x = (int)symbol.x + 20;
    int label_width = MeasureText(label, font_size);
    if (label_x + label_width > POWER_GRID_MAP_WIDTH - 8)
        label_x = (int)symbol.x - 20 - label_width;
    DrawText(label, label_x, (int)symbol.y - 10, font_size, POWER_GRID_TEXT);
}

static void power_grid_draw_stations(const PowerGrid *env)
{
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        Vector2 position = POWER_GRID_STATION_POSITIONS[bus];
        Rectangle panel = {position.x - 60.0f, position.y - 29.0f, 120.0f, 58.0f};
        DrawRectangleRounded(panel, 0.18f, 4, POWER_GRID_PANEL);
        DrawRectangleRoundedLinesEx(panel, 0.18f, 4, 1.0f, POWER_GRID_PANEL_EDGE);
        DrawText(TextFormat("SUB %d", bus + 1), (int)position.x - 54,
                 (int)position.y - 27, 14, POWER_GRID_MUTED);
        DrawLineEx((Vector2){position.x - 51, position.y - 8},
                   (Vector2){position.x + 51, position.y - 8}, 6.0f, POWER_GRID_BB1);
        DrawLineEx((Vector2){position.x - 51, position.y + 8},
                   (Vector2){position.x + 51, position.y + 8}, 6.0f, POWER_GRID_BB2);
        if (env->topology.coupler_closed[bus])
        {
            DrawLineEx((Vector2){position.x, position.y - 8},
                       (Vector2){position.x, position.y + 8}, 5.0f, POWER_GRID_SAFE);
        }
        else
        {
            DrawCircleLines((int)position.x, (int)position.y - 5, 3.0f, POWER_GRID_OPEN);
            DrawCircleLines((int)position.x, (int)position.y + 5, 3.0f, POWER_GRID_OPEN);
        }
        DrawText("1", (int)position.x - 58, (int)position.y - 16, 12, POWER_GRID_BB1);
        DrawText("2", (int)position.x - 58, (int)position.y + 1, 12, POWER_GRID_BB2);
        double injection = env->solution.substation_injection_mw[bus];
        DrawText(TextFormat("NET %+.0f", injection), (int)position.x + 2,
                 (int)position.y - 27, 14,
                 injection >= 0.0 ? POWER_GRID_SAFE : POWER_GRID_MUTED);
    }
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
    {
        power_grid_draw_generator(env, generator);
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        power_grid_draw_load(env, load);
}

static void power_grid_draw_legend_item(int x, int y, Color color, const char *text)
{
    DrawCircle(x, y + 7, 5.0f, color);
    DrawText(text, x + 11, y, 15, POWER_GRID_TEXT);
}

static void power_grid_action_summary(const PowerGrid *env, char *buffer, size_t size)
{
    int action = env->last_action;
    if (action == POWER_GRID_ACTION_NONE)
    {
        snprintf(buffer, size, "DO NOTHING");
    }
    else if (env->last_action_type == POWER_GRID_ACTION_LINE)
    {
        int line = action - POWER_GRID_LINE_ACTION_OFFSET;
        snprintf(buffer, size, "LINE %s  %s", POWER_GRID_BRANCH_NAMES[line],
                 env->topology.line_closed[line] ? "CLOSED" : "OPEN");
    }
    else if (env->last_action_type == POWER_GRID_ACTION_TERMINAL)
    {
        int terminal = action - POWER_GRID_TERMINAL_ACTION_OFFSET;
        int busbar = env->topology.terminal_busbar[terminal] + 1;
        if (terminal < 2 * POWER_GRID_NUM_BRANCHES)
        {
            snprintf(buffer, size, "LINE %s %s -> BUSBAR %d",
                     POWER_GRID_BRANCH_NAMES[terminal / 2], terminal % 2 ? "TO" : "FROM", busbar);
        }
        else if (terminal < 2 * POWER_GRID_NUM_BRANCHES + POWER_GRID_NUM_GENERATORS)
        {
            int generator = terminal - 2 * POWER_GRID_NUM_BRANCHES;
            snprintf(buffer, size, "GENERATOR %d -> BUSBAR %d", generator, busbar);
        }
        else
        {
            snprintf(buffer, size, "LOAD %d -> BUSBAR %d",
                     terminal - 2 * POWER_GRID_NUM_BRANCHES - POWER_GRID_NUM_GENERATORS,
                     busbar);
        }
    }
    else if (env->last_action_type == POWER_GRID_ACTION_COUPLER)
    {
        int substation = action - POWER_GRID_COUPLER_ACTION_OFFSET;
        snprintf(buffer, size, "COUPLER SUB %d  %s", substation + 1,
                 env->topology.coupler_closed[substation] ? "CLOSED" : "OPEN");
    }
    else if (env->last_action_type == POWER_GRID_ACTION_LOAD_SHED)
    {
        int load = action - POWER_GRID_LOAD_SHED_ACTION_OFFSET;
        snprintf(buffer, size, "EMERGENCY SHED LOAD %d AT SUB %d", load,
                 POWER_GRID_LOAD_BUSES[load] + 1);
    }
    else if (env->last_action_type == POWER_GRID_ACTION_GENERATOR_TRIP)
    {
        int generator = 1 + action - POWER_GRID_GENERATOR_TRIP_ACTION_OFFSET;
        snprintf(buffer, size, "EMERGENCY TRIP GENERATOR %d AT SUB %d", generator,
                 POWER_GRID_GENERATOR_BUSES[generator] + 1);
    }
    else
    {
        snprintf(buffer, size, "INVALID ACTION %d", action);
    }
}

static void power_grid_draw_sidebar(const PowerGrid *env)
{
    const int x = POWER_GRID_MAP_WIDTH + 20;
    const int width = POWER_GRID_RENDER_WIDTH - x - 20;
    Rectangle panel = {(float)x, 18.0f, (float)width, 974.0f};
    DrawRectangleRounded(panel, 0.03f, 6, POWER_GRID_PANEL);
    DrawRectangleRoundedLinesEx(panel, 0.03f, 6, 1.0f, POWER_GRID_PANEL_EDGE);

    int invalid = env->solution.status != POWER_GRID_SOLVE_OK;
    int overloaded = !invalid && env->solution.max_rho > 1.0;
    Color status_color = invalid || overloaded ? POWER_GRID_OVERLOAD : POWER_GRID_SAFE;
    const char *status = invalid ? power_grid_solve_status_name(env->solution.status) : (overloaded ? "OVERLOAD" : "SECURE");
    DrawText("GRID STATUS", x + 18, 36, 18, POWER_GRID_MUTED);
    DrawText(status, x + 18, 62, 28, status_color);
    DrawText(TextFormat("max %.1f%%", 100.0 * env->solution.max_rho),
             x + width - 125, 68, 18, POWER_GRID_TEXT);

    char period_label[16];
    if (env->offline_scenarios && env->offline_scenario != NULL)
        snprintf(period_label, sizeof(period_label), "%u",
                 env->offline_scenario->date_yyyymmdd);
    else
        snprintf(period_label, sizeof(period_label), "P%d",
                 (int)env->synthetic_profile);
    if (env->max_episode_steps == 0)
        DrawText(TextFormat("DAY %d  STEP %d/%d  PERIOD %d  %s",
                            env->episode_step / POWER_GRID_EPISODE_STEPS + 1,
                            env->episode_step % POWER_GRID_EPISODE_STEPS,
                            POWER_GRID_EPISODE_STEPS, env->current_period, period_label),
                 x + 18, 104, 17, POWER_GRID_TEXT);
    else
        DrawText(TextFormat("STEP %d/%d   PERIOD %d  %s", env->episode_step,
                            POWER_GRID_EPISODE_STEPS, env->current_period, period_label),
                 x + 18, 104, 17, POWER_GRID_TEXT);
    if (env->offline_scenarios && env->offline_scenario != NULL)
    {
        int period = env->current_period;
        DrawText(TextFormat("%.1f C  wind %.1f m/s  rating %.0f%%",
                            env->offline_scenario->ambient_temperature_c[period],
                            env->offline_scenario->wind_speed_mps[period],
                            100.0 * env->branch_rating_scale),
                 x + 18, 132, 14, POWER_GRID_MUTED);
    }

    DrawRectangleRounded((Rectangle){x + 14.0f, 166.0f, width - 28.0f, 104.0f},
                         0.12f, 5, Fade(POWER_GRID_BG, 0.75f));
    DrawText("LAST ACTION", x + 28, 178, 17, POWER_GRID_MUTED);
    char action_text[128];
    power_grid_action_summary(env, action_text, sizeof(action_text));
    DrawText(TextFormat("ID %d", env->last_action), x + 28, 205, 16, POWER_GRID_BB1);
    DrawText(action_text, x + 28, 230, 16, POWER_GRID_TEXT);

    DrawText("RUN SUMMARY", x + 18, 304, 18, POWER_GRID_MUTED);
    float episodes = env->log.n;
    float total_steps = env->log.episode_length;
    float failed_pct = episodes > 0.0f ? 100.0f * env->log.total_failure / episodes : 0.0f;
    float safe_pct = total_steps > 0.0f ? 100.0f * env->log.overload_free_steps / total_steps : 0.0f;
    float episode_safe_pct = env->episode_step > 0 ?
        100.0f * env->episode.safe_steps / env->episode_step : 0.0f;
    DrawText(TextFormat("safe %.1f%%   served %.1f%%   controls %d", episode_safe_pct,
                        100.0 * power_grid_served_load_fraction(env),
                        env->episode.switches[0]),
             x + 18, 334, 17, POWER_GRID_TEXT);
    DrawText(TextFormat("completed %.0f   safe %.1f%%   failures %.1f%%",
                        episodes, safe_pct, failed_pct),
             x + 18, 360, 17, failed_pct > 0.0f ? POWER_GRID_WARN : POWER_GRID_SAFE);

    DrawText("HOTTEST LINES", x + 18, 404, 18, POWER_GRID_MUTED);
    int used[POWER_GRID_NUM_BRANCHES] = {0};
    for (int rank = 0; rank < 3; rank++)
    {
        int best = -1;
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        {
            if (!used[line] && env->topology.line_closed[line] &&
                (best < 0 || env->solution.branch_rho[line] > env->solution.branch_rho[best]))
            {
                best = line;
            }
        }
        if (best < 0)
            break;
        used[best] = 1;
        Color color = power_grid_line_color(env, best);
        double loading = env->ac_power_flow ? fmax(env->ac_solution.branch_from_mva[best], env->ac_solution.branch_to_mva[best]) : fabs(env->solution.branch_flow_mw[best]);
        DrawRectangle(x + 18, 436 + rank * 30, 5, 21, color);
        DrawText(TextFormat("%s  %.0f/%.0f %s", POWER_GRID_BRANCH_NAMES[best], loading,
                            power_grid_branch_rating(env, best), env->ac_power_flow ? "MVA" : "MW"),
                 x + 31, 434 + rank * 30, 16, POWER_GRID_TEXT);
        DrawText(TextFormat("%.0f%%", 100.0 * env->solution.branch_rho[best]),
                 x + width - 58, 434 + rank * 30, 16, color);
    }

    DrawText("LEGEND", x + 18, 548, 18, POWER_GRID_MUTED);
    DrawLineEx((Vector2){x + 22, 580}, (Vector2){x + 48, 580}, 4.0f, POWER_GRID_SAFE);
    DrawText("closed", x + 56, 572, 15, POWER_GRID_TEXT);
    power_grid_draw_dashed_line((Vector2){x + 168, 580}, (Vector2){x + 198, 580},
                                7.0f, 5.0f, 3.0f, POWER_GRID_OPEN);
    DrawText("open", x + 206, 572, 15, POWER_GRID_TEXT);
    power_grid_draw_legend_item(x + 22, 600, POWER_GRID_SAFE, "safe");
    power_grid_draw_legend_item(x + 112, 600, POWER_GRID_WARN, "near");
    power_grid_draw_legend_item(x + 212, 600, POWER_GRID_OVERLOAD, "over");
    DrawLineEx((Vector2){x + 22, 632}, (Vector2){x + 48, 632}, 5.0f, POWER_GRID_BB1);
    DrawText("bus 1", x + 56, 624, 15, POWER_GRID_TEXT);
    DrawLineEx((Vector2){x + 168, 632}, (Vector2){x + 194, 632}, 5.0f, POWER_GRID_BB2);
    DrawText("bus 2", x + 202, 624, 15, POWER_GRID_TEXT);

    DrawText(env->ac_power_flow ? "AC VALIDATION" : "DC TRAINING", x + 18, 680, 18,
             env->ac_power_flow ? POWER_GRID_WARN : POWER_GRID_MUTED);
    if (env->ac_power_flow)
    {
        DrawText(TextFormat("V %.3f..%.3f pu   losses %.1f MW",
                            env->ac_solution.min_voltage_pu, env->ac_solution.max_voltage_pu,
                            env->ac_solution.total_p_loss_mw),
                 x + 18, 710, 16, POWER_GRID_TEXT);
        DrawText(TextFormat("Q limits %d   P violations %d",
                            env->ac_solution.q_limit_count, env->ac_solution.generator_p_violation_count),
                 x + 18, 736, 16, POWER_GRID_TEXT);
        DrawText(TextFormat("solve %s   trips %d",
                            power_grid_ac_status_name(env->ac_solution.status), env->episode.thermal_trips),
                 x + 18, 762, 16,
                 env->ac_solution.status == POWER_GRID_AC_OK ? POWER_GRID_SAFE : POWER_GRID_OVERLOAD);
    }
    else
    {
        DrawText("Use AC replay for voltage and MVA diagnostics", x + 18, 710, 16,
                 POWER_GRID_TEXT);
    }
}
void c_render(PowerGrid *env)
{
    if (!env->rendering)
    {
        env->rendering = 1;
        InitWindow(POWER_GRID_RENDER_WIDTH, POWER_GRID_RENDER_HEIGHT,
                   "IEEE-14 Power Grid Topology Control");
        SetTargetFPS(POWER_GRID_RENDER_FPS);
    }
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    BeginDrawing();
    ClearBackground(POWER_GRID_BG);
    DrawText(TextFormat("IEEE-14  |  %s SINGLE-LINE DIAGRAM",
                        env->ac_power_flow ? "AC VALIDATION" : "DC"),
             22, 14, 32, POWER_GRID_TEXT);
    DrawText(env->ac_power_flow ?
                 "Arrows show active-power direction; dotted gray lines are open. Loading uses MVA." :
                 "Arrows show active-power direction; dotted gray lines are open. Loading uses DC MW.",
             22, 52, 20, POWER_GRID_MUTED);
    power_grid_draw_branches(env);
    power_grid_draw_stations(env);
    power_grid_draw_sidebar(env);
    EndDrawing();
}
#else
void c_render(PowerGrid *env)
{
    (void)env;
}
#endif

void c_close(PowerGrid *env)
{
    if (env->rendering)
    {
#ifndef POWER_GRID_NO_RENDER
        if (IsWindowReady())
            CloseWindow();
#endif
        env->rendering = 0;
    }
    if (env->owns_buffers)
    {
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        env->observations = NULL;
        env->actions = NULL;
        env->rewards = NULL;
        env->terminals = NULL;
        env->owns_buffers = 0;
    }
}

#endif
