#ifndef POWER_GRID_H
#define POWER_GRID_H

#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef POWER_GRID_NO_RENDER
#include "raylib.h"
#endif

#define POWER_GRID_OBS_SIZE 221
#define POWER_GRID_EPISODE_STEPS 72
#define POWER_GRID_STEPS_PER_PERIOD 6
#define POWER_GRID_NUM_PERIODS (POWER_GRID_EPISODE_STEPS / POWER_GRID_STEPS_PER_PERIOD)
#define POWER_GRID_MAX_RANDOM_OUTAGES 8

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
#define POWER_GRID_DEFAULT_SAFE_STEP_REWARD 0.01f
#define POWER_GRID_DEFAULT_SWITCH_PENALTY 0.01f
#define POWER_GRID_DEFAULT_CONGESTION_COST_WEIGHT 1.0f

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

_Static_assert(POWER_GRID_GENERATOR_Q_OBS_OFFSET + POWER_GRID_NUM_GENERATORS ==
                   POWER_GRID_OBS_SIZE,
               "power-grid observation layout must total 221 floats");
_Static_assert(POWER_GRID_EPISODE_STEPS % POWER_GRID_STEPS_PER_PERIOD == 0,
               "power-grid periods must divide the episode evenly");
_Static_assert(POWER_GRID_ACTION_NONE == 0 && POWER_GRID_ACTION_LINE == 1 &&
                   POWER_GRID_ACTION_TERMINAL == 2 && POWER_GRID_ACTION_COUPLER == 3,
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
    int switches[4]; /* total, line, terminal/busbar, coupler */
    int no_op_actions;
    int safe_steps;
    int thermal_trips;
    int random_events;
    int demand_served_steps;
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
    PowerGridEpisodeStats episode;
    unsigned char line_available[POWER_GRID_NUM_BRANCHES];
    double line_thermal_stress[POWER_GRID_NUM_BRANCHES];
    int scheduled_random_event_count;
    int scheduled_random_event_period[POWER_GRID_MAX_RANDOM_OUTAGES];
    int scheduled_random_event_line[POWER_GRID_MAX_RANDOM_OUTAGES];
    int active_random_event_line;
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
    int single_episode_evaluation;
    float failure_reward;
    float safe_step_reward;
    float switch_penalty;
    float congestion_cost_weight;
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
        active_node[power_grid_terminal_node(&env->topology,
            POWER_GRID_GENERATOR_TERMINAL(generator), POWER_GRID_GENERATOR_BUSES[generator])] = 1;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
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
    action.type = power_grid_apply_action(&env->topology, action.value);
    action.switched = action.type > POWER_GRID_ACTION_NONE;
    env->last_action = action.value;
    env->last_action_type = action.type;

    if (action.type == POWER_GRID_ACTION_LINE)
    {
        int line = action.value - POWER_GRID_LINE_ACTION_OFFSET;
        int previous_state = env->topology.line_closed[line] ^ 1;
        if (!env->line_available[line])
            env->topology.line_closed[line] = 0;
        action.switched = env->topology.line_closed[line] != previous_state;
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
    env->episode.no_op_actions += action.type == POWER_GRID_ACTION_NONE;
    env->episode.switches[0] += action.switched;
    if (action.switched)
        env->episode.switches[action.type]++;
    return action;
}

static float calculate_reward(const PowerGrid *env,
                              PowerGridSolveStatus solve_status,
                              double constraint_cost, int switched, int safe)
{
    if (solve_status != POWER_GRID_SOLVE_OK)
        return env->failure_reward;
    return (float)(-constraint_cost - env->switch_penalty * switched +
                   env->safe_step_reward * safe);
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
        (float)env->episode.demand_served_steps / POWER_GRID_EPISODE_STEPS;
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
            env->scheduled_random_event_count =
                requested < POWER_GRID_MAX_RANDOM_OUTAGES ? requested :
                                                            POWER_GRID_MAX_RANDOM_OUTAGES;
            for (int event = 0; event < env->scheduled_random_event_count; event++)
            {
                int period;
                int unique_period;
                do
                {
                    period = 1 + (int)(next_random(env) % (POWER_GRID_NUM_PERIODS - 1));
                    unique_period = 1;
                    for (int prior = 0; prior < event; prior++)
                        unique_period &= env->scheduled_random_event_period[prior] != period;
                }
                while (!unique_period);
                env->scheduled_random_event_period[event] = period;

                int line;
                int unique_line;
                do
                {
                    line = (int)(next_random(env) % POWER_GRID_NUM_BRANCHES);
                    unique_line = power_grid_random_event_eligible(line);
                    for (int prior = 0; prior < event; prior++)
                        unique_line &= env->scheduled_random_event_line[prior] != line;
                    if (unique_line)
                    {
                        PowerGridTopology candidate;
                        power_grid_topology_normal(&candidate);
                        candidate.line_closed[line] = 0;
                        for (int prior = 0; prior < event; prior++)
                            candidate.line_closed[env->scheduled_random_event_line[prior]] = 0;
                        unique_line = power_grid_validate_topology(
                            &candidate, NULL, NULL) == POWER_GRID_SOLVE_OK;
                    }
                }
                while (!unique_line);
                env->scheduled_random_event_line[event] = line;
            }
        }
    }
    env->current_period = 0;
    power_grid_set_operating_period(env, 0);
    power_grid_solve_environment(env);
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
    PowerGridAppliedAction action = apply_agent_action(env, env->actions[0]);
    env->terminals[0] = 0.0f;
    /* In DC mode, a no-op before a period transition leaves every value in the
     * observation vector unchanged. Track whether a later state mutation
     * actually requires rebuilding the 221-float observation. */
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

    int metric_safe = 0;
    int reward_safe = 0;
    double constraint_cost = 0.0;
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
        env->episode.safe_steps += metric_safe;
        env->episode.demand_served_steps++;
    }

    env->rewards[0] = calculate_reward(
        env, status, constraint_cost, action.switched, reward_safe);
    env->terminals[0] = status != POWER_GRID_SOLVE_OK ||
                        env->episode_step >= POWER_GRID_EPISODE_STEPS;

    if (env->terminals[0])
    {
        env->episode_return += env->rewards[0];
        power_grid_finish_episode(env);
        return;
    }

    int random_event_applied = 0;
    int next_period = env->episode_step / POWER_GRID_STEPS_PER_PERIOD;
    if (next_period != env->current_period)
    {
        observations_dirty = 1;
        observation_only_terminal = -1;
        env->current_period = next_period;
        power_grid_set_operating_period(env, next_period);
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
        /* New injections can expose an infeasible state without another agent action. */
        env->log.event_failure += random_event_applied;
        env->rewards[0] = calculate_reward(env, status, 0.0, 0, 0);
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
    env->failure_reward = POWER_GRID_DEFAULT_FAILURE_REWARD;
    env->safe_step_reward = POWER_GRID_DEFAULT_SAFE_STEP_REWARD;
    env->switch_penalty = POWER_GRID_DEFAULT_SWITCH_PENALTY;
    env->congestion_cost_weight = POWER_GRID_DEFAULT_CONGESTION_COST_WEIGHT;
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

static void power_grid_prepare_layout(void)
{
    /* IEEE-14 uses the fixed single-line layout above. */
}

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
    if (fabsf(dx) > 4.0f)
        x += dx > 0.0f ? 17.0f : -17.0f;
    return (Vector2){x, station.y + (busbar ? 3.0f : -3.0f)};
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
        DrawLineEx(from, to, 3.5f, Fade(BLACK, 0.65f));
        if (env->topology.line_closed[line])
        {
            float width = 1.0f + 1.5f * fminf((float)env->solution.branch_rho[line], 1.2f);
            DrawLineEx(from, to, width, color);
        }
        else
        {
            power_grid_draw_dashed_line(from, to, 5.0f, 4.0f, 1.5f, color);
        }
    }
}

static void power_grid_draw_generator(const PowerGrid *env, int generator)
{
    int bus = POWER_GRID_GENERATOR_BUSES[generator];
    int terminal = POWER_GRID_GENERATOR_TERMINAL(generator);
    int bar = env->topology.terminal_busbar[terminal];
    Vector2 station = POWER_GRID_STATION_POSITIONS[bus];
    Vector2 symbol = {station.x - 10.0f + 4.0f * (generator % 4), station.y - 13.0f};
    DrawCircleV(symbol, 2.5f, bar ? POWER_GRID_BB2 : POWER_GRID_BB1);
}

static void power_grid_draw_load(const PowerGrid *env, int load)
{
    int bus = POWER_GRID_LOAD_BUSES[load];
    int terminal = POWER_GRID_LOAD_TERMINAL(load);
    int bar = env->topology.terminal_busbar[terminal];
    Vector2 station = POWER_GRID_STATION_POSITIONS[bus];
    Vector2 symbol = {station.x - 10.0f + 4.0f * (load % 6), station.y + 14.0f};
    DrawCircleV(symbol, 2.0f, bar ? POWER_GRID_BB2 : (Color){255, 154, 105, 255});
}

static void power_grid_draw_stations(const PowerGrid *env)
{
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        Vector2 position = POWER_GRID_STATION_POSITIONS[bus];
        Rectangle panel = {position.x - 18.0f, position.y - 9.0f, 36.0f, 18.0f};
        DrawRectangleRounded(panel, 0.18f, 4, POWER_GRID_PANEL);
        DrawRectangleRoundedLinesEx(panel, 0.18f, 4, 1.0f, POWER_GRID_PANEL_EDGE);
        DrawText(TextFormat("%d", bus + 1), (int)position.x - 7, (int)position.y - 7,
                 11, POWER_GRID_TEXT);
        DrawLineEx((Vector2){position.x - 15, position.y - 3},
                   (Vector2){position.x + 15, position.y - 3}, 2.0f, POWER_GRID_BB1);
        DrawLineEx((Vector2){position.x - 15, position.y + 3},
                   (Vector2){position.x + 15, position.y + 3}, 2.0f, POWER_GRID_BB2);
        if (env->topology.coupler_closed[bus])
        {
            DrawLineEx((Vector2){position.x, position.y - 3},
                       (Vector2){position.x, position.y + 3}, 2.0f, POWER_GRID_SAFE);
        }
        else
        {
            DrawCircleLines((int)position.x, (int)position.y - 2, 1.0f, POWER_GRID_OPEN);
            DrawCircleLines((int)position.x, (int)position.y + 2, 1.0f, POWER_GRID_OPEN);
        }
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
    DrawText(TextFormat("STEP %d/%d   PERIOD %d  %s", env->episode_step,
                        POWER_GRID_EPISODE_STEPS, env->current_period, period_label),
             x + 18, 104, 17, POWER_GRID_TEXT);
    float steps = env->episode_step > 0 ? (float)env->episode_step : 1.0f;
    float current_perf = env->solution.status == POWER_GRID_SOLVE_OK
                             ? env->episode.safe_steps / steps
                             : 0.0f;
    float current_score = env->episode.no_op_actions / steps;
    DrawText(TextFormat("score %.3f   perf %.3f", current_score, current_perf),
             x + 18, 130, 17, POWER_GRID_TEXT);
    if (env->offline_scenarios && env->offline_scenario != NULL)
    {
        int period = env->current_period;
        DrawText(TextFormat("%.1f C  wind %.1f m/s  rating %.0f%%",
                            env->offline_scenario->ambient_temperature_c[period],
                            env->offline_scenario->wind_speed_mps[period],
                            100.0 * env->branch_rating_scale),
                 x + 18, 149, 14, POWER_GRID_MUTED);
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
    DrawText(TextFormat("episodes %.0f   switches %.0f", episodes, env->log.total_switches),
             x + 18, 334, 17, POWER_GRID_TEXT);
    DrawText(TextFormat("safe steps %.1f%%   failures %.1f%%", safe_pct, failed_pct),
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
    power_grid_prepare_layout();
    DrawText(TextFormat("IEEE-14  |  %s TOPOLOGY OVERVIEW",
                        env->ac_power_flow ? "AC VALIDATION" : "DC"),
             22, 14, 32, POWER_GRID_TEXT);
    DrawText(env->ac_power_flow ? "Compact 12-column layout; line color shows MVA loading, dotted lines are open." : "Compact 12-column layout; line color shows DC MW loading, dotted lines are open.",
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
