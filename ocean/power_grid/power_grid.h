#ifndef POWER_GRID_H
#define POWER_GRID_H

#include "power_grid_solver.h"
#include "power_grid_ac.h"

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
/* Any line except radial 7-8 can be removed without disconnecting the normal topology. */
#define POWER_GRID_RANDOM_EVENT_LINE_MASK \
    (((1u << POWER_GRID_NUM_BRANCHES) - 1u) & ~(1u << 13))

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
#define POWER_GRID_FAILURE_REWARD (-5.0f)
#define POWER_GRID_SAFE_STEP_REWARD 0.2f
#define POWER_GRID_SWITCH_PENALTY 0.001f
#define POWER_GRID_CONGESTION_COST_WEIGHT 1.0

/* AC evaluation-only reward configuration. Ignored during DC training. */
#define POWER_GRID_AC_VOLTAGE_VIOLATION_COST_WEIGHT 1.0
#define POWER_GRID_AC_THERMAL_TRIP_PENALTY 1.0

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
    float topology_failure;
    float solver_failure;
    float event_failure;
    float total_switches;
    float line_switches;
    float busbar_switches;
    float coupler_switches;
    float overload_free_steps;
    float ac_voltage_violation_steps;
    float ac_generator_p_violation_steps;
    float ac_q_limit_events;
    float ac_mean_active_loss_mw;
    float ac_nonconvergence;
    float ac_thermal_trips;
    float ac_peak_thermal_stress;
    float maintenance_events;
    float random_events;
    float n;
} Log;

typedef struct
{
    int switches[4]; /* total, line, terminal/busbar, coupler */
    int no_op_actions;
    int safe_steps;
    int voltage_violation_steps;
    int generator_p_violation_steps;
    int q_limit_events;
    int ac_nonconvergence;
    int thermal_trips;
    int maintenance_events;
    int random_events;
    double active_loss_mw_sum;
    double peak_thermal_stress;
} PowerGridEpisodeStats;

typedef struct
{
    int value;
    PowerGridActionType type;
    int switched;
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
    PowerGridProfile episode_profiles[POWER_GRID_NUM_PERIODS];
    int episode_step;
    int current_period;
    PowerGridEpisodeStats episode;
    unsigned char line_available[POWER_GRID_NUM_BRANCHES];
    double line_thermal_stress[POWER_GRID_NUM_BRANCHES];
    unsigned char line_maintenance[POWER_GRID_NUM_BRANCHES];
    unsigned char maintenance_was_closed[POWER_GRID_NUM_BRANCHES];
    int scheduled_random_event_period;
    int scheduled_random_event_line;
    int active_random_event_line;
    float episode_return;
    int pending_reset;
    int last_action;
    PowerGridActionType last_action_type;
    int ac_power_flow;
    int evaluation_scenarios;
    int offline_scenarios;
    int offline_scenario_validation;
    double offline_scenario_probability;
    int random_events;
    double random_event_probability;
    double branch_rating_scale;
    double ambient_temperature_c;
    double wind_speed_mps;
    double solar_irradiance_wm2;
} PowerGrid;

void c_reset(PowerGrid *env);

static const char *const POWER_GRID_EVALUATION_TIMES[POWER_GRID_NUM_PERIODS] = {
    "00:00",
    "02:00",
    "04:00",
    "06:00",
    "08:00",
    "10:00",
    "12:00",
    "14:00",
    "16:00",
    "18:00",
    "20:00",
    "22:00",
};

static void power_grid_chronological_point(PowerGridOperatingPoint *point, int period)
{
    static const double load_scale[POWER_GRID_NUM_PERIODS] = {
        .78,
        .76,
        .80,
        .90,
        1.00,
        1.08,
        1.10,
        1.05,
        1.12,
        1.08,
        .95,
        .85,
    };
    static const double solar_mw[POWER_GRID_NUM_PERIODS] = {
        0,
        0,
        0,
        10,
        35,
        70,
        90,
        75,
        35,
        5,
        0,
        0,
    };
    static const double wind_mw[POWER_GRID_NUM_PERIODS] = {
        35,
        30,
        28,
        25,
        20,
        18,
        22,
        28,
        35,
        40,
        42,
        38,
    };
    power_grid_operating_point_nominal(point);
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
    {
        point->load_mw[load] *= load_scale[period];
    }
    point->generator_mw[2] = solar_mw[period]; /* Synthetic solar at generator bus 3. */
    point->generator_mw[3] = wind_mw[period];  /* Synthetic wind at generator bus 6. */
}

static void power_grid_set_operating_period(PowerGrid *env, int period)
{
    env->branch_rating_scale = 1.0;
    env->ambient_temperature_c = 3.6;
    env->wind_speed_mps = 3.13;
    env->solar_irradiance_wm2 = 7.0;
    if (env->evaluation_scenarios)
    {
        power_grid_chronological_point(&env->operating_point, period);
    }
    else if (env->offline_scenarios && env->offline_scenario != NULL)
    {
        const PowerGridOfflineScenario *scenario = env->offline_scenario;
        power_grid_operating_point_nominal(&env->operating_point);
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            env->operating_point.load_mw[load] *= scenario->load_scale[period];
        env->operating_point.generator_mw[2] = scenario->solar_mw[period];
        env->operating_point.generator_mw[3] = scenario->wind_mw[period];
        env->ambient_temperature_c = scenario->ambient_temperature_c[period];
        env->wind_speed_mps = scenario->wind_speed_mps[period];
        env->solar_irradiance_wm2 = scenario->solar_irradiance_wm2[period];
        env->branch_rating_scale = power_grid_weather_rating_scale(
            env->ambient_temperature_c, env->wind_speed_mps,
            env->solar_irradiance_wm2);
    }
    else
    {
        power_grid_operating_point_profile(&env->operating_point, env->episode_profiles[period]);
    }
}

static void power_grid_apply_dynamic_ratings(PowerGrid *env)
{
    if (env->offline_scenario == NULL || env->evaluation_scenarios ||
        env->solution.status != POWER_GRID_SOLVE_OK)
        return;

    env->solution.max_rho = 0.0;
    env->solution.congestion_cost = 0.0;
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
        memset(&env->ac_solution, 0, sizeof(env->ac_solution));
        PowerGridSolveStatus status = power_grid_solve(
            &env->topology, &env->operating_point, &env->solution);
        power_grid_apply_dynamic_ratings(env);
        return status;
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

/* Evaluation-only inverse-time protection. Stress is an intentionally simple
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
        if (env->line_thermal_stress[line] > env->episode.peak_thermal_stress)
        {
            env->episode.peak_thermal_stress = env->line_thermal_stress[line];
        }
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
    };
    action.type = power_grid_apply_action(&env->topology, action.value);
    action.switched = action.type > POWER_GRID_ACTION_NONE;
    env->last_action = action.value;
    env->last_action_type = action.type;

    if (action.type == POWER_GRID_ACTION_LINE)
    {
        int line = action.value - POWER_GRID_LINE_ACTION_OFFSET;
        int previous_state = env->topology.line_closed[line] ^ 1;
        if (!env->line_available[line] || env->line_maintenance[line])
            env->topology.line_closed[line] = 0;
        action.switched = env->topology.line_closed[line] != previous_state;
    }
    env->episode.no_op_actions += action.type == POWER_GRID_ACTION_NONE;
    env->episode.switches[0] += action.switched;
    if (action.switched)
        env->episode.switches[action.type]++;
    return action;
}

static float calculate_reward(PowerGridSolveStatus solve_status,
                              double constraint_cost, int switched, int safe)
{
    if (solve_status != POWER_GRID_SOLVE_OK)
        return POWER_GRID_FAILURE_REWARD;
    return (float)(-constraint_cost - POWER_GRID_SWITCH_PENALTY * switched +
                   POWER_GRID_SAFE_STEP_REWARD * safe);
}

static void power_grid_finish_episode(PowerGrid *env)
{
    if (env->ac_power_flow && env->solution.status != POWER_GRID_SOLVE_OK &&
        (env->ac_solution.status == POWER_GRID_AC_DIVERGED ||
         env->ac_solution.status == POWER_GRID_AC_SINGULAR ||
         env->ac_solution.status == POWER_GRID_AC_NONFINITE))
    {
        env->episode.ac_nonconvergence++;
    }
    int failed = env->solution.status != POWER_GRID_SOLVE_OK;
    float steps = env->episode_step > 0 ? (float)env->episode_step : 1.0f;
    float perf = failed ? 0.0f : (float)env->episode.safe_steps / steps;
    float score = (float)env->episode.no_op_actions / steps;
    /* perf measures grid security; score measures how often switching was avoided. */
    env->log.perf += perf;
    env->log.overload_free_steps += (float)env->episode.safe_steps;
    if (env->ac_power_flow)
    {
        env->log.ac_voltage_violation_steps += (float)env->episode.voltage_violation_steps;
        env->log.ac_generator_p_violation_steps +=
            (float)env->episode.generator_p_violation_steps;
        env->log.ac_q_limit_events += (float)env->episode.q_limit_events;
        env->log.ac_mean_active_loss_mw += (float)(env->episode.active_loss_mw_sum / steps);
        env->log.ac_nonconvergence += (float)env->episode.ac_nonconvergence;
        env->log.ac_thermal_trips += (float)env->episode.thermal_trips;
        env->log.ac_peak_thermal_stress += (float)env->episode.peak_thermal_stress;
    }
    env->log.maintenance_events += (float)env->episode.maintenance_events;
    env->log.random_events += (float)env->episode.random_events;
    env->log.score += score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_step;
    env->log.total_failure += failed;
    env->log.topology_failure += env->solution.status == POWER_GRID_INVALID_TOPOLOGY ||
                                 env->solution.status == POWER_GRID_DISCONNECTED_LOAD ||
                                 env->solution.status == POWER_GRID_DISCONNECTED_GENERATOR ||
                                 env->solution.status == POWER_GRID_ISLANDED;
    env->log.solver_failure += env->solution.status == POWER_GRID_SINGULAR ||
                               env->solution.status == POWER_GRID_NONFINITE || env->solution.status == POWER_GRID_INVALID_INPUT;
    env->log.event_failure += failed && env->episode.random_events > 0;
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
    else
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
    memset(env->line_maintenance, 0, sizeof(env->line_maintenance));
    memset(env->maintenance_was_closed, 0, sizeof(env->maintenance_was_closed));
    env->pending_reset = 0;
    env->last_action = POWER_GRID_ACTION_NONE;
    env->last_action_type = POWER_GRID_ACTION_NONE;
    env->episode_return = 0.0f;
    env->active_random_event_line = -1;
    env->offline_scenario = NULL;
    env->branch_rating_scale = 1.0;
    if (env->offline_scenarios && !env->evaluation_scenarios)
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
    env->episode_profiles[0] = POWER_GRID_PROFILE_P0_NOMINAL;
    for (int period = 1; period < POWER_GRID_NUM_PERIODS; period++)
    {
        if (env->evaluation_scenarios || env->offline_scenario != NULL)
        {
            env->episode_profiles[period] = POWER_GRID_PROFILE_P0_NOMINAL;
        }
        else
        {
            /* Puffer initializes a separate rng per vector environment. Mix high and low LCG bits. */
            unsigned int sample = next_random(env);
            env->episode_profiles[period] = (PowerGridProfile)(1 +
                                                               sample % (POWER_GRID_NUM_PROFILES - 1));
        }
    }
    env->scheduled_random_event_period = -1;
    env->scheduled_random_event_line = -1;
    if (env->random_events && !env->evaluation_scenarios && !env->ac_power_flow)
    {
        double unit_sample = (double)(next_random(env) >> 8) / 16777216.0;
        if (unit_sample < env->random_event_probability)
        {
            env->scheduled_random_event_period = 1 +
                (int)(next_random(env) % (POWER_GRID_NUM_PERIODS - 1));
            unsigned int selected = next_random(env) %
                (unsigned int)__builtin_popcount(POWER_GRID_RANDOM_EVENT_LINE_MASK);
            for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            {
                if ((POWER_GRID_RANDOM_EVENT_LINE_MASK & (1u << line)) == 0)
                    continue;
                if (selected-- == 0)
                {
                    env->scheduled_random_event_line = line;
                    break;
                }
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
    if (env->rendering && env->pending_reset)
    {
        c_reset(env);
        return;
    }
    PowerGridAppliedAction action = apply_agent_action(env, env->actions[0]);
    env->terminals[0] = 0.0f;

    int new_trips = 0;
    PowerGridSolveStatus status;
    if (action.type == POWER_GRID_ACTION_INVALID)
    {
        status = env->solution.status = POWER_GRID_INVALID_INPUT;
    }
    else
    {
        status = power_grid_solve_environment(env);
        if (status == POWER_GRID_SOLVE_OK)
            new_trips = power_grid_update_ac_protection(env);
        status = env->solution.status;
    }
    env->episode_step++;

    int safe = 0;
    double constraint_cost = 0.0;
    if (status == POWER_GRID_SOLVE_OK)
    {
        /* Scheduled generator-P violations are exogenous data faults, not agent actions. */
        safe = env->solution.max_rho <= 1.0 &&
               (!env->ac_power_flow || env->ac_solution.voltage_violation_count == 0);
        constraint_cost = POWER_GRID_CONGESTION_COST_WEIGHT * env->solution.congestion_cost;
        if (env->ac_power_flow)
        {
            constraint_cost += POWER_GRID_AC_VOLTAGE_VIOLATION_COST_WEIGHT *
                               env->ac_solution.voltage_violation_cost;
            constraint_cost += POWER_GRID_AC_THERMAL_TRIP_PENALTY * new_trips;
        }
        env->episode.safe_steps += safe;
        if (env->ac_power_flow)
        {
            env->episode.voltage_violation_steps +=
                env->ac_solution.voltage_violation_count > 0;
            env->episode.generator_p_violation_steps +=
                env->ac_solution.generator_p_violation_count > 0;
            env->episode.q_limit_events += env->ac_solution.q_limit_count;
            env->episode.active_loss_mw_sum += env->ac_solution.total_p_loss_mw;
        }
    }

    env->rewards[0] = calculate_reward(status, constraint_cost, action.switched, safe);
    env->terminals[0] = status != POWER_GRID_SOLVE_OK ||
                        env->episode_step >= POWER_GRID_EPISODE_STEPS;

    if (env->terminals[0])
    {
        env->episode_return += env->rewards[0];
        power_grid_finish_episode(env);
        return;
    }

    int next_period = env->episode_step / POWER_GRID_STEPS_PER_PERIOD;
    if (next_period != env->current_period)
    {
        env->current_period = next_period;
        if (env->evaluation_scenarios)
        {
            const int maintenance_line = 16; /* Connected 9-14 daytime outage. */
            if (next_period == 4)
            {
                env->line_maintenance[maintenance_line] = 1;
                env->maintenance_was_closed[maintenance_line] =
                    env->topology.line_closed[maintenance_line];
                env->topology.line_closed[maintenance_line] = 0;
                env->episode.maintenance_events++;
            }
            else if (next_period == 8)
            {
                env->line_maintenance[maintenance_line] = 0;
                if (env->line_available[maintenance_line])
                    env->topology.line_closed[maintenance_line] =
                        env->maintenance_was_closed[maintenance_line];
            }
        }
        power_grid_set_operating_period(env, next_period);
        if (next_period == env->scheduled_random_event_period)
        {
            PowerGridTopology normal;
            power_grid_topology_normal(&normal);
            if (memcmp(&env->topology, &normal, sizeof(normal)) == 0)
            {
                int line = env->scheduled_random_event_line;
                env->line_available[line] = 0;
                env->topology.line_closed[line] = 0;
                env->active_random_event_line = line;
                env->episode.random_events++;
            }
        }
        status = power_grid_solve_environment(env);
    }
    if (status != POWER_GRID_SOLVE_OK)
    {
        /* New injections can expose an infeasible state without another agent action. */
        env->rewards[0] = calculate_reward(status, 0.0, 0, 0);
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        power_grid_finish_episode(env);
        return;
    }
    env->episode_return += env->rewards[0];
    power_grid_compute_observations(env);
}

void power_grid_allocate(PowerGrid *env)
{
    env->num_agents = 1;
    env->rng = 0;
    env->observations = calloc(POWER_GRID_OBS_SIZE, sizeof(float));
    env->actions = calloc(1, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(float));
    env->owns_buffers = 1;
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
    {100, 285},
    {315, 205},
    {570, 135},
    {575, 355},
    {305, 415},
    {455, 620},
    {785, 350},
    {1020, 205},
    {1020, 485},
    {1250, 570},
    {870, 635},
    {530, 835},
    {870, 845},
    {1235, 810},
};

static const Vector2 POWER_GRID_LABEL_OFFSETS[POWER_GRID_NUM_BRANCHES] = {
    {30, 25},
    {-34, 18},
    {30, 0},
    {-5, 30},
    {-30, 22},
    {24, 0},
    {0, 24},
    {-24, -27},
    {25, 10},
    {-28, 5},
    {-34, -22},
    {-32, 14},
    {25, 16},
    {0, -25},
    {24, 10},
    {18, -23},
    {30, 18},
    {-27, 16},
    {0, -23},
    {20, 11},
};

static Color power_grid_line_color(const PowerGrid *env, int line)
{
    if (!env->line_available[line])
        return POWER_GRID_OVERLOAD;
    if (env->line_maintenance[line])
        return POWER_GRID_WARN;
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
    if (length < 1.0f)
        return;

    dx /= length;
    dy /= length;

    /* Near-zero active power has no meaningful direction. The branch label
       states this explicitly; the solid conductor still shows that it is closed. */
    if (fabs(flow) < 0.05)
        return;

    if (flow < 0.0)
    {
        Vector2 tmp = from;
        from = to;
        to = tmp;
        dx = to.x - from.x;
        dy = to.y - from.y;
        dx /= length;
        dy /= length;
    }

    float progress = fmodf((float)(GetTime() * 0.40) + phase, 1.0f);
    float head = fminf(7.0f, length * 0.12f);
    float tail = fminf(6.0f, length * 0.10f);
    for (int marker = 0; marker < 2; marker++)
    {
        /* Keep markers away from the center, where the branch label is placed. */
        float fraction = marker == 0 ? 0.10f + 0.22f * progress : 0.68f + 0.22f * progress;
        float distance = fminf(fmaxf(length * fraction, tail), length - head);
        Vector2 tip = {from.x + dx * (distance + head), from.y + dy * (distance + head)};
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
    Rectangle box = {position.x - width * 0.5f, position.y - 12.0f, (float)width, 24.0f};
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
            power_grid_draw_dashed_line(from, to, 9.0f, 7.0f, 2.5f, color);
            DrawCircleLines((int)from.x, (int)from.y, 5.0f, POWER_GRID_OPEN);
            DrawCircleLines((int)to.x, (int)to.y, 5.0f, POWER_GRID_OPEN);
        }
    }

    /* Labels are a second pass so they remain legible over crossing conductors. */
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
                char active_power_text[24];
                if (fabs(active_power_mw) < 0.05)
                    snprintf(active_power_text, sizeof(active_power_text), "NO MW FLOW");
                else
                    snprintf(active_power_text, sizeof(active_power_text), "%+.0fMW",
                             active_power_mw);
                snprintf(text, sizeof(text), "%s%s  %s %.0f/%.0fMVA %.0f%%",
                         branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                         active_power_text,
                         fmax(env->ac_solution.branch_from_mva[line],
                              env->ac_solution.branch_to_mva[line]),
                         power_grid_branch_rating(env, line),
                         100.0 * env->solution.branch_rho[line]);
            }
            else
            {
                snprintf(text, sizeof(text), "%s%s  %+.0f/%.0fMW  %.0f%%",
                         branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                         env->solution.branch_flow_mw[line], branch->thermal_limit_mw,
                         100.0 * env->solution.branch_rho[line]);
            }
        }
        else
        {
            snprintf(text, sizeof(text), "%s  %s", POWER_GRID_BRANCH_NAMES[line],
                     env->ac_power_flow && !env->line_available[line] ? "TRIPPED" : (env->line_maintenance[line] ? "MAINTENANCE" : "OPEN"));
        }
        power_grid_draw_label(label, text, power_grid_line_color(env, line));
    }

    /* Draw flow markers last so wide branch labels cannot hide them. */
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
    DrawLineEx(busbar, (Vector2){symbol.x, symbol.y + 13.0f}, 2.0f,
               bar ? POWER_GRID_BB2 : POWER_GRID_BB1);
    DrawCircleV(symbol, 17.0f, (Color){34, 78, 100, 255});
    DrawCircleLines((int)symbol.x, (int)symbol.y, 17.0f, POWER_GRID_BB1);
    DrawText("G", (int)symbol.x - 7, (int)symbol.y - 11, 21, RAYWHITE);
    double output = generator == 0 ? env->solution.slack_generation_mw : env->operating_point.generator_mw[generator];
    const char *label = env->ac_power_flow ? TextFormat("%.0fMW %+.0fMVAr", output,
                                                        env->ac_solution.generator_q_mvar[generator])
                                           : TextFormat("%.1f MW", output);
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
    DrawLineEx(busbar, (Vector2){symbol.x, symbol.y - 12.0f}, 2.0f,
               bar ? POWER_GRID_BB2 : POWER_GRID_BB1);
    Rectangle load_box = {symbol.x - 15.0f, symbol.y - 13.0f, 30.0f, 26.0f};
    DrawRectangleRounded(load_box, 0.25f, 4, (Color){178, 77, 50, 255});
    DrawRectangleRoundedLinesEx(load_box, 0.25f, 4, 1.0f, (Color){255, 154, 105, 255});
    DrawText("L", (int)symbol.x - 6, (int)symbol.y - 10, 20, RAYWHITE);
    const char *label = env->ac_power_flow ? TextFormat("%.0fMW %+.0fMVAr",
                                                        env->operating_point.load_mw[load],
                                                        power_grid_ac_load_q_mvar(&env->operating_point, load))
                                           : TextFormat("%.1f MW", env->operating_point.load_mw[load]);
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
        DrawText(TextFormat("SUB %d", bus + 1), (int)position.x - 54, (int)position.y - 27,
                 14, POWER_GRID_MUTED);
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
                 (int)position.y - 27, 14, injection >= 0.0 ? POWER_GRID_SAFE : POWER_GRID_MUTED);
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
        if (terminal < 40)
        {
            snprintf(buffer, size, "LINE %s %s -> BUSBAR %d",
                     POWER_GRID_BRANCH_NAMES[terminal / 2], terminal % 2 ? "TO" : "FROM", busbar);
        }
        else if (terminal < 45)
        {
            int generator = terminal - 40;
            snprintf(buffer, size, "GENERATOR %d -> BUSBAR %d", generator, busbar);
        }
        else
        {
            snprintf(buffer, size, "LOAD %d -> BUSBAR %d", terminal - 45, busbar);
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
    if (env->evaluation_scenarios)
        snprintf(period_label, sizeof(period_label), "%s",
                 POWER_GRID_EVALUATION_TIMES[env->current_period]);
    else if (env->offline_scenarios && env->offline_scenario != NULL)
        snprintf(period_label, sizeof(period_label), "%u",
                 env->offline_scenario->date_yyyymmdd);
    else
        snprintf(period_label, sizeof(period_label), "P%d",
                 (int)env->episode_profiles[env->current_period]);
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
    DrawText(TextFormat("IEEE-14  |  %s SINGLE-LINE DIAGRAM",
                        env->ac_power_flow ? "AC VALIDATION" : "DC"),
             22, 14, 32, POWER_GRID_TEXT);
    DrawText(env->ac_power_flow ? "Arrows show active-power direction. NO MW FLOW lines stay solid because they are closed. Loading uses MVA." : "Arrows show active-power direction. NO MW FLOW lines stay solid because they are closed. Loading uses DC MW.",
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
