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

#define POWER_GRID_OBS_SIZE 144
#define POWER_GRID_EPISODE_STEPS 48
#define POWER_GRID_STEPS_PER_PERIOD 4
#define POWER_GRID_NUM_PERIODS (POWER_GRID_EPISODE_STEPS / POWER_GRID_STEPS_PER_PERIOD)
#define POWER_GRID_LINE_OBS_FEATURES 3
#define POWER_GRID_LINE_OBS_OFFSET 0
#define POWER_GRID_TERMINAL_OBS_OFFSET \
    (POWER_GRID_NUM_BRANCHES * POWER_GRID_LINE_OBS_FEATURES)
#define POWER_GRID_COUPLER_OBS_OFFSET \
    (POWER_GRID_TERMINAL_OBS_OFFSET + POWER_GRID_NUM_TERMINALS)
#define POWER_GRID_INJECTION_OBS_OFFSET \
    (POWER_GRID_COUPLER_OBS_OFFSET + POWER_GRID_NUM_SUBSTATIONS)

_Static_assert(POWER_GRID_INJECTION_OBS_OFFSET + POWER_GRID_NUM_SUBSTATIONS ==
    POWER_GRID_OBS_SIZE, "power-grid observation layout must total 144 floats");
_Static_assert(POWER_GRID_EPISODE_STEPS % POWER_GRID_STEPS_PER_PERIOD == 0,
    "power-grid periods must divide the episode evenly");
_Static_assert(POWER_GRID_ACTION_NONE == 0 && POWER_GRID_ACTION_LINE == 1 &&
    POWER_GRID_ACTION_TERMINAL == 2 && POWER_GRID_ACTION_COUPLER == 3,
    "power-grid action types must index episode switch counters");

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float total_failure;
    float topology_failure;
    float solver_failure;
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
    float n;
} Log;

typedef struct {
    int switches[4]; /* total, line, terminal/busbar, coupler */
    int safe_steps;
    int voltage_violation_steps;
    int generator_p_violation_steps;
    int q_limit_events;
    int ac_nonconvergence;
    int thermal_trips;
    int maintenance_events;
    double active_loss_mw_sum;
    double peak_thermal_stress;
} PowerGridEpisodeStats;

typedef struct {
    Log log;
    int rendering;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    int owns_buffers;

    PowerGridTopology topology;
    PowerGridOperatingPoint operating_point;
    PowerGridSolveResult solution;
    PowerGridACSolveResult ac_solution;
    PowerGridProfile episode_profiles[POWER_GRID_NUM_PERIODS];
    int episode_step;
    int current_period;
    PowerGridEpisodeStats episode;
    unsigned char line_available[POWER_GRID_NUM_BRANCHES];
    double line_thermal_stress[POWER_GRID_NUM_BRANCHES];
    unsigned char line_maintenance[POWER_GRID_NUM_BRANCHES];
    unsigned char maintenance_was_closed[POWER_GRID_NUM_BRANCHES];
    float episode_return;
    int pending_reset;
    int ac_power_flow;
    int evaluation_scenarios;
} PowerGrid;

void c_reset(PowerGrid* env);

static const char* const POWER_GRID_EVALUATION_TIMES[POWER_GRID_NUM_PERIODS] = {
    "00:00", "02:00", "04:00", "06:00", "08:00", "10:00",
    "12:00", "14:00", "16:00", "18:00", "20:00", "22:00",
};

static void power_grid_chronological_point(PowerGridOperatingPoint* point, int period) {
    static const double load_scale[POWER_GRID_NUM_PERIODS] = {
        .78, .76, .80, .90, 1.00, 1.08, 1.10, 1.05, 1.12, 1.08, .95, .85,
    };
    static const double solar_mw[POWER_GRID_NUM_PERIODS] = {
        0, 0, 0, 10, 35, 70, 90, 75, 35, 5, 0, 0,
    };
    static const double wind_mw[POWER_GRID_NUM_PERIODS] = {
        35, 30, 28, 25, 20, 18, 22, 28, 35, 40, 42, 38,
    };
    power_grid_operating_point_nominal(point);
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        point->load_mw[load] *= load_scale[period];
    }
    point->generator_mw[2] = solar_mw[period]; /* Synthetic solar at generator bus 3. */
    point->generator_mw[3] = wind_mw[period];  /* Synthetic wind at generator bus 6. */
}

static void power_grid_set_operating_period(PowerGrid* env, int period) {
    if (env->evaluation_scenarios) {
        power_grid_chronological_point(&env->operating_point, period);
    } else {
        power_grid_operating_point_profile(&env->operating_point, env->episode_profiles[period]);
    }
}

static void power_grid_update_maintenance(PowerGrid* env, int period) {
    if (!env->evaluation_scenarios) return;
    const int maintenance_line = 16; /* Connected 9-14 outage during the daytime peak. */
    if (period == 4) {
        env->line_maintenance[maintenance_line] = 1;
        env->maintenance_was_closed[maintenance_line] =
            env->topology.line_closed[maintenance_line];
        env->topology.line_closed[maintenance_line] = 0;
        env->episode.maintenance_events++;
    } else if (period == 8) {
        env->line_maintenance[maintenance_line] = 0;
        if (env->line_available[maintenance_line]) {
            env->topology.line_closed[maintenance_line] =
                env->maintenance_was_closed[maintenance_line];
        }
    }
}

static PowerGridSolveStatus power_grid_solve_environment(PowerGrid* env) {
    if (!env->ac_power_flow) {
        memset(&env->ac_solution, 0, sizeof(env->ac_solution));
        return power_grid_solve(&env->topology, &env->operating_point, &env->solution);
    }
    power_grid_ac_solve(&env->topology, &env->operating_point, &env->ac_solution);
    power_grid_ac_to_compatible(&env->ac_solution, &env->solution);
    return env->solution.status;
}

static double power_grid_branch_rating(const PowerGrid* env, int line) {
    return env->ac_power_flow ? power_grid_ac_branch_rating_mva(line) :
        POWER_GRID_BRANCHES[line].thermal_limit_mw;
}

/* Evaluation-only inverse-time protection. Stress is an intentionally simple
 * thermal proxy: 200% trips in one step, 150% in four, and 120% in about 25.
 * A tripped line is locked out for the rest of the episode. */
static int power_grid_update_ac_protection(PowerGrid* env) {
    if (!env->ac_power_flow || env->solution.status != POWER_GRID_SOLVE_OK) return 0;
    int new_trips = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!env->line_available[line]) continue;
        if (!env->topology.line_closed[line]) {
            env->line_thermal_stress[line] = power_grid_ac_thermal_step(
                env->line_thermal_stress[line], 0.0);
            continue;
        }
        env->line_thermal_stress[line] = power_grid_ac_thermal_step(
            env->line_thermal_stress[line], env->solution.branch_rho[line]);
        if (env->line_thermal_stress[line] > env->episode.peak_thermal_stress) {
            env->episode.peak_thermal_stress = env->line_thermal_stress[line];
        }
        if (env->line_thermal_stress[line] >= POWER_GRID_THERMAL_TRIP_THRESHOLD) {
            env->line_available[line] = 0;
            env->topology.line_closed[line] = 0;
            new_trips++;
        }
    }
    env->episode.thermal_trips += new_trips;
    if (new_trips) power_grid_solve_environment(env);
    return new_trips;
}

static void power_grid_compute_observations(PowerGrid* env) {
    int index = POWER_GRID_LINE_OBS_OFFSET;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        float signed_loading = (float)(env->solution.branch_flow_mw[line] /
            power_grid_branch_rating(env, line));
        float rho = (float)env->solution.branch_rho[line];
        /* Natural per-unit values retain overload severity: 1.0 is exactly the line limit. */
        env->observations[index++] = signed_loading;
        env->observations[index++] = rho;
        env->observations[index++] = env->topology.line_closed[line] ? 1.0f : 0.0f;
    }
    index = POWER_GRID_TERMINAL_OBS_OFFSET;
    /* A bit is the non-redundant one-hot encoding for each two-state busbar category. */
    for (int terminal = 0; terminal < POWER_GRID_NUM_TERMINALS; terminal++) {
        env->observations[index++] = env->topology.terminal_busbar[terminal] ? 1.0f : 0.0f;
    }
    index = POWER_GRID_COUPLER_OBS_OFFSET;
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++) {
        env->observations[index++] = env->topology.coupler_closed[bus] ? 1.0f : 0.0f;
    }
    index = POWER_GRID_INJECTION_OBS_OFFSET;
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++) {
        /* Normalize MW by the IEEE case's 100 MVA system base without clipping. */
        env->observations[index++] = (float)(env->solution.substation_injection_mw[bus] /
            POWER_GRID_BASE_MVA);
    }
}

static void power_grid_finish_episode(PowerGrid* env) {
    if (env->ac_power_flow && env->solution.status != POWER_GRID_SOLVE_OK &&
            (env->ac_solution.status == POWER_GRID_AC_DIVERGED ||
             env->ac_solution.status == POWER_GRID_AC_SINGULAR ||
             env->ac_solution.status == POWER_GRID_AC_NONFINITE)) {
        env->episode.ac_nonconvergence++;
    }
    float steps = env->episode_step > 0 ? (float)env->episode_step : 1.0f;
    float overload_free_fraction = (float)env->episode.safe_steps / steps;
    float max_return = 0.2f * steps;
    int failed = env->solution.status != POWER_GRID_SOLVE_OK;
    float normalized_return = failed ? 0.0f : env->episode_return / max_return;
    normalized_return = fminf(1.0f, fmaxf(0.0f, normalized_return));
    /* perf measures grid security; score measures overall reward efficiency. */
    env->log.perf += overload_free_fraction;
    env->log.overload_free_steps += (float)env->episode.safe_steps;
    if (env->ac_power_flow) {
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
    env->log.score += normalized_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_step;
    env->log.total_failure += failed;
    env->log.topology_failure += env->solution.status == POWER_GRID_INVALID_TOPOLOGY ||
        env->solution.status == POWER_GRID_DISCONNECTED_LOAD ||
        env->solution.status == POWER_GRID_DISCONNECTED_GENERATOR ||
        env->solution.status == POWER_GRID_ISLANDED;
    env->log.solver_failure += env->solution.status == POWER_GRID_SINGULAR ||
        env->solution.status == POWER_GRID_NONFINITE || env->solution.status == POWER_GRID_INVALID_INPUT;
    env->log.total_switches += (float)env->episode.switches[0];
    env->log.line_switches += (float)env->episode.switches[POWER_GRID_ACTION_LINE];
    env->log.busbar_switches += (float)env->episode.switches[POWER_GRID_ACTION_TERMINAL];
    env->log.coupler_switches += (float)env->episode.switches[POWER_GRID_ACTION_COUPLER];
    env->log.n += 1.0f;
    if (env->rendering) {
        env->pending_reset = 1;
        power_grid_compute_observations(env);
    } else {
        c_reset(env);
    }
}

void c_reset(PowerGrid* env) {
    power_grid_topology_normal(&env->topology);
    env->episode_step = 0;
    memset(&env->episode, 0, sizeof(env->episode));
    memset(env->line_available, 1, sizeof(env->line_available));
    memset(env->line_thermal_stress, 0, sizeof(env->line_thermal_stress));
    memset(env->line_maintenance, 0, sizeof(env->line_maintenance));
    memset(env->maintenance_was_closed, 0, sizeof(env->maintenance_was_closed));
    env->pending_reset = 0;
    env->episode_return = 0.0f;
    env->episode_profiles[0] = POWER_GRID_PROFILE_P0_NOMINAL;
    for (int period = 1; period < POWER_GRID_NUM_PERIODS; period++) {
        if (env->evaluation_scenarios) {
            env->episode_profiles[period] = POWER_GRID_PROFILE_P0_NOMINAL;
        } else {
            /* Puffer initializes a separate rng per vector environment. Mix high and low LCG bits. */
            env->rng = env->rng * 1664525u + 1013904223u;
            unsigned int sample = env->rng ^ (env->rng >> 16);
            env->episode_profiles[period] = (PowerGridProfile)(1 +
                sample % (POWER_GRID_NUM_PROFILES - 1));
        }
    }
    env->current_period = 0;
    power_grid_set_operating_period(env, 0);
    power_grid_solve_environment(env);
    power_grid_compute_observations(env);
}

void c_step(PowerGrid* env) {
    if (env->rendering && env->pending_reset) {
        c_reset(env);
        return;
    }
    float raw_action = env->actions[0];
    int action = isfinite(raw_action) ? (int)raw_action : -1;
    PowerGridActionType action_type = power_grid_apply_action(&env->topology, action);
    if (action_type == POWER_GRID_ACTION_LINE) {
        int line = action - POWER_GRID_LINE_ACTION_OFFSET;
        if (!env->line_available[line] || env->line_maintenance[line]) {
            env->topology.line_closed[line] = 0;
        }
    }
    int switched = action_type > POWER_GRID_ACTION_NONE;
    env->episode.switches[0] += switched;
    if (switched) env->episode.switches[action_type]++;
    env->terminals[0] = 0.0f;

    PowerGridSolveStatus status;
    int new_trips = 0;
    if (action_type == POWER_GRID_ACTION_INVALID) {
        status = env->solution.status = POWER_GRID_INVALID_INPUT;
    } else {
        status = power_grid_solve_environment(env);
        if (status == POWER_GRID_SOLVE_OK) new_trips = power_grid_update_ac_protection(env);
        status = env->solution.status;
    }
    env->episode_step++;

    if (status != POWER_GRID_SOLVE_OK) {
        env->rewards[0] = -5.0f;
        env->terminals[0] = 1.0f;
    } else {
        /* Scheduled generator-P violations are exogenous data faults, not agent actions. */
        int safe = env->solution.max_rho <= 1.0 && (!env->ac_power_flow ||
            env->ac_solution.voltage_violation_count == 0);
        double constraint_cost = env->solution.congestion_cost + new_trips;
        if (env->ac_power_flow) constraint_cost += env->ac_solution.voltage_violation_cost;
        float raw_reward = (float)(-constraint_cost - 0.01 * switched +
            0.2 * safe);
        env->rewards[0] = raw_reward < -4.0f ? -4.0f : raw_reward;
        env->episode.safe_steps += safe;
        if (env->ac_power_flow) {
            env->episode.voltage_violation_steps += env->ac_solution.voltage_violation_count > 0;
            env->episode.generator_p_violation_steps +=
                env->ac_solution.generator_p_violation_count > 0;
            env->episode.q_limit_events += env->ac_solution.q_limit_count;
            env->episode.active_loss_mw_sum += env->ac_solution.total_p_loss_mw;
        }
        if (env->episode_step >= POWER_GRID_EPISODE_STEPS) env->terminals[0] = 1.0f;
    }
    env->episode_return += env->rewards[0];

    if (env->terminals[0]) {
        power_grid_finish_episode(env);
        return;
    }

    int next_period = env->episode_step / POWER_GRID_STEPS_PER_PERIOD;
    if (next_period != env->current_period) {
        env->current_period = next_period;
        power_grid_update_maintenance(env, next_period);
        power_grid_set_operating_period(env, next_period);
        status = power_grid_solve_environment(env);
        if (status != POWER_GRID_SOLVE_OK) {
            /* The new injections may expose an AC-infeasible state even though topology is unchanged. */
            env->rewards[0] = -5.0f;
            env->terminals[0] = 1.0f;
            env->episode_return += env->rewards[0];
            power_grid_finish_episode(env);
            return;
        }
    }
    power_grid_compute_observations(env);
}

void power_grid_allocate(PowerGrid* env) {
    env->num_agents = 1;
    env->rng = 0;
    env->observations = calloc(POWER_GRID_OBS_SIZE, sizeof(float));
    env->actions = calloc(1, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(float));
    env->owns_buffers = 1;
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
    {100, 285}, {315, 205}, {570, 135}, {575, 355}, {305, 415}, {455, 620}, {785, 350},
    {1020, 205}, {1020, 485}, {1250, 570}, {870, 635}, {530, 835}, {870, 845}, {1235, 810},
};

static const Vector2 POWER_GRID_LABEL_OFFSETS[POWER_GRID_NUM_BRANCHES] = {
    {30, 25}, {-34, 18}, {30, 0}, {-5, 30}, {-30, 22}, {24, 0}, {0, 24},
    {-24, -27}, {25, 10}, {-28, 5}, {-34, -22}, {-32, 14}, {25, 16}, {0, -25},
    {24, 10}, {18, -23}, {30, 18}, {-27, 16}, {0, -23}, {20, 11},
};

static Color power_grid_line_color(const PowerGrid* env, int line) {
    if (env->ac_power_flow && !env->line_available[line]) return POWER_GRID_OVERLOAD;
    if (env->line_maintenance[line]) return POWER_GRID_WARN;
    if (!env->topology.line_closed[line]) return POWER_GRID_OPEN;
    double rho = env->solution.branch_rho[line];
    if (rho > 1.0) return POWER_GRID_OVERLOAD;
    if (rho > 0.8) return POWER_GRID_WARN;
    return POWER_GRID_SAFE;
}

static Vector2 power_grid_branch_endpoint(Vector2 station, Vector2 other, int busbar) {
    float dx = other.x - station.x;
    float x = station.x;
    if (fabsf(dx) > 12.0f) x += dx > 0.0f ? 54.0f : -54.0f;
    return (Vector2){x, station.y + (busbar ? 8.0f : -8.0f)};
}

static void power_grid_draw_dashed_line(Vector2 from, Vector2 to, float dash, float gap,
        float thickness, Color color) {
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float length = sqrtf(dx * dx + dy * dy);
    if (length <= 0.0f) return;
    dx /= length;
    dy /= length;
    for (float start = 0.0f; start < length; start += dash + gap) {
        float end = fminf(start + dash, length);
        DrawLineEx((Vector2){from.x + dx * start, from.y + dy * start},
            (Vector2){from.x + dx * end, from.y + dy * end}, thickness, color);
    }
}

static void power_grid_draw_flow_arrow(Vector2 from, Vector2 to, double flow, Color color) {
    if (fabs(flow) < 0.05) return;
    if (flow < 0.0) {
        Vector2 tmp = from;
        from = to;
        to = tmp;
    }
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float length = sqrtf(dx * dx + dy * dy);
    if (length < 1.0f) return;
    dx /= length;
    dy /= length;
    Vector2 center = {(from.x + to.x) * 0.5f, (from.y + to.y) * 0.5f};
    Vector2 tip = {center.x + dx * 8.0f, center.y + dy * 8.0f};
    Vector2 left = {center.x - dx * 6.0f - dy * 5.0f, center.y - dy * 6.0f + dx * 5.0f};
    Vector2 right = {center.x - dx * 6.0f + dy * 5.0f, center.y - dy * 6.0f - dx * 5.0f};
    DrawTriangle(tip, left, right, color);
}

static void power_grid_draw_label(Vector2 position, const char* text, Color border) {
    int width = MeasureText(text, 16) + 8;
    Rectangle box = {position.x - width * 0.5f, position.y - 12.0f, (float)width, 24.0f};
    DrawRectangleRounded(box, 0.3f, 4, Fade(POWER_GRID_BG, 0.94f));
    DrawRectangleRoundedLinesEx(box, 0.3f, 4, 1.0f, Fade(border, 0.75f));
    DrawText(text, (int)(box.x + 4), (int)(box.y + 4), 16, POWER_GRID_TEXT);
}

static void power_grid_draw_branches(const PowerGrid* env) {
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        Vector2 from_station = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to_station = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        int from_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 0)];
        int to_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 1)];
        Vector2 from = power_grid_branch_endpoint(from_station, to_station, from_bar);
        Vector2 to = power_grid_branch_endpoint(to_station, from_station, to_bar);
        Color color = power_grid_line_color(env, line);
        DrawLineEx(from, to, 7.0f, Fade(BLACK, 0.65f));
        if (env->topology.line_closed[line]) {
            float width = 2.5f + 2.5f * fminf((float)env->solution.branch_rho[line], 1.2f);
            DrawLineEx(from, to, width, color);
            power_grid_draw_flow_arrow(from, to, env->solution.branch_flow_mw[line], RAYWHITE);
        } else {
            power_grid_draw_dashed_line(from, to, 9.0f, 7.0f, 2.5f, color);
            DrawCircleLines((int)from.x, (int)from.y, 5.0f, POWER_GRID_OPEN);
            DrawCircleLines((int)to.x, (int)to.y, 5.0f, POWER_GRID_OPEN);
        }
    }

    /* Labels are a second pass so they remain legible over crossing conductors. */
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        Vector2 from = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        Vector2 label = {(from.x + to.x) * 0.5f + POWER_GRID_LABEL_OFFSETS[line].x,
            (from.y + to.y) * 0.5f + POWER_GRID_LABEL_OFFSETS[line].y};
        char text[80];
        if (env->topology.line_closed[line]) {
            if (env->ac_power_flow) {
                snprintf(text, sizeof(text), "%s%s  %+.0fMW %.0f/%.0fMVA %.0f%%",
                    branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                    env->ac_solution.branch_from_p_mw[line],
                    fmax(env->ac_solution.branch_from_mva[line],
                        env->ac_solution.branch_to_mva[line]), power_grid_branch_rating(env, line),
                    100.0 * env->solution.branch_rho[line]);
            } else {
                snprintf(text, sizeof(text), "%s%s  %+.0f/%.0fMW  %.0f%%",
                    branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                    env->solution.branch_flow_mw[line], branch->thermal_limit_mw,
                    100.0 * env->solution.branch_rho[line]);
            }
        } else {
            snprintf(text, sizeof(text), "%s  %s", POWER_GRID_BRANCH_NAMES[line],
                env->ac_power_flow && !env->line_available[line] ? "TRIPPED" :
                (env->line_maintenance[line] ? "MAINTENANCE" : "OPEN"));
        }
        power_grid_draw_label(label, text, power_grid_line_color(env, line));
    }
}

static void power_grid_draw_generator(const PowerGrid* env, int generator) {
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
    double output = generator == 0 ? env->solution.slack_generation_mw :
        env->operating_point.generator_mw[generator];
    const char* label = env->ac_power_flow ? TextFormat("%.1f MW  %+.1f MVAr", output,
        env->ac_solution.generator_q_mvar[generator]) : TextFormat("%.1f MW", output);
    DrawText(label, (int)symbol.x + 22, (int)symbol.y - 10,
        env->ac_power_flow ? 16 : 18, POWER_GRID_TEXT);
}

static void power_grid_draw_load(const PowerGrid* env, int load) {
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
    const char* label = env->ac_power_flow ? TextFormat("%.1f MW  %+.1f MVAr",
        env->operating_point.load_mw[load],
        power_grid_ac_load_q_mvar(&env->operating_point, load)) :
        TextFormat("%.1f MW", env->operating_point.load_mw[load]);
    DrawText(label, (int)symbol.x + 20, (int)symbol.y - 10,
        env->ac_power_flow ? 16 : 18, POWER_GRID_TEXT);
}

static void power_grid_draw_stations(const PowerGrid* env) {
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++) {
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
        if (env->topology.coupler_closed[bus]) {
            DrawLineEx((Vector2){position.x, position.y - 8},
                (Vector2){position.x, position.y + 8}, 5.0f, POWER_GRID_SAFE);
        } else {
            DrawCircleLines((int)position.x, (int)position.y - 5, 3.0f, POWER_GRID_OPEN);
            DrawCircleLines((int)position.x, (int)position.y + 5, 3.0f, POWER_GRID_OPEN);
        }
        DrawText("1", (int)position.x - 58, (int)position.y - 16, 12, POWER_GRID_BB1);
        DrawText("2", (int)position.x - 58, (int)position.y + 1, 12, POWER_GRID_BB2);
        double injection = env->solution.substation_injection_mw[bus];
        DrawText(TextFormat("NET %+.0f", injection), (int)position.x + 2,
            (int)position.y - 27, 14, injection >= 0.0 ? POWER_GRID_SAFE : POWER_GRID_MUTED);
    }
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++) {
        power_grid_draw_generator(env, generator);
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) power_grid_draw_load(env, load);
}

static void power_grid_draw_legend_item(int x, int y, Color color, const char* text) {
    DrawCircle(x, y + 7, 5.0f, color);
    DrawText(text, x + 11, y, 15, POWER_GRID_TEXT);
}

static void power_grid_draw_sidebar(const PowerGrid* env) {
    const int x = POWER_GRID_MAP_WIDTH + 20;
    const int width = POWER_GRID_RENDER_WIDTH - x - 20;
    Rectangle panel = {(float)x, 18.0f, (float)width, 974.0f};
    DrawRectangleRounded(panel, 0.03f, 6, POWER_GRID_PANEL);
    DrawRectangleRoundedLinesEx(panel, 0.03f, 6, 1.0f, POWER_GRID_PANEL_EDGE);

    int invalid = env->solution.status != POWER_GRID_SOLVE_OK;
    int overloaded = !invalid && env->solution.max_rho > 1.0;
    Color status_color = invalid || overloaded ? POWER_GRID_OVERLOAD : POWER_GRID_SAFE;
    const char* status = invalid ? power_grid_solve_status_name(env->solution.status) :
        (overloaded ? "OVERLOAD" : "SECURE");
    DrawText("GRID STATUS", x + 18, 36, 20, POWER_GRID_MUTED);
    DrawRectangleRounded((Rectangle){x + 16.0f, 62.0f, width - 32.0f, 48.0f},
        0.18f, 5, Fade(status_color, 0.18f));
    DrawText(status, x + 31, 70, 30, status_color);
    DrawText(TextFormat("max %.1f%%", 100.0 * env->solution.max_rho), x + width - 125,
        76, 20, POWER_GRID_TEXT);

    float current_steps = env->episode_step > 0 ? (float)env->episode_step : 1.0f;
    float current_perf = env->episode.safe_steps / current_steps;
    float current_score = env->episode_return / (0.20f * current_steps);
    if (env->solution.status != POWER_GRID_SOLVE_OK) current_score = 0.0f;
    current_score = fminf(1.0f, fmaxf(0.0f, current_score));

    float episodes = env->log.n;
    float total_steps = env->log.episode_length;
    float safe_steps = env->log.overload_free_steps;
    if (safe_steps > total_steps) safe_steps = total_steps;
    float failed_episode_pct = episodes > 0.0f ? 100.0f * env->log.total_failure / episodes : 0.0f;
    float overloaded_step_pct = total_steps > 0.0f ?
        100.0f * (total_steps - safe_steps) / total_steps : 0.0f;

    DrawText("CURRENT EPISODE", x + 18, 128, 20, POWER_GRID_MUTED);
    DrawText(TextFormat("Step                 %2d / %d", env->episode_step,
        POWER_GRID_EPISODE_STEPS), x + 18, 158, 20, POWER_GRID_TEXT);
    const char* period_name = env->evaluation_scenarios ?
        POWER_GRID_EVALUATION_TIMES[env->current_period] :
        power_grid_profile_name(env->episode_profiles[env->current_period]);
    DrawText(TextFormat("Period               %d  %s", env->current_period, period_name),
        x + 18, 186, 20, POWER_GRID_TEXT);
    DrawText(TextFormat("Score                %.3f", current_score), x + 18, 214, 20,
        POWER_GRID_TEXT);
    DrawText(TextFormat("Perf                 %.3f", current_perf), x + 18, 242, 20,
        POWER_GRID_TEXT);

    DrawText("RUN SUMMARY", x + 18, 286, 20, POWER_GRID_MUTED);
    DrawText(TextFormat("Episodes completed   %.0f", episodes), x + 18, 316, 20,
        POWER_GRID_TEXT);
    DrawText(TextFormat("Failed episodes      %6.2f%%", failed_episode_pct), x + 18, 344, 20,
        failed_episode_pct > 0.0f ? POWER_GRID_OVERLOAD : POWER_GRID_TEXT);
    DrawText(TextFormat("Overload-free steps  %6.2f%%", 100.0f - overloaded_step_pct),
        x + 18, 372, 20, POWER_GRID_SAFE);
    DrawText(TextFormat("Overloaded steps     %6.2f%%", overloaded_step_pct), x + 18, 400, 20,
        overloaded_step_pct > 0.0f ? POWER_GRID_WARN : POWER_GRID_TEXT);

    DrawText("CURRENT HOTSPOTS", x + 18, 444, 20, POWER_GRID_MUTED);
    int used[POWER_GRID_NUM_BRANCHES] = {0};
    for (int rank = 0; rank < 5; rank++) {
        int best = -1;
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
            if (!used[line] && env->topology.line_closed[line] &&
                    (best < 0 || env->solution.branch_rho[line] > env->solution.branch_rho[best])) {
                best = line;
            }
        }
        if (best < 0) break;
        used[best] = 1;
        Color color = power_grid_line_color(env, best);
        DrawRectangle(x + 18, 476 + rank * 31, 5, 22, color);
        double loading = env->ac_power_flow ? fmax(env->ac_solution.branch_from_mva[best],
            env->ac_solution.branch_to_mva[best]) : fabs(env->solution.branch_flow_mw[best]);
        DrawText(TextFormat("%-5s %3.0f/%3.0f %s", POWER_GRID_BRANCH_NAMES[best], loading,
            power_grid_branch_rating(env, best), env->ac_power_flow ? "MVA" : "MW"),
            x + 31, 475 + rank * 31, 18, POWER_GRID_TEXT);
        DrawText(TextFormat("%5.1f%%", 100.0 * env->solution.branch_rho[best]),
            x + width - 86, 475 + rank * 31, 18, color);
    }

    DrawText("COLOR KEY", x + 18, 680, 17, POWER_GRID_MUTED);
    power_grid_draw_legend_item(x + 22, 704, POWER_GRID_SAFE, "safe <80%");
    power_grid_draw_legend_item(x + 180, 704, POWER_GRID_WARN, "near 80-100%");
    power_grid_draw_legend_item(x + 22, 728, POWER_GRID_OVERLOAD, "overload >100%");
    power_grid_draw_legend_item(x + 180, 728, POWER_GRID_OPEN, "open");
    DrawLineEx((Vector2){x + 22, 767}, (Vector2){x + 48, 767}, 5.0f, POWER_GRID_BB1);
    DrawText("busbar 1", x + 56, 758, 15, POWER_GRID_TEXT);
    DrawLineEx((Vector2){x + 180, 767}, (Vector2){x + 206, 767}, 5.0f, POWER_GRID_BB2);
    DrawText("busbar 2", x + 214, 758, 15, POWER_GRID_TEXT);
    DrawLineEx((Vector2){x + 22, 790}, (Vector2){x + 22, 802}, 4.0f, POWER_GRID_SAFE);
    DrawText("coupler closed", x + 34, 790, 15, POWER_GRID_TEXT);

    DrawText(env->ac_power_flow ? "AC VALIDATION" : "DC TRAINING MODEL", x + 18, 828, 18,
        env->ac_power_flow ? POWER_GRID_WARN : POWER_GRID_MUTED);
    if (env->ac_power_flow) {
        DrawText(TextFormat("Voltage       %.3f .. %.3f pu",
            env->ac_solution.min_voltage_pu, env->ac_solution.max_voltage_pu),
            x + 18, 856, 17, POWER_GRID_TEXT);
        DrawText(TextFormat("Losses        %.2f MW  %+.2f MVAr",
            env->ac_solution.total_p_loss_mw, env->ac_solution.total_q_loss_mvar),
            x + 18, 880, 17, POWER_GRID_TEXT);
        DrawText(TextFormat("Q limits      %d converted PV buses", env->ac_solution.q_limit_count),
            x + 18, 904, 17, env->ac_solution.q_limit_count ? POWER_GRID_WARN : POWER_GRID_TEXT);
        DrawText(TextFormat("P violations  %d  (%.1f MW)",
            env->ac_solution.generator_p_violation_count,
            env->ac_solution.generator_p_violation_mw), x + 18, 928, 17,
            env->ac_solution.generator_p_violation_count ? POWER_GRID_OVERLOAD : POWER_GRID_TEXT);
        DrawText(TextFormat("Newton solve  %s, %d iterations",
            power_grid_ac_status_name(env->ac_solution.status), env->ac_solution.iterations),
            x + 18, 952, 17,
            env->ac_solution.status == POWER_GRID_AC_OK ? POWER_GRID_SAFE : POWER_GRID_OVERLOAD);
        DrawText(TextFormat("Protection    %d trips, peak %.2f",
            env->episode.thermal_trips, env->episode.peak_thermal_stress), x + 18, 974, 15,
            env->episode.thermal_trips ? POWER_GRID_OVERLOAD : POWER_GRID_TEXT);
    } else {
        DrawText("Use --env.ac-power-flow True for AC replay", x + 18, 860, 17,
            POWER_GRID_TEXT);
    }
}

void c_render(PowerGrid* env) {
    if (!env->rendering) {
        env->rendering = 1;
        InitWindow(POWER_GRID_RENDER_WIDTH, POWER_GRID_RENDER_HEIGHT,
            "IEEE-14 Power Grid Topology Control");
        SetTargetFPS(POWER_GRID_RENDER_FPS);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground(POWER_GRID_BG);
    DrawText(TextFormat("IEEE-14  |  %s SINGLE-LINE DIAGRAM",
        env->ac_power_flow ? "AC VALIDATION" : "DC"), 22, 14, 32, POWER_GRID_TEXT);
    DrawText(env->ac_power_flow ?
        "Arrows show active-power direction. Loading uses worst-end apparent power (MVA)." :
        "Arrows show active-power direction. Loading uses the DC active-power approximation.",
        22, 52, 20, POWER_GRID_MUTED);
    power_grid_draw_branches(env);
    power_grid_draw_stations(env);
    power_grid_draw_sidebar(env);
    EndDrawing();
}
#else
void c_render(PowerGrid* env) {
    (void)env;
}
#endif

void c_close(PowerGrid* env) {
    if (env->rendering) {
#ifndef POWER_GRID_NO_RENDER
        if (IsWindowReady()) CloseWindow();
#endif
        env->rendering = 0;
    }
    if (env->owns_buffers) {
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
