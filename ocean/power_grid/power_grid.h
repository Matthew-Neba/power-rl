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
    float n;
} Log;

typedef struct {
    int selected_action;
} Client;

typedef struct {
    Log log;
    Client* client;
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
    PowerGridProfile episode_profiles[POWER_GRID_NUM_PERIODS];
    int episode_step;
    int current_period;
    int total_switches;
    int line_switches;
    int busbar_switches;
    int coupler_switches;
    int overload_free_steps;
    float episode_return;
    int pending_reset;
} PowerGrid;

static void power_grid_compute_observations(PowerGrid* env) {
    int index = POWER_GRID_LINE_OBS_OFFSET;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        float signed_loading = (float)(env->solution.branch_flow_mw[line] /
            POWER_GRID_BRANCHES[line].thermal_limit_mw);
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

static void power_grid_add_log(PowerGrid* env) {
    float steps = env->episode_step > 0 ? (float)env->episode_step : 1.0f;
    float overload_free_fraction = (float)env->overload_free_steps / steps;
    float max_return = 0.2f * steps;
    int failed = env->solution.status != POWER_GRID_SOLVE_OK;
    float normalized_return = failed ? 0.0f : env->episode_return / max_return;
    if (normalized_return < 0.0f) normalized_return = 0.0f;
    if (normalized_return > 1.0f) normalized_return = 1.0f;
    /* perf measures grid security; score measures overall reward efficiency. */
    env->log.perf += overload_free_fraction;
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
    env->log.total_switches += (float)env->total_switches;
    env->log.line_switches += (float)env->line_switches;
    env->log.busbar_switches += (float)env->busbar_switches;
    env->log.coupler_switches += (float)env->coupler_switches;
    env->log.n += 1.0f;
}

void c_reset(PowerGrid* env) {
    power_grid_topology_normal(&env->topology);
    env->episode_step = 0;
    env->total_switches = 0;
    env->line_switches = 0;
    env->busbar_switches = 0;
    env->coupler_switches = 0;
    env->overload_free_steps = 0;
    env->pending_reset = 0;
    env->episode_return = 0.0f;
    env->episode_profiles[0] = POWER_GRID_PROFILE_P0_NOMINAL;
    for (int period = 1; period < POWER_GRID_NUM_PERIODS; period++) {
        /* Puffer initializes a separate rng per vector environment. Mix high and low LCG bits. */
        env->rng = env->rng * 1664525u + 1013904223u;
        unsigned int sample = env->rng ^ (env->rng >> 16);
        env->episode_profiles[period] = (PowerGridProfile)(1 +
            sample % (POWER_GRID_NUM_PROFILES - 1));
    }
    env->current_period = 0;
    power_grid_operating_point_profile(&env->operating_point, env->episode_profiles[0]);
    power_grid_solve(&env->topology, &env->operating_point, &env->solution);
    power_grid_compute_observations(env);
}

void c_step(PowerGrid* env) {
    if (env->client != NULL && env->pending_reset) {
        c_reset(env);
        return;
    }
    float raw_action = env->actions[0];
    int action = isfinite(raw_action) ? (int)raw_action : -1;
    PowerGridActionType action_type = power_grid_apply_action(&env->topology, action);
    int switched = action_type > POWER_GRID_ACTION_NONE;
    env->line_switches += action_type == POWER_GRID_ACTION_LINE;
    env->busbar_switches += action_type == POWER_GRID_ACTION_TERMINAL;
    env->coupler_switches += action_type == POWER_GRID_ACTION_COUPLER;
    env->total_switches += switched;
    env->terminals[0] = 0.0f;

    PowerGridSolveStatus status = action_type == POWER_GRID_ACTION_INVALID ?
        POWER_GRID_INVALID_INPUT :
        power_grid_solve(&env->topology, &env->operating_point, &env->solution);
    if (action_type == POWER_GRID_ACTION_INVALID) env->solution.status = status;
    env->episode_step++;

    if (status != POWER_GRID_SOLVE_OK) {
        env->rewards[0] = -5.0f;
        env->terminals[0] = 1.0f;
    } else {
        int safe = env->solution.max_rho <= 1.0;
        float raw_reward = (float)(-env->solution.congestion_cost - 0.01 * switched +
            0.2 * safe);
        env->rewards[0] = raw_reward < -4.0f ? -4.0f : raw_reward;
        env->overload_free_steps += safe;
        if (env->episode_step >= POWER_GRID_EPISODE_STEPS) env->terminals[0] = 1.0f;
    }
    env->episode_return += env->rewards[0];

    if (env->terminals[0]) {
        power_grid_add_log(env);
        if (env->client != NULL) {
            env->pending_reset = 1;
            power_grid_compute_observations(env);
        } else {
            c_reset(env);
        }
        return;
    }

    int next_period = env->episode_step / POWER_GRID_STEPS_PER_PERIOD;
    if (next_period != env->current_period) {
        env->current_period = next_period;
        power_grid_operating_point_profile(&env->operating_point,
            env->episode_profiles[next_period]);
        status = power_grid_solve(&env->topology, &env->operating_point, &env->solution);
        if (status != POWER_GRID_SOLVE_OK) {
            /* Connectivity does not change with injections, so this indicates bad scenario data. */
            env->rewards[0] = -5.0f;
            env->terminals[0] = 1.0f;
            env->episode_return += env->rewards[0];
            power_grid_add_log(env);
            c_reset(env);
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

void c_close(PowerGrid* env) {
    if (env->client != NULL) {
#ifndef POWER_GRID_NO_RENDER
        if (IsWindowReady()) CloseWindow();
#endif
        free(env->client);
        env->client = NULL;
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

#ifndef POWER_GRID_NO_RENDER
#define POWER_GRID_RENDER_WIDTH 1800
#define POWER_GRID_RENDER_HEIGHT 1000
#define POWER_GRID_MAP_WIDTH 1400

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
    if (!env->topology.line_closed[line]) return POWER_GRID_OPEN;
    double rho = env->solution.branch_rho[line];
    if (rho > 1.0) return POWER_GRID_OVERLOAD;
    if (rho > 0.8) return POWER_GRID_WARN;
    return POWER_GRID_SAFE;
}

static int power_grid_selected_terminal(const PowerGrid* env) {
    int action = env->client->selected_action;
    if (action < POWER_GRID_TERMINAL_ACTION_OFFSET ||
            action >= POWER_GRID_COUPLER_ACTION_OFFSET) return -1;
    return action - POWER_GRID_TERMINAL_ACTION_OFFSET;
}

static int power_grid_terminal_substation(int terminal) {
    if (terminal < 0 || terminal >= POWER_GRID_NUM_TERMINALS) return -1;
    if (terminal < 40) {
        int branch = terminal / 2;
        return terminal % 2 == 0 ? POWER_GRID_BRANCHES[branch].from_bus :
            POWER_GRID_BRANCHES[branch].to_bus;
    }
    if (terminal < 45) return POWER_GRID_GENERATOR_BUSES[terminal - 40];
    return POWER_GRID_LOAD_BUSES[terminal - 45];
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
    int selected_action = env->client->selected_action;
    int selected_terminal = power_grid_selected_terminal(env);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        Vector2 from_station = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to_station = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        int from_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 0)];
        int to_bar = env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 1)];
        Vector2 from = power_grid_branch_endpoint(from_station, to_station, from_bar);
        Vector2 to = power_grid_branch_endpoint(to_station, from_station, to_bar);
        Color color = power_grid_line_color(env, line);
        int selected = selected_action == line + 1 || selected_terminal == 2 * line ||
            selected_terminal == 2 * line + 1;
        if (selected) DrawLineEx(from, to, 10.0f, Fade(RAYWHITE, 0.28f));
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
            snprintf(text, sizeof(text), "%s%s  %+.0f/%.0fMW  %.0f%%",
                branch->tap_ratio != 1.0 ? "T " : "", POWER_GRID_BRANCH_NAMES[line],
                env->solution.branch_flow_mw[line], branch->thermal_limit_mw,
                100.0 * env->solution.branch_rho[line]);
        } else {
            snprintf(text, sizeof(text), "%s  OPEN", POWER_GRID_BRANCH_NAMES[line]);
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
    int selected = power_grid_selected_terminal(env) == terminal;
    DrawLineEx(busbar, (Vector2){symbol.x, symbol.y + 13.0f}, 2.0f,
        bar ? POWER_GRID_BB2 : POWER_GRID_BB1);
    if (selected) DrawCircleV(symbol, 21.0f, Fade(RAYWHITE, 0.35f));
    DrawCircleV(symbol, 17.0f, (Color){34, 78, 100, 255});
    DrawCircleLines((int)symbol.x, (int)symbol.y, 17.0f, POWER_GRID_BB1);
    DrawText("G", (int)symbol.x - 7, (int)symbol.y - 11, 21, RAYWHITE);
    double output = generator == 0 ? env->solution.slack_generation_mw :
        env->operating_point.generator_mw[generator];
    DrawText(TextFormat("%.1f MW", output), (int)symbol.x + 22, (int)symbol.y - 10, 18,
        POWER_GRID_TEXT);
}

static void power_grid_draw_load(const PowerGrid* env, int load) {
    int bus = POWER_GRID_LOAD_BUSES[load];
    int terminal = POWER_GRID_LOAD_TERMINAL(load);
    int bar = env->topology.terminal_busbar[terminal];
    Vector2 station = POWER_GRID_STATION_POSITIONS[bus];
    Vector2 busbar = {station.x + 19.0f, station.y + (bar ? 8.0f : -8.0f)};
    Vector2 symbol = {station.x + 19.0f, station.y + 49.0f};
    int selected = power_grid_selected_terminal(env) == terminal;
    DrawLineEx(busbar, (Vector2){symbol.x, symbol.y - 12.0f}, 2.0f,
        bar ? POWER_GRID_BB2 : POWER_GRID_BB1);
    if (selected) DrawCircleV(symbol, 21.0f, Fade(RAYWHITE, 0.35f));
    Rectangle load_box = {symbol.x - 15.0f, symbol.y - 13.0f, 30.0f, 26.0f};
    DrawRectangleRounded(load_box, 0.25f, 4, (Color){178, 77, 50, 255});
    DrawRectangleRoundedLinesEx(load_box, 0.25f, 4, 1.0f, (Color){255, 154, 105, 255});
    DrawText("L", (int)symbol.x - 6, (int)symbol.y - 10, 20, RAYWHITE);
    DrawText(TextFormat("%.1f MW", env->operating_point.load_mw[load]),
        (int)symbol.x + 20, (int)symbol.y - 10, 18, POWER_GRID_TEXT);
}

static void power_grid_draw_stations(const PowerGrid* env) {
    int selected_station = power_grid_terminal_substation(power_grid_selected_terminal(env));
    int action = env->client->selected_action;
    if (action >= POWER_GRID_COUPLER_ACTION_OFFSET && action < POWER_GRID_NUM_ACTIONS) {
        selected_station = action - POWER_GRID_COUPLER_ACTION_OFFSET;
    }
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++) {
        Vector2 position = POWER_GRID_STATION_POSITIONS[bus];
        Rectangle panel = {position.x - 60.0f, position.y - 29.0f, 120.0f, 58.0f};
        int selected = selected_station == bus;
        DrawRectangleRounded(panel, 0.18f, 4, POWER_GRID_PANEL);
        DrawRectangleRoundedLinesEx(panel, 0.18f, 4, selected ? 3.0f : 1.0f,
            selected ? RAYWHITE : POWER_GRID_PANEL_EDGE);
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
    Rectangle panel = {(float)x, 18.0f, (float)width, 964.0f};
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

    double total_load = 0.0;
    double total_generation = env->solution.slack_generation_mw;
    for (int i = 0; i < POWER_GRID_NUM_LOADS; i++) total_load += env->operating_point.load_mw[i];
    for (int i = 1; i < POWER_GRID_NUM_GENERATORS; i++) {
        total_generation += env->operating_point.generator_mw[i];
    }
    DrawText("OPERATING POINT", x + 18, 128, 20, POWER_GRID_MUTED);
    DrawText(TextFormat("Step                 %2d / %d", env->episode_step,
        POWER_GRID_EPISODE_STEPS), x + 18, 158, 20, POWER_GRID_TEXT);
    DrawText(TextFormat("Period               %d  %s", env->current_period,
        power_grid_profile_name(env->episode_profiles[env->current_period])),
        x + 18, 186, 20, POWER_GRID_TEXT);
    DrawText(TextFormat("Demand               %7.1f MW", total_load), x + 18, 214, 20,
        POWER_GRID_TEXT);
    DrawText(TextFormat("Generation           %7.1f MW", total_generation), x + 18, 242, 20,
        POWER_GRID_TEXT);
    DrawText(TextFormat("Slack bus 1          %7.1f MW", env->solution.slack_generation_mw),
        x + 18, 270, 20, POWER_GRID_TEXT);
    DrawText(TextFormat("Congestion cost      %9.5f", env->solution.congestion_cost),
        x + 18, 298, 20, POWER_GRID_TEXT);
    DrawText(TextFormat("Switches             %7d", env->total_switches), x + 18, 326, 20,
        POWER_GRID_TEXT);

    DrawText("MOST LOADED BRANCHES", x + 18, 366, 20, POWER_GRID_MUTED);
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
        DrawRectangle(x + 18, 398 + rank * 31, 5, 22, color);
        DrawText(TextFormat("%-5s %3.0f/%3.0f MW", POWER_GRID_BRANCH_NAMES[best],
            fabs(env->solution.branch_flow_mw[best]), POWER_GRID_BRANCHES[best].thermal_limit_mw),
            x + 31, 397 + rank * 31, 18,
            POWER_GRID_TEXT);
        DrawText(TextFormat("%5.1f%%", 100.0 * env->solution.branch_rho[best]),
            x + width - 86, 397 + rank * 31, 18, color);
    }

    DrawText("COLOR KEY", x + 18, 568, 17, POWER_GRID_MUTED);
    power_grid_draw_legend_item(x + 22, 592, POWER_GRID_SAFE, "safe <80%");
    power_grid_draw_legend_item(x + 180, 592, POWER_GRID_WARN, "near 80-100%");
    power_grid_draw_legend_item(x + 22, 616, POWER_GRID_OVERLOAD, "overload >100%");
    power_grid_draw_legend_item(x + 180, 616, POWER_GRID_OPEN, "open");
    DrawLineEx((Vector2){x + 22, 655}, (Vector2){x + 48, 655}, 5.0f, POWER_GRID_BB1);
    DrawText("busbar 1", x + 56, 646, 15, POWER_GRID_TEXT);
    DrawLineEx((Vector2){x + 180, 655}, (Vector2){x + 206, 655}, 5.0f, POWER_GRID_BB2);
    DrawText("busbar 2", x + 214, 646, 15, POWER_GRID_TEXT);
    DrawLineEx((Vector2){x + 22, 678}, (Vector2){x + 22, 690}, 4.0f, POWER_GRID_SAFE);
    DrawText("coupler closed", x + 34, 678, 15, POWER_GRID_TEXT);

    char action_name[160];
    power_grid_action_name(env->client->selected_action, action_name, sizeof(action_name));
    DrawText("SELECTED CONTROL", x + 18, 710, 20, POWER_GRID_MUTED);
    DrawRectangleRounded((Rectangle){x + 16.0f, 740.0f, width - 32.0f, 66.0f},
        0.12f, 5, (Color){29, 40, 58, 255});
    DrawText(TextFormat("%02d", env->client->selected_action), x + 28, 756, 27, POWER_GRID_WARN);
    DrawText(action_name, x + 78, 762, 15, POWER_GRID_TEXT);
    DrawText("LEFT/RIGHT +/-1   UP/DOWN +/-10", x + 18, 836, 17,
        POWER_GRID_MUTED);
    DrawText("SPACE apply       R reset", x + 18, 864, 17,
        POWER_GRID_MUTED);
    if (env->pending_reset) {
        DrawText("Terminal state shown - SPACE resets", x + 18, 884, 18,
            POWER_GRID_OVERLOAD);
    }
}

void c_render(PowerGrid* env) {
    if (env->client == NULL) {
        env->client = calloc(1, sizeof(Client));
        InitWindow(POWER_GRID_RENDER_WIDTH, POWER_GRID_RENDER_HEIGHT,
            "IEEE-14 Power Grid Topology Control");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground(POWER_GRID_BG);
    DrawText("IEEE-14  |  LIVE SINGLE-LINE DIAGRAM", 22, 14, 32, POWER_GRID_TEXT);
    DrawText("Arrows show active-power direction. Branch labels show signed MW and thermal loading.",
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

#endif
