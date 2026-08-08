#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

const PowerGridBranch POWER_GRID_BRANCHES[POWER_GRID_NUM_BRANCHES] = {
    {0, 1, 0.05917, 1.000, 155}, {0, 4, 0.22304, 1.000, 145},
    {1, 2, 0.19797, 1.000, 130}, {1, 3, 0.17632, 1.000, 80},
    {1, 4, 0.17388, 1.000, 70},  {2, 3, 0.17103, 1.000, 50},
    {3, 4, 0.04211, 1.000, 80},  {3, 6, 0.20912, 0.978, 50},
    {3, 8, 0.55618, 0.969, 35},  {4, 5, 0.25202, 0.932, 65},
    {5, 10, 0.19890, 1.000, 25}, {5, 11, 0.25581, 1.000, 25},
    {5, 12, 0.13027, 1.000, 30}, {6, 7, 0.17615, 1.000, 25},
    {6, 8, 0.11001, 1.000, 50},  {8, 9, 0.08450, 1.000, 20},
    {8, 13, 0.27038, 1.000, 15}, {9, 10, 0.19207, 1.000, 20},
    {11, 12, 0.19988, 1.000, 20},{12, 13, 0.34802, 1.000, 25},
};

const int POWER_GRID_GENERATOR_BUSES[POWER_GRID_NUM_GENERATORS] = {0, 1, 2, 5, 7};
const int POWER_GRID_LOAD_BUSES[POWER_GRID_NUM_LOADS] = {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13};
const char* const POWER_GRID_BRANCH_NAMES[POWER_GRID_NUM_BRANCHES] = {
    "1-2", "1-5", "2-3", "2-4", "2-5", "3-4", "4-5", "4-7", "4-9", "5-6",
    "6-11", "6-12", "6-13", "7-8", "7-9", "9-10", "9-14", "10-11", "12-13", "13-14",
};

typedef struct {
    const char* name;
    double load_scale;
    int source_load;
    int target_load;
    double transfer_mw;
    int generator;
    double generation_mw;
} PowerGridProfileSpec;

static const PowerGridProfileSpec POWER_GRID_PROFILES[POWER_GRID_NUM_PROFILES] = {
    {"P0 nominal",             1.00, -1, -1,  0.0, -1,  0.0},
    {"P1 high load",           1.10, -1, -1,  0.0,  1, 44.0},
    {"P2 bus 14 shift",        1.00,  1, 10, 10.0, -1,  0.0},
    {"P3 bus 4 restoration",   1.00,  1,  2, 45.0, -1,  0.0},
    {"P4 bus 6 load shift",    1.00,  1,  4, 40.0, -1,  0.0},
    {"P5 bus 9 load shift",    1.00,  1,  5, 50.0, -1,  0.0},
    {"P6 bus 10 load shift",   1.00,  1,  6, 20.0, -1,  0.0},
    {"P7 bus 6 generation",    1.00, -1, -1,  0.0,  3, 90.0},
    {"P8 bus 13 load shift",   1.00,  5,  9, 20.0, -1,  0.0},
    {"P9 bus 3 generation",    1.00, -1, -1,  0.0,  2, 170.0},
    {"P10 bus 4 load/gen",     1.00, -1,  2, 60.0,  1, 120.0},
    {"P11 bus 5 load/gen",     1.00, -1,  3, 80.0,  1, 140.0},
    {"P12 bus 11 load/gen",    1.00, -1,  7, 20.0,  3, 60.0},
    {"P13 bus 12 load/gen",    1.00, -1,  8, 30.0,  3, 60.0},
    {"P14 bus 2 local peak",   1.00, -1,  0, 30.0, -1,  0.0},
};

void power_grid_topology_normal(PowerGridTopology* topology) {
    memset(topology, 0, sizeof(*topology));
    memset(topology->line_closed, 1, sizeof(topology->line_closed));
}

PowerGridActionType power_grid_apply_action(PowerGridTopology* topology, int action) {
    if (action < 0 || action >= POWER_GRID_NUM_ACTIONS) {
        return POWER_GRID_ACTION_INVALID;
    }
    if (action == 0) return POWER_GRID_ACTION_NONE;
    if (action < POWER_GRID_TERMINAL_ACTION_OFFSET) {
        topology->line_closed[action - POWER_GRID_LINE_ACTION_OFFSET] ^= 1;
        return POWER_GRID_ACTION_LINE;
    }
    if (action < POWER_GRID_COUPLER_ACTION_OFFSET) {
        topology->terminal_busbar[action - POWER_GRID_TERMINAL_ACTION_OFFSET] ^= 1;
        return POWER_GRID_ACTION_TERMINAL;
    }
    topology->coupler_closed[action - POWER_GRID_COUPLER_ACTION_OFFSET] ^= 1;
    return POWER_GRID_ACTION_COUPLER;
}

int power_grid_terminal_node(const PowerGridTopology* topology, int terminal, int substation) {
    /* An ideal closed coupler makes both physical busbars one electrical node. */
    int busbar = topology->coupler_closed[substation] ? 0 : topology->terminal_busbar[terminal];
    return 2 * substation + busbar;
}

void power_grid_operating_point_nominal(PowerGridOperatingPoint* point) {
    static const double loads[POWER_GRID_NUM_LOADS] = {
        21.7, 94.2, 47.8, 7.6, 11.2, 29.5, 9.0, 3.5, 6.1, 13.5, 14.9,
    };
    memset(point, 0, sizeof(*point));
    memcpy(point->load_mw, loads, sizeof(loads));
    point->generator_mw[1] = 40.0;
}

void power_grid_operating_point_profile(PowerGridOperatingPoint* point, PowerGridProfile profile) {
    const PowerGridProfileSpec* spec = &POWER_GRID_PROFILES[profile];
    power_grid_operating_point_nominal(point);
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        point->load_mw[load] *= spec->load_scale;
    }
    if (spec->source_load >= 0) point->load_mw[spec->source_load] -= spec->transfer_mw;
    if (spec->target_load >= 0) point->load_mw[spec->target_load] += spec->transfer_mw;
    if (spec->generator >= 0) point->generator_mw[spec->generator] = spec->generation_mw;
}

const char* power_grid_profile_name(PowerGridProfile profile) {
    return POWER_GRID_PROFILES[profile].name;
}

static void build_topology_graph(const PowerGridTopology* topology,
        unsigned char active[POWER_GRID_NUM_NODES],
        unsigned char adjacency[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES]) {
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 0),
            line->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 1),
            line->to_bus);
        active[from] = active[to] = 1;
        if (adjacency != NULL) adjacency[from][to] = adjacency[to][from] = 1;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        active[node] = 1;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        active[node] = 1;
    }
}

static void mark_component(
        const unsigned char adjacency[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES],
        const unsigned char active[POWER_GRID_NUM_NODES], int start,
        unsigned char visited[POWER_GRID_NUM_NODES]) {
    int queue[POWER_GRID_NUM_NODES];
    int head = 0, tail = 0;
    visited[start] = 1;
    queue[tail++] = start;
    while (head < tail) {
        int node = queue[head++];
        for (int next = 0; next < POWER_GRID_NUM_NODES; next++) {
            if (active[next] && adjacency[node][next] && !visited[next]) {
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
}

PowerGridSolveStatus power_grid_validate_topology(const PowerGridTopology* topology,
        int* component_count, int* active_node_count) {
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    unsigned char adjacency[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES] = {{0}};
    unsigned char visited[POWER_GRID_NUM_NODES] = {0};
    if (component_count) *component_count = 0;
    if (active_node_count) *active_node_count = 0;
    build_topology_graph(topology, active, adjacency);

    int active_count = 0;
    int components = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) active_count += active[node] != 0;
    for (int start = 0; start < POWER_GRID_NUM_NODES; start++) {
        if (!active[start] || visited[start]) continue;
        components++;
        mark_component(adjacency, active, start, visited);
    }
    if (component_count) *component_count = components;
    if (active_node_count) *active_node_count = active_count;

    int reference = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(0),
        POWER_GRID_GENERATOR_BUSES[0]);
    /* Find exactly the nodes reachable from the slack/reference generator. */
    memset(visited, 0, sizeof(visited));
    mark_component(adjacency, active, reference, visited);
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        if (!visited[node]) return POWER_GRID_DISCONNECTED_LOAD;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        if (!visited[node]) return POWER_GRID_DISCONNECTED_GENERATOR;
    }
    if (components != 1) return POWER_GRID_ISLANDED;
    return POWER_GRID_SOLVE_OK;
}

int power_grid_solve_dense(double* matrix, double* rhs, double* solution,
        int dimensions, int stride) {
    for (int col = 0; col < dimensions; col++) {
        int pivot = col;
        for (int row = col + 1; row < dimensions; row++) {
            if (fabs(matrix[row * stride + col]) > fabs(matrix[pivot * stride + col])) pivot = row;
        }
        if (!isfinite(matrix[pivot * stride + col]) ||
                fabs(matrix[pivot * stride + col]) < 1e-12) return 0;
        if (pivot != col) {
            for (int j = col; j < dimensions; j++) {
                double swap = matrix[col * stride + j];
                matrix[col * stride + j] = matrix[pivot * stride + j];
                matrix[pivot * stride + j] = swap;
            }
            double swap = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = swap;
        }
        for (int row = col + 1; row < dimensions; row++) {
            double factor = matrix[row * stride + col] / matrix[col * stride + col];
            matrix[row * stride + col] = 0.0;
            for (int j = col + 1; j < dimensions; j++) {
                matrix[row * stride + j] -= factor * matrix[col * stride + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for (int row = dimensions - 1; row >= 0; row--) {
        double value = rhs[row];
        for (int col = row + 1; col < dimensions; col++) {
            value -= matrix[row * stride + col] * solution[col];
        }
        solution[row] = value / matrix[row * stride + row];
        if (!isfinite(solution[row])) return 0;
    }
    return 1;
}

PowerGridSolveStatus power_grid_solve(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridSolveResult* result) {
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    int row_for_node[POWER_GRID_NUM_NODES];
    double injection[POWER_GRID_NUM_NODES] = {0};
    double matrix[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES] = {{0}};
    double rhs[POWER_GRID_NUM_NODES] = {0};
    double solution[POWER_GRID_NUM_NODES] = {0};
    memset(result, 0, sizeof(*result));
    result->status = power_grid_validate_topology(topology, &result->component_count,
        &result->active_node_count);
    if (result->status != POWER_GRID_SOLVE_OK) return result->status;

    double total_load = 0.0;
    double non_slack_generation = 0.0;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        if (!isfinite(point->load_mw[load]) || point->load_mw[load] < 0.0) {
            result->status = POWER_GRID_INVALID_INPUT;
            return result->status;
        }
        total_load += point->load_mw[load];
    }
    for (int gen = 1; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        if (!isfinite(point->generator_mw[gen]) || point->generator_mw[gen] < 0.0) {
            result->status = POWER_GRID_INVALID_INPUT;
            return result->status;
        }
        non_slack_generation += point->generator_mw[gen];
    }
    result->slack_generation_mw = total_load - non_slack_generation;
    if (!isfinite(result->slack_generation_mw) || result->slack_generation_mw < 0.0) {
        result->status = POWER_GRID_INVALID_INPUT;
        return result->status;
    }

    build_topology_graph(topology, active, NULL);
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        active[node] = 1;
        double output = gen == 0 ? result->slack_generation_mw : point->generator_mw[gen];
        injection[node] += output;
        result->substation_injection_mw[POWER_GRID_GENERATOR_BUSES[gen]] += output;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        active[node] = 1;
        injection[node] -= point->load_mw[load];
        result->substation_injection_mw[POWER_GRID_LOAD_BUSES[load]] -= point->load_mw[load];
    }

    int reference = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(0),
        POWER_GRID_GENERATOR_BUSES[0]);
    int dimensions = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        row_for_node[node] = -1;
        if (active[node] && node != reference) row_for_node[node] = dimensions++;
    }
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus);
        double susceptance = POWER_GRID_BASE_MVA / (line->reactance * line->tap_ratio);
        int from_row = row_for_node[from];
        int to_row = row_for_node[to];
        if (from_row >= 0) matrix[from_row][from_row] += susceptance;
        if (to_row >= 0) matrix[to_row][to_row] += susceptance;
        if (from_row >= 0 && to_row >= 0) {
            matrix[from_row][to_row] -= susceptance;
            matrix[to_row][from_row] -= susceptance;
        }
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (row_for_node[node] >= 0) rhs[row_for_node[node]] = injection[node];
    }
    if (!power_grid_solve_dense(&matrix[0][0], rhs, solution, dimensions,
            POWER_GRID_NUM_NODES)) {
        result->status = POWER_GRID_SINGULAR;
        return result->status;
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (row_for_node[node] >= 0) result->node_angle[node] = solution[row_for_node[node]];
    }

    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus);
        double flow = POWER_GRID_BASE_MVA * (result->node_angle[from] - result->node_angle[to]) /
            (line->reactance * line->tap_ratio);
        double rho = fabs(flow) / line->thermal_limit_mw;
        if (!isfinite(flow) || !isfinite(rho)) {
            result->status = POWER_GRID_NONFINITE;
            return result->status;
        }
        result->branch_flow_mw[branch] = flow;
        result->branch_rho[branch] = rho;
        if (rho > result->max_rho) result->max_rho = rho;
        if (rho > 1.0) {
            double overload = rho - 1.0;
            result->congestion_cost += overload * overload;
        }
    }
    result->status = POWER_GRID_SOLVE_OK;
    return result->status;
}

const char* power_grid_solve_status_name(PowerGridSolveStatus status) {
    static const char* names[] = {
        "ok", "invalid_topology", "disconnected_load", "disconnected_generator",
        "islanded", "singular", "nonfinite", "invalid_input",
    };
    return names[status];
}

const char* power_grid_action_name(int action, char* buffer, size_t size) {
    if (action == 0) {
        snprintf(buffer, size, "do-nothing");
    } else if (action >= POWER_GRID_LINE_ACTION_OFFSET &&
            action < POWER_GRID_TERMINAL_ACTION_OFFSET) {
        snprintf(buffer, size, "toggle-line-%s",
            POWER_GRID_BRANCH_NAMES[action - POWER_GRID_LINE_ACTION_OFFSET]);
    } else if (action >= POWER_GRID_TERMINAL_ACTION_OFFSET &&
            action < POWER_GRID_COUPLER_ACTION_OFFSET) {
        int terminal = action - POWER_GRID_TERMINAL_ACTION_OFFSET;
        if (terminal < 40) {
            snprintf(buffer, size, "toggle-line-%s-%s-end-busbar",
                POWER_GRID_BRANCH_NAMES[terminal / 2], terminal % 2 == 0 ? "from" : "to");
        } else if (terminal < 45) {
            int gen = terminal - 40;
            snprintf(buffer, size, "toggle-generator-%d-bus-%d-busbar", gen,
                POWER_GRID_GENERATOR_BUSES[gen] + 1);
        } else {
            int load = terminal - 45;
            snprintf(buffer, size, "toggle-load-%d-bus-%d-busbar", load,
                POWER_GRID_LOAD_BUSES[load] + 1);
        }
    } else if (action >= POWER_GRID_COUPLER_ACTION_OFFSET && action < POWER_GRID_NUM_ACTIONS) {
        snprintf(buffer, size, "toggle-substation-%d-coupler",
            action - POWER_GRID_COUPLER_ACTION_OFFSET + 1);
    } else {
        snprintf(buffer, size, "invalid-action-%d", action);
    }
    return buffer;
}
