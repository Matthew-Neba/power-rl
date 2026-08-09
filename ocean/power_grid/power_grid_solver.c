#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdatomic.h>
#include <string.h>

static const char* const POWER_GRID_PROFILE_NAMES[POWER_GRID_NUM_PROFILES] = {
    "P0 nominal", "P1 high load", "P2 low load", "P3 region A peak",
    "P4 region B peak", "P5 region C peak", "P6 region D peak",
    "P7 A-to-D shift", "P8 D-to-A shift", "P9 high wind", "P10 high solar",
};

static int generator_for_bus(int bus)
{
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
        if (POWER_GRID_GENERATOR_BUSES[generator] == bus)
            return generator;
    return -1;
}

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
    memset(point, 0, sizeof(*point));
    memcpy(point->load_mw, POWER_GRID_LOAD_P_NOMINAL, sizeof(point->load_mw));
    memcpy(point->generator_mw, POWER_GRID_GENERATOR_P_NOMINAL,
           sizeof(point->generator_mw));
}

void power_grid_operating_point_profile(PowerGridOperatingPoint* point, PowerGridProfile profile) {
    power_grid_operating_point_nominal(point);
    if (profile == POWER_GRID_PROFILE_P1_HIGH || profile == POWER_GRID_PROFILE_P2_LOW)
    {
        double scale = profile == POWER_GRID_PROFILE_P1_HIGH ? 1.15 : 0.95;
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            point->load_mw[load] *= scale;
    }
    else if (profile >= POWER_GRID_PROFILE_P3_REGION_A_PEAK &&
             profile <= POWER_GRID_PROFILE_P6_REGION_D_PEAK)
    {
        int region = profile - POWER_GRID_PROFILE_P3_REGION_A_PEAK;
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            if (POWER_GRID_LOAD_BUSES[load] / 30 == region)
                point->load_mw[load] *= 1.30;
    }
    else if (profile == POWER_GRID_PROFILE_P7_A_TO_D_SHIFT ||
             profile == POWER_GRID_PROFILE_P8_D_TO_A_SHIFT)
    {
        int source = profile == POWER_GRID_PROFILE_P7_A_TO_D_SHIFT ? 0 : 3;
        int target = 3 - source;
        double removed = 0.0, target_total = 0.0;
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        {
            int region = POWER_GRID_LOAD_BUSES[load] / 30;
            if (region == source)
            {
                double transfer = 0.20 * point->load_mw[load];
                point->load_mw[load] -= transfer;
                removed += transfer;
            }
            if (region == target)
                target_total += point->load_mw[load];
        }
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            if (POWER_GRID_LOAD_BUSES[load] / 30 == target)
                point->load_mw[load] += removed * point->load_mw[load] / target_total;
    }
    else if (profile == POWER_GRID_PROFILE_P9_HIGH_WIND ||
             profile == POWER_GRID_PROFILE_P10_HIGH_SOLAR)
    {
        int bus = profile == POWER_GRID_PROFILE_P9_HIGH_WIND ?
                  POWER_GRID_WIND_GENERATOR_BUS : POWER_GRID_SOLAR_GENERATOR_BUS;
        int generator = generator_for_bus(bus);
        if (generator >= 0)
            point->generator_mw[generator] = profile == POWER_GRID_PROFILE_P9_HIGH_WIND ?
                POWER_GRID_WIND_NAMEPLATE_MW : POWER_GRID_SOLAR_NAMEPLATE_MW;
    }
}

const char* power_grid_profile_name(PowerGridProfile profile) {
    return POWER_GRID_PROFILE_NAMES[profile];
}

int power_grid_random_event_eligible(int line)
{
    return line >= 0 && line < POWER_GRID_NUM_BRANCHES &&
           POWER_GRID_RANDOM_EVENT_ELIGIBLE[line];
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

/* Training uses DC flows; float32 is sufficient for IEEE-118 branch/rating
 * scales and halves the nodal workspace/bandwidth. AC evaluation keeps its
 * independent double-precision Newton solve in power_grid_ac.c. */
static int power_grid_solve_dense_float(float* restrict matrix, float* restrict rhs,
        float* restrict solution,
        int dimensions, int stride)
{
    for (int col = 0; col < dimensions; col++) {
        int pivot = col;
        for (int row = col + 1; row < dimensions; row++)
            if (fabsf(matrix[row * stride + col]) > fabsf(matrix[pivot * stride + col])) pivot = row;
        if (!isfinite(matrix[pivot * stride + col]) || fabsf(matrix[pivot * stride + col]) < 1e-12f)
            return 0;
        if (pivot != col) {
            for (int j = col; j < dimensions; j++) {
                float swap = matrix[col * stride + j];
                matrix[col * stride + j] = matrix[pivot * stride + j];
                matrix[pivot * stride + j] = swap;
            }
            float swap = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = swap;
        }
        for (int row = col + 1; row < dimensions; row++) {
            float factor = matrix[row * stride + col] / matrix[col * stride + col];
            matrix[row * stride + col] = 0.0f;
            for (int j = col + 1; j < dimensions; j++)
                matrix[row * stride + j] -= factor * matrix[col * stride + j];
            rhs[row] -= factor * rhs[col];
        }
    }
    for (int row = dimensions - 1; row >= 0; row--) {
        float value = rhs[row];
        for (int col = row + 1; col < dimensions; col++)
            value -= matrix[row * stride + col] * solution[col];
        solution[row] = value / matrix[row * stride + row];
        if (!isfinite(solution[row])) return 0;
    }
    return 1;
}

typedef struct {
    int dimensions;
    int pivot[POWER_GRID_NUM_NODES];
    float factor[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
} PowerGridNormalFactor;

static PowerGridNormalFactor power_grid_normal_factor;
static atomic_int power_grid_normal_factor_state;

static int power_grid_is_normal_topology(const PowerGridTopology* topology)
{
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        if (!topology->line_closed[line]) return 0;
    for (int terminal = 0; terminal < POWER_GRID_NUM_TERMINALS; terminal++)
        if (topology->terminal_busbar[terminal]) return 0;
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
        if (topology->coupler_closed[bus]) return 0;
    return 1;
}

static int power_grid_factor_normal(float matrix[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES],
        int dimensions)
{
    memcpy(power_grid_normal_factor.factor, matrix, sizeof(power_grid_normal_factor.factor));
    for (int col = 0; col < dimensions; col++) {
        int pivot = col;
        for (int row = col + 1; row < dimensions; row++)
            if (fabsf(power_grid_normal_factor.factor[row][col]) >
                fabsf(power_grid_normal_factor.factor[pivot][col])) pivot = row;
        if (!isfinite(power_grid_normal_factor.factor[pivot][col]) ||
            fabsf(power_grid_normal_factor.factor[pivot][col]) < 1e-12f) return 0;
        power_grid_normal_factor.pivot[col] = pivot;
        if (pivot != col)
            for (int j = 0; j < dimensions; j++) {
                float swap = power_grid_normal_factor.factor[col][j];
                power_grid_normal_factor.factor[col][j] = power_grid_normal_factor.factor[pivot][j];
                power_grid_normal_factor.factor[pivot][j] = swap;
            }
        for (int row = col + 1; row < dimensions; row++) {
            power_grid_normal_factor.factor[row][col] /= power_grid_normal_factor.factor[col][col];
            for (int j = col + 1; j < dimensions; j++)
                power_grid_normal_factor.factor[row][j] -=
                    power_grid_normal_factor.factor[row][col] * power_grid_normal_factor.factor[col][j];
        }
    }
    power_grid_normal_factor.dimensions = dimensions;
    return 1;
}

static int power_grid_solve_normal(float* rhs, float* solution, int dimensions)
{
    for (int col = 0; col < dimensions; col++) {
        int pivot = power_grid_normal_factor.pivot[col];
        if (pivot != col) {
            float swap = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = swap;
        }
    }
    for (int row = 0; row < dimensions; row++) {
        float value = rhs[row];
        for (int col = 0; col < row; col++)
            value -= power_grid_normal_factor.factor[row][col] * solution[col];
        solution[row] = value;
    }
    for (int row = dimensions - 1; row >= 0; row--) {
        float value = solution[row];
        for (int col = row + 1; col < dimensions; col++)
            value -= power_grid_normal_factor.factor[row][col] * solution[col];
        solution[row] = value / power_grid_normal_factor.factor[row][row];
        if (!isfinite(solution[row])) return 0;
    }
    return 1;
}

static int power_grid_ensure_normal_factor(
        float matrix[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES], int dimensions)
{
    int state = atomic_load_explicit(&power_grid_normal_factor_state, memory_order_acquire);
    if (state == 0) {
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(&power_grid_normal_factor_state,
                &expected, 1, memory_order_acq_rel, memory_order_acquire)) {
            int valid = power_grid_factor_normal(matrix, dimensions);
            atomic_store_explicit(&power_grid_normal_factor_state, valid ? 2 : -1,
                memory_order_release);
            state = valid ? 2 : -1;
        } else state = expected;
    }
    while (state == 1)
        state = atomic_load_explicit(&power_grid_normal_factor_state, memory_order_acquire);
    return state == 2 && power_grid_normal_factor.dimensions == dimensions;
}

PowerGridSolveStatus power_grid_solve(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridSolveResult* result) {
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    int row_for_node[POWER_GRID_NUM_NODES];
    double injection[POWER_GRID_NUM_NODES] = {0};
    float matrix[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    float rhs[POWER_GRID_NUM_NODES] = {0};
    float solution[POWER_GRID_NUM_NODES] = {0};
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
    int normal_topology = power_grid_is_normal_topology(topology);
    int normal_factor_ready = normal_topology &&
        atomic_load_explicit(&power_grid_normal_factor_state, memory_order_acquire) == 2 &&
        power_grid_normal_factor.dimensions == dimensions;
    if (!normal_factor_ready) {
        memset(matrix, 0, sizeof(matrix));
        for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
            if (!topology->line_closed[branch]) continue;
            const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
            int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus);
            int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus);
            float susceptance = (float)(POWER_GRID_BASE_MVA / (line->reactance * line->tap_ratio));
            int from_row = row_for_node[from];
            int to_row = row_for_node[to];
            if (from_row >= 0) matrix[from_row][from_row] += susceptance;
            if (to_row >= 0) matrix[to_row][to_row] += susceptance;
            if (from_row >= 0 && to_row >= 0) {
                matrix[from_row][to_row] -= susceptance;
                matrix[to_row][from_row] -= susceptance;
            }
        }
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (row_for_node[node] >= 0) rhs[row_for_node[node]] = (float)injection[node];
    }
    int solved = 0;
    if (normal_topology &&
        power_grid_ensure_normal_factor(matrix, dimensions))
        solved = power_grid_solve_normal(rhs, solution, dimensions);
    else
        solved = power_grid_solve_dense_float(&matrix[0][0], rhs, solution, dimensions,
            POWER_GRID_NUM_NODES);
    if (!solved) {
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
        if (terminal < 2 * POWER_GRID_NUM_BRANCHES) {
            snprintf(buffer, size, "toggle-line-%s-%s-end-busbar",
                POWER_GRID_BRANCH_NAMES[terminal / 2], terminal % 2 == 0 ? "from" : "to");
        } else if (terminal < 2 * POWER_GRID_NUM_BRANCHES + POWER_GRID_NUM_GENERATORS) {
            int gen = terminal - 2 * POWER_GRID_NUM_BRANCHES;
            snprintf(buffer, size, "toggle-generator-%d-bus-%d-busbar", gen,
                POWER_GRID_GENERATOR_BUSES[gen] + 1);
        } else {
            int load = terminal - 2 * POWER_GRID_NUM_BRANCHES - POWER_GRID_NUM_GENERATORS;
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
