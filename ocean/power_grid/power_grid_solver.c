#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const char* const POWER_GRID_PROFILE_NAMES[POWER_GRID_NUM_PROFILES] = {
    "P0 nominal", "P1 high load", "P2 low load", "P3 region A peak",
    "P4 region B peak", "P5 region C peak", "P6 region D peak",
    "P7 A-to-D shift", "P8 D-to-A shift", "P9 high wind", "P10 high solar",
};

/* A fixed minimum-degree ordering for the IEEE-118 bus graph. Applying the
 * same bus ordering to both busbars preserves exact DC solutions while
 * substantially reducing sparse-Cholesky fill for all valid topologies. */
static const int POWER_GRID_BUS_ORDER[POWER_GRID_NUM_SUBSTATIONS] = {
    9, 8, 72, 86, 85, 110, 111, 115, 116, 0, 1, 2, 3, 5, 6, 7, 12,
    13, 15, 17, 19, 20, 21, 25, 27, 28, 32, 34, 35, 38, 40, 39, 41,
    42, 43, 47, 49, 51, 52, 56, 57, 62, 66, 70, 71, 23, 73, 75, 77,
    78, 80, 83, 82, 87, 84, 89, 90, 88, 92, 94, 96, 97, 98, 100, 101,
    106, 105, 107, 108, 109, 112, 113, 114, 117, 4, 10, 11, 24, 30,
    31, 26, 33, 37, 45, 46, 50, 54, 53, 55, 59, 60, 58, 61, 63, 65,
    67, 69, 74, 81, 93, 91, 95, 102, 103, 104, 99, 76, 79, 14, 16,
    18, 29, 22, 36, 44, 48, 64, 68,
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

static PowerGridSolveStatus power_grid_validate_topology_internal(
        const PowerGridTopology* topology, int* component_count,
        int* active_node_count, unsigned char* active_out) {
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    int parent[POWER_GRID_NUM_NODES];
    int rank[POWER_GRID_NUM_NODES] = {0};
    if (component_count) *component_count = 0;
    if (active_node_count) *active_node_count = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) parent[node] = node;

    /* Union-find is equivalent to the old adjacency/BFS validation here, but
     * touches only the 186 closed branches instead of a 236x236 matrix. */
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 0),
            line->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(branch, 1),
            line->to_bus);
        active[from] = active[to] = 1;
        int root_from = from, root_to = to;
        while (parent[root_from] != root_from) root_from = parent[root_from];
        while (parent[root_to] != root_to) root_to = parent[root_to];
        while (parent[from] != from) { int next = parent[from]; parent[from] = root_from; from = next; }
        while (parent[to] != to) { int next = parent[to]; parent[to] = root_to; to = next; }
        if (root_from != root_to) {
            if (rank[root_from] < rank[root_to]) parent[root_from] = root_to;
            else {
                parent[root_to] = root_from;
                if (rank[root_from] == rank[root_to]) rank[root_from]++;
            }
        }
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        active[power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen])] = 1;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        active[power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load])] = 1;
    }

    int active_count = 0;
    int components = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) active_count += active[node] != 0;
    if (active_out) memcpy(active_out, active, sizeof(active));
    for (int start = 0; start < POWER_GRID_NUM_NODES; start++) {
        if (!active[start]) continue;
        int root = start;
        while (parent[root] != root) root = parent[root];
        if (root != start) continue;
        components++;
    }
    if (component_count) *component_count = components;
    if (active_node_count) *active_node_count = active_count;

    int reference = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(0),
        POWER_GRID_GENERATOR_BUSES[0]);
    int reference_root = reference;
    while (parent[reference_root] != reference_root) reference_root = parent[reference_root];
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        int root = node;
        while (parent[root] != root) root = parent[root];
        if (root != reference_root) return POWER_GRID_DISCONNECTED_LOAD;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        int root = node;
        while (parent[root] != root) root = parent[root];
        if (root != reference_root) return POWER_GRID_DISCONNECTED_GENERATOR;
    }
    if (components != 1) return POWER_GRID_ISLANDED;
    return POWER_GRID_SOLVE_OK;
}

PowerGridSolveStatus power_grid_validate_topology(const PowerGridTopology* topology,
        int* component_count, int* active_node_count) {
    return power_grid_validate_topology_internal(topology, component_count,
        active_node_count, NULL);
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

/* A DC topology changes much less often than injections do. Keep one exact
 * double-precision LU factorization per worker thread and reuse it whenever
 * the complete topology is unchanged. The cache is deliberately private to
 * this translation unit: direct solver callers retain the same API and no
 * mutable state is added to each environment instance. */
typedef struct {
    int valid;
    int dimensions;
    int row_for_node[POWER_GRID_NUM_NODES];
    int component_count;
    int active_node_count;
    unsigned char active[POWER_GRID_NUM_NODES];
    int from_node[POWER_GRID_NUM_BRANCHES];
    int to_node[POWER_GRID_NUM_BRANCHES];
    unsigned char pattern[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    PowerGridTopology topology;
    double lu[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
} PowerGridDCFactorCache;

#define POWER_GRID_DC_CACHE_SLOTS 4
static __thread PowerGridDCFactorCache power_grid_dc_cache[POWER_GRID_DC_CACHE_SLOTS];
static __thread unsigned int power_grid_dc_cache_next;

#define POWER_GRID_VALIDATION_CACHE_SLOTS 32
typedef struct {
    int valid;
    PowerGridSolveStatus status;
    int component_count;
    int active_node_count;
    unsigned char active[POWER_GRID_NUM_NODES];
    PowerGridTopology topology;
} PowerGridValidationCache;
static __thread PowerGridValidationCache power_grid_validation_cache[POWER_GRID_VALIDATION_CACHE_SLOTS];
static __thread unsigned int power_grid_validation_cache_next;

static int power_grid_factorize_cached(PowerGridDCFactorCache* cache,
        const PowerGridTopology* topology, const int* row_for_node, int dimensions) {
    if (cache->valid && cache->dimensions == dimensions &&
            memcmp(&cache->topology, topology, sizeof(*topology)) == 0 &&
            memcmp(cache->row_for_node, row_for_node,
                   sizeof(cache->row_for_node)) == 0) {
        return 1;
    }

    memset(cache->lu, 0, sizeof(cache->lu));
    memset(cache->pattern, 0, sizeof(cache->pattern));
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        cache->from_node[branch] = -1;
        cache->to_node[branch] = -1;
    }
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus);
        int to = power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus);
        cache->from_node[branch] = from;
        cache->to_node[branch] = to;
        double susceptance = POWER_GRID_BASE_MVA / (line->reactance * line->tap_ratio);
        int from_row = row_for_node[from];
        int to_row = row_for_node[to];
        if (from_row >= 0) cache->lu[from_row][from_row] += susceptance;
        if (to_row >= 0) cache->lu[to_row][to_row] += susceptance;
        if (from_row >= 0 && to_row >= 0) {
            cache->lu[from_row][to_row] -= susceptance;
            cache->lu[to_row][from_row] -= susceptance;
            int high = from_row > to_row ? from_row : to_row;
            int low = from_row > to_row ? to_row : from_row;
            cache->pattern[high][low] = 1;
        }
    }
    for (int row = 0; row < dimensions; row++) cache->pattern[row][row] = 1;

    /* Symbolic elimination connects the higher-index neighbors of each
     * eliminated node. Numeric Cholesky then visits only this sparse fill
     * pattern, while retaining exact double-precision arithmetic. */
    for (int col = 0; col < dimensions; col++) {
        int neighbors[POWER_GRID_NUM_NODES];
        int count = 0;
        for (int row = col + 1; row < dimensions; row++)
            if (cache->pattern[row][col]) neighbors[count++] = row;
        for (int a = 0; a < count; a++) {
            for (int b = 0; b < a; b++) {
                int high = neighbors[a] > neighbors[b] ? neighbors[a] : neighbors[b];
                int low = neighbors[a] > neighbors[b] ? neighbors[b] : neighbors[a];
                cache->pattern[high][low] = 1;
            }
        }
    }

    for (int row = 0; row < dimensions; row++) {
        for (int col = 0; col <= row; col++) {
            if (!cache->pattern[row][col]) continue;
            double value = cache->lu[row][col];
            for (int k = 0; k < col; k++)
                if (cache->pattern[row][k] && cache->pattern[col][k])
                    value -= cache->lu[row][k] * cache->lu[col][k];
            if (row == col) {
                if (!isfinite(value) || value < 1e-12) {
                    cache->valid = 0;
                    return 0;
                }
                cache->lu[row][col] = sqrt(value);
            } else {
                cache->lu[row][col] = value / cache->lu[col][col];
            }
        }
    }
    cache->dimensions = dimensions;
    cache->topology = *topology;
    memcpy(cache->row_for_node, row_for_node, sizeof(cache->row_for_node));
    cache->valid = 1;
    return 1;
}

static int power_grid_solve_cached_rhs(const PowerGridDCFactorCache* cache,
        const double* rhs, double* solution) {
    int dimensions = cache->dimensions;
    for (int i = 0; i < dimensions; i++) solution[i] = rhs[i];
    for (int row = 0; row < dimensions; row++) {
        for (int col = 0; col < row; col++) solution[row] -= cache->lu[row][col] * solution[col];
        solution[row] /= cache->lu[row][row];
    }
    for (int row = dimensions - 1; row >= 0; row--) {
        for (int col = row + 1; col < dimensions; col++)
            solution[row] -= cache->lu[col][row] * solution[col];
        solution[row] /= cache->lu[row][row];
        if (!isfinite(solution[row])) return 0;
    }
    return 1;
}

PowerGridSolveStatus power_grid_solve_scaled(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridSolveResult* result,
        double branch_rating_scale) {
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    int row_for_node[POWER_GRID_NUM_NODES];
    double injection[POWER_GRID_NUM_NODES] = {0};
    double rhs[POWER_GRID_NUM_NODES] = {0};
    double solution[POWER_GRID_NUM_NODES] = {0};
    memset(result, 0, sizeof(*result));
    int validation_cache_hit = 0;
    for (int slot = 0; slot < POWER_GRID_DC_CACHE_SLOTS; slot++) {
        PowerGridDCFactorCache* candidate = &power_grid_dc_cache[slot];
        if (candidate->valid &&
                memcmp(&candidate->topology, topology, sizeof(*topology)) == 0) {
            memcpy(active, candidate->active, sizeof(active));
            result->component_count = candidate->component_count;
            result->active_node_count = candidate->active_node_count;
            validation_cache_hit = 1;
            break;
        }
    }
    if (!validation_cache_hit) {
        for (int slot = 0; slot < POWER_GRID_VALIDATION_CACHE_SLOTS; slot++) {
            PowerGridValidationCache* candidate = &power_grid_validation_cache[slot];
            if (candidate->valid &&
                    memcmp(&candidate->topology, topology, sizeof(*topology)) == 0) {
                memcpy(active, candidate->active, sizeof(active));
                result->status = candidate->status;
                result->component_count = candidate->component_count;
                result->active_node_count = candidate->active_node_count;
                validation_cache_hit = 1;
                break;
            }
        }
    }
    if (!validation_cache_hit) {
        result->status = power_grid_validate_topology_internal(topology,
            &result->component_count, &result->active_node_count, active);
        if (result->status != POWER_GRID_SOLVE_OK) {
            PowerGridValidationCache* entry = &power_grid_validation_cache[
                power_grid_validation_cache_next++ % POWER_GRID_VALIDATION_CACHE_SLOTS];
            entry->valid = 1;
            entry->status = result->status;
            entry->component_count = result->component_count;
            entry->active_node_count = result->active_node_count;
            memcpy(entry->active, active, sizeof(entry->active));
            entry->topology = *topology;
        }
    }
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
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) row_for_node[node] = -1;
    for (int rank = 0; rank < POWER_GRID_NUM_SUBSTATIONS; rank++) {
        int bus = POWER_GRID_BUS_ORDER[rank];
        for (int bar = 0; bar < POWER_GRID_NUM_BUSBARS; bar++) {
            int node = 2 * bus + bar;
            if (active[node] && node != reference) row_for_node[node] = dimensions++;
        }
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (row_for_node[node] >= 0) rhs[row_for_node[node]] = injection[node];
    }
    PowerGridDCFactorCache* cache = NULL;
    for (int slot = 0; slot < POWER_GRID_DC_CACHE_SLOTS; slot++) {
        PowerGridDCFactorCache* candidate = &power_grid_dc_cache[slot];
        if (candidate->valid && candidate->dimensions == dimensions &&
                memcmp(&candidate->topology, topology, sizeof(*topology)) == 0 &&
                memcmp(candidate->row_for_node, row_for_node,
                       sizeof(candidate->row_for_node)) == 0) {
            cache = candidate;
            break;
        }
    }
    if (!cache) {
        cache = &power_grid_dc_cache[power_grid_dc_cache_next++ % POWER_GRID_DC_CACHE_SLOTS];
    }
    if (!power_grid_factorize_cached(cache, topology,
            row_for_node, dimensions) ||
            !power_grid_solve_cached_rhs(cache, rhs, solution)) {
        result->status = POWER_GRID_SINGULAR;
        return result->status;
    }
    cache->component_count = result->component_count;
    cache->active_node_count = result->active_node_count;
    memcpy(cache->active, active, sizeof(cache->active));
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (row_for_node[node] >= 0) result->node_angle[node] = solution[row_for_node[node]];
    }

    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = cache->from_node[branch];
        int to = cache->to_node[branch];
        double flow = POWER_GRID_BASE_MVA * (result->node_angle[from] - result->node_angle[to]) /
            (line->reactance * line->tap_ratio);
        double rho = fabs(flow) / (line->thermal_limit_mw * branch_rating_scale);
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

PowerGridSolveStatus power_grid_solve(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridSolveResult* result) {
    return power_grid_solve_scaled(topology, point, result, 1.0);
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
