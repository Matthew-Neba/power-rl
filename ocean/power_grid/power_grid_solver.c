#include "power_grid_solver.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static const char* const POWER_GRID_PROFILE_NAMES[POWER_GRID_NUM_PROFILES] = {
    "P0 nominal", "P1 high load", "P2 low load", "P3 region A peak",
    "P4 region B peak", "P5 region C peak", "P6 region D peak",
    "P7 A-to-D shift", "P8 D-to-A shift", "P9 high wind", "P10 high solar",
};

/* A fixed minimum-degree ordering for the IEEE-14 bus graph. Applying the
 * same bus ordering to both busbars preserves exact DC solutions while
 * substantially reducing sparse-Cholesky fill for all valid topologies. */
static const int POWER_GRID_BUS_ORDER[POWER_GRID_NUM_SUBSTATIONS] = {
    7, 0, 2, 1, 4, 6, 3, 9, 10, 8, 11, 5, 12, 13,
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
            if ((POWER_GRID_LOAD_BUSES[load] * 4) /
                    POWER_GRID_NUM_SUBSTATIONS == region)
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
            int region = (POWER_GRID_LOAD_BUSES[load] * 4) /
                         POWER_GRID_NUM_SUBSTATIONS;
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
            if ((POWER_GRID_LOAD_BUSES[load] * 4) /
                    POWER_GRID_NUM_SUBSTATIONS == target)
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
     * touches only the closed branches instead of a dense node matrix. */
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
    unsigned short factor_cols[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    unsigned short factor_count[POWER_GRID_NUM_NODES];
    unsigned short factor_rows[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    unsigned short factor_row_count[POWER_GRID_NUM_NODES];
    PowerGridTopology topology;
#ifdef POWER_GRID_DC_FLOAT
    float lu[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
#else
    double lu[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
#endif
} PowerGridDCFactorCache;

#define POWER_GRID_DC_CACHE_SLOTS 1
#ifdef POWER_GRID_DC_FLOAT
typedef float PowerGridDCScalar;
#define POWER_GRID_DC_SQRT sqrtf
#else
typedef double PowerGridDCScalar;
#define POWER_GRID_DC_SQRT sqrt
#endif

static __thread PowerGridDCFactorCache power_grid_dc_cache[POWER_GRID_DC_CACHE_SLOTS];
static __thread unsigned int power_grid_dc_cache_next;
static __thread int power_grid_branch_constants_initialized;
static __thread double power_grid_branch_susceptance[POWER_GRID_NUM_BRANCHES];
static __thread double power_grid_branch_angle_factor[POWER_GRID_NUM_BRANCHES];

static inline void power_grid_init_branch_constants(void)
{
    if (power_grid_branch_constants_initialized)
        return;
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        double denominator = line->reactance * line->tap_ratio;
        power_grid_branch_susceptance[branch] = POWER_GRID_BASE_MVA / denominator;
        power_grid_branch_angle_factor[branch] = POWER_GRID_BASE_MVA / denominator;
    }
    power_grid_branch_constants_initialized = 1;
}

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
        double susceptance = power_grid_branch_susceptance[branch];
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
        cache->factor_count[row] = 0;
        for (int col = 0; col < row; col++) {
            if (cache->pattern[row][col])
                cache->factor_cols[row][cache->factor_count[row]++] = (unsigned short)col;
        }
    }
    for (int col = 0; col < dimensions; col++) cache->factor_row_count[col] = 0;
    for (int row = 0; row < dimensions; row++) {
        for (int index = 0; index < cache->factor_count[row]; index++) {
            int col = cache->factor_cols[row][index];
            cache->factor_rows[col][cache->factor_row_count[col]++] = (unsigned short)row;
        }
    }

    for (int row = 0; row < dimensions; row++) {
        for (int col = 0; col <= row; col++) {
            if (!cache->pattern[row][col]) continue;
            PowerGridDCScalar value = cache->lu[row][col];
            for (int index = 0; index < cache->factor_count[row]; index++) {
                int k = cache->factor_cols[row][index];
                if (k >= col) break;
                if (cache->pattern[col][k]) value -= cache->lu[row][k] * cache->lu[col][k];
            }
            if (row == col) {
                if (!isfinite(value) || value < 1e-12) {
                    cache->valid = 0;
                    return 0;
                }
                cache->lu[row][col] = POWER_GRID_DC_SQRT(value);
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
        const PowerGridDCScalar* rhs, PowerGridDCScalar* solution) {
    int dimensions = cache->dimensions;
    for (int i = 0; i < dimensions; i++) solution[i] = rhs[i];
    for (int row = 0; row < dimensions; row++) {
        for (int index = 0; index < cache->factor_count[row]; index++) {
            int col = cache->factor_cols[row][index];
            solution[row] -= cache->lu[row][col] * solution[col];
        }
        solution[row] /= cache->lu[row][row];
    }
    for (int row = dimensions - 1; row >= 0; row--) {
        for (int index = 0; index < cache->factor_row_count[row]; index++) {
            int col = cache->factor_rows[row][index];
            solution[row] -= cache->lu[col][row] * solution[col];
        }
        solution[row] /= cache->lu[row][row];
        if (!isfinite(solution[row])) return 0;
    }
    return 1;
}

PowerGridSolveStatus power_grid_solve_scaled(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridSolveResult* result,
        double branch_rating_scale) {
    power_grid_init_branch_constants();
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    int row_for_node[POWER_GRID_NUM_NODES];
    double injection[POWER_GRID_NUM_NODES] = {0};
    PowerGridDCScalar rhs[POWER_GRID_NUM_NODES] = {0};
    PowerGridDCScalar solution[POWER_GRID_NUM_NODES] = {0};
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
        double output = gen == 0 ? result->slack_generation_mw : point->generator_mw[gen];
        injection[node] += output;
        result->substation_injection_mw[POWER_GRID_GENERATOR_BUSES[gen]] += output;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
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
        if (row_for_node[node] >= 0) rhs[row_for_node[node]] = (PowerGridDCScalar)injection[node];
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
#ifdef POWER_GRID_DC_FLOAT
    /* Recover most of the double-precision solution accuracy with one cheap
     * residual corrections while retaining the faster float factorization. */
    double refined[POWER_GRID_NUM_NODES] = {0};
    double residual[POWER_GRID_NUM_NODES];
    PowerGridDCScalar correction_rhs[POWER_GRID_NUM_NODES];
    PowerGridDCScalar correction[POWER_GRID_NUM_NODES];
    int node_for_row[POWER_GRID_NUM_NODES];
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
        if (row_for_node[node] >= 0) node_for_row[row_for_node[node]] = node;
    for (int row = 0; row < dimensions; row++) refined[row] = solution[row];
    for (int iteration = 0; iteration < 1; iteration++) {
        for (int row = 0; row < dimensions; row++) {
            residual[row] = injection[node_for_row[row]];
        }
        for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
            if (!topology->line_closed[branch]) continue;
            int from = cache->from_node[branch];
            int to = cache->to_node[branch];
            int from_row = row_for_node[from];
            int to_row = row_for_node[to];
            double from_angle = from_row >= 0 ? refined[from_row] : 0.0;
            double to_angle = to_row >= 0 ? refined[to_row] : 0.0;
            double flow = power_grid_branch_angle_factor[branch] *
                (from_angle - to_angle);
            if (from_row >= 0) residual[from_row] -= flow;
            if (to_row >= 0) residual[to_row] += flow;
        }
        for (int row = 0; row < dimensions; row++) correction_rhs[row] =
            (PowerGridDCScalar)residual[row];
        if (!power_grid_solve_cached_rhs(cache, correction_rhs, correction)) {
            result->status = POWER_GRID_SINGULAR;
            return result->status;
        }
        for (int row = 0; row < dimensions; row++) refined[row] += correction[row];
    }
#endif
    cache->component_count = result->component_count;
    cache->active_node_count = result->active_node_count;
    memcpy(cache->active, active, sizeof(cache->active));
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (row_for_node[node] >= 0)
#ifdef POWER_GRID_DC_FLOAT
            result->node_angle[node] = refined[row_for_node[node]];
#else
            result->node_angle[node] = solution[row_for_node[node]];
#endif
    }

    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch* line = &POWER_GRID_BRANCHES[branch];
        int from = cache->from_node[branch];
        int to = cache->to_node[branch];
        double flow = power_grid_branch_angle_factor[branch] *
            (result->node_angle[from] - result->node_angle[to]);
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
            result->overloaded_branch_count++;
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

/* AC Newton-Raphson evaluation. It shares the topology, operating point, and
 * dense pivoted solve above with the DC training model. */
#define AC_STATE_SIZE (2 * POWER_GRID_NUM_NODES)
#define AC_TOLERANCE 1e-8

/* Electrical data comes from canonical MATPOWER case14. Its source ratings are
 * unusable, so the shared data supplies documented synthetic ratings for both
 * the DC training proxy and AC validation. */
#define AC_BRANCH_R POWER_GRID_BRANCH_R
#define AC_BRANCH_B POWER_GRID_BRANCH_B
#define AC_LOAD_Q_NOMINAL POWER_GRID_LOAD_Q_NOMINAL
#define AC_LOAD_P_NOMINAL POWER_GRID_LOAD_P_NOMINAL
#define AC_GEN_V_SETPOINT POWER_GRID_GENERATOR_V_SETPOINT
#define AC_GEN_Q_MIN POWER_GRID_GENERATOR_Q_MIN
#define AC_GEN_Q_MAX POWER_GRID_GENERATOR_Q_MAX
#define AC_GEN_P_MAX POWER_GRID_GENERATOR_P_MAX

typedef struct {
    unsigned char active[POWER_GRID_NUM_NODES];
    unsigned char pv[POWER_GRID_NUM_NODES];
    double g[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    double b[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    double p_spec[POWER_GRID_NUM_NODES];
    double q_spec[POWER_GRID_NUM_NODES];
    double load_q[POWER_GRID_NUM_NODES];
    double voltage[POWER_GRID_NUM_NODES];
    double angle[POWER_GRID_NUM_NODES];
    int generator_node[POWER_GRID_NUM_GENERATORS];
    int slack;
} ACNetwork;

double power_grid_ac_load_q_mvar(const PowerGridOperatingPoint* point, int load) {
    return AC_LOAD_Q_NOMINAL[load] * point->load_mw[load] / AC_LOAD_P_NOMINAL[load];
}

double power_grid_ac_branch_rating_mva(int branch) {
    return POWER_GRID_BRANCHES[branch].thermal_limit_mw;
}

double power_grid_ac_thermal_step(double previous_stress, double rho) {
    if (rho <= 1.0) return fmax(0.0, previous_stress - POWER_GRID_THERMAL_COOLING_PER_STEP);
    double overload = rho - 1.0;
    return previous_stress + overload * overload;
}

static void ac_build_network(const PowerGridTopology* topology, ACNetwork* ac) {
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!topology->line_closed[line]) continue;
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
            branch->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
            branch->to_bus);
        ac->active[from] = ac->active[to] = 1;
        double r = AC_BRANCH_R[line], x = branch->reactance;
        double denominator = r * r + x * x;
        double series_g = r / denominator;
        double series_b = -x / denominator;
        double tap = branch->tap_ratio;
        ac->g[from][from] += series_g / (tap * tap);
        ac->b[from][from] += (series_b + AC_BRANCH_B[line] / 2.0) / (tap * tap);
        ac->g[to][to] += series_g;
        ac->b[to][to] += series_b + AC_BRANCH_B[line] / 2.0;
        ac->g[from][to] -= series_g / tap;
        ac->b[from][to] -= series_b / tap;
        ac->g[to][from] -= series_g / tap;
        ac->b[to][from] -= series_b / tap;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        ac->active[node] = 1;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        ac->active[node] = 1;
    }
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        int node = 2 * bus;
        ac->b[node][node] += POWER_GRID_BUS_SHUNT_B_MVAR[bus] / POWER_GRID_BASE_MVA;
    }
}

static void ac_calculate_power(const ACNetwork* ac, double* p, double* q) {
    memset(p, 0, POWER_GRID_NUM_NODES * sizeof(*p));
    memset(q, 0, POWER_GRID_NUM_NODES * sizeof(*q));
    for (int i = 0; i < POWER_GRID_NUM_NODES; i++) {
        if (!ac->active[i]) continue;
        for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
            if (!ac->active[j]) continue;
            double difference = ac->angle[i] - ac->angle[j];
            double cosine = cos(difference), sine = sin(difference);
            p[i] += ac->voltage[i] * ac->voltage[j] *
                (ac->g[i][j] * cosine + ac->b[i][j] * sine);
            q[i] += ac->voltage[i] * ac->voltage[j] *
                (ac->g[i][j] * sine - ac->b[i][j] * cosine);
        }
    }
}

static PowerGridACStatus ac_newton_raphson(ACNetwork* ac, int* total_iterations) {
    int angle_index[POWER_GRID_NUM_NODES], voltage_index[POWER_GRID_NUM_NODES];
    int dimensions = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        angle_index[node] = voltage_index[node] = -1;
        if (ac->active[node] && node != ac->slack) angle_index[node] = dimensions++;
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (ac->active[node] && node != ac->slack && !ac->pv[node]) {
            voltage_index[node] = dimensions++;
        }
    }

    for (int iteration = 0; iteration < POWER_GRID_AC_MAX_ITERATIONS; iteration++) {
        double p[POWER_GRID_NUM_NODES], q[POWER_GRID_NUM_NODES];
        double jacobian[AC_STATE_SIZE][AC_STATE_SIZE] = {{0}};
        double mismatch[AC_STATE_SIZE] = {0}, update[AC_STATE_SIZE] = {0};
        ac_calculate_power(ac, p, q);
        double largest = 0.0;
        for (int i = 0; i < POWER_GRID_NUM_NODES; i++) {
            if (angle_index[i] >= 0) {
                mismatch[angle_index[i]] = ac->p_spec[i] - p[i];
                if (fabs(mismatch[angle_index[i]]) > largest) largest = fabs(mismatch[angle_index[i]]);
            }
            if (voltage_index[i] >= 0) {
                mismatch[voltage_index[i]] = ac->q_spec[i] - q[i];
                if (fabs(mismatch[voltage_index[i]]) > largest) largest = fabs(mismatch[voltage_index[i]]);
            }
        }
        (*total_iterations)++;
        if (largest < AC_TOLERANCE) return POWER_GRID_AC_OK;

        for (int i = 0; i < POWER_GRID_NUM_NODES; i++) {
            if (!ac->active[i] || i == ac->slack) continue;
            int pi = angle_index[i];
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (!ac->active[j] || j == ac->slack) continue;
                int tj = angle_index[j];
                if (i == j) {
                    jacobian[pi][tj] = -q[i] - ac->b[i][i] *
                        ac->voltage[i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[pi][tj] = ac->voltage[i] * ac->voltage[j] *
                        (ac->g[i][j] * sin(d) - ac->b[i][j] * cos(d));
                }
            }
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (voltage_index[j] < 0) continue;
                int vj = voltage_index[j];
                if (i == j) {
                    jacobian[pi][vj] = p[i] / ac->voltage[i] +
                        ac->g[i][i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[pi][vj] = ac->voltage[i] *
                        (ac->g[i][j] * cos(d) + ac->b[i][j] * sin(d));
                }
            }
            if (voltage_index[i] < 0) continue;
            int qi = voltage_index[i];
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (!ac->active[j] || j == ac->slack) continue;
                int tj = angle_index[j];
                if (i == j) {
                    jacobian[qi][tj] = p[i] - ac->g[i][i] *
                        ac->voltage[i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[qi][tj] = -ac->voltage[i] * ac->voltage[j] *
                        (ac->g[i][j] * cos(d) + ac->b[i][j] * sin(d));
                }
            }
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (voltage_index[j] < 0) continue;
                int vj = voltage_index[j];
                if (i == j) {
                    jacobian[qi][vj] = q[i] / ac->voltage[i] -
                        ac->b[i][i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[qi][vj] = ac->voltage[i] *
                        (ac->g[i][j] * sin(d) - ac->b[i][j] * cos(d));
                }
            }
        }
        if (!power_grid_solve_dense(&jacobian[0][0], mismatch, update, dimensions,
                AC_STATE_SIZE)) {
            return POWER_GRID_AC_SINGULAR;
        }
        double scale = 1.0;
        for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
            if (angle_index[node] >= 0 && fabs(update[angle_index[node]]) > 0.5) scale = 0.5;
            if (voltage_index[node] >= 0 && fabs(update[voltage_index[node]]) > 0.2) scale = 0.5;
        }
        for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
            if (angle_index[node] >= 0) ac->angle[node] += scale * update[angle_index[node]];
            if (voltage_index[node] >= 0) {
                ac->voltage[node] += scale * update[voltage_index[node]];
                if (!isfinite(ac->voltage[node]) || ac->voltage[node] < 0.2 ||
                        ac->voltage[node] > 2.0) {
                    return POWER_GRID_AC_DIVERGED;
                }
            }
        }
    }
    return POWER_GRID_AC_DIVERGED;
}

static void ac_branch_flow(int line, int from, int to, const double* voltage,
        const double* angle, double* from_p, double* from_q, double* to_p, double* to_q) {
    const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
    double tap = branch->tap_ratio;
    double complex series = 1.0 / (AC_BRANCH_R[line] + I * branch->reactance);
    double complex charging = I * AC_BRANCH_B[line] / 2.0;
    double complex from_voltage = voltage[from] * cexp(I * angle[from]);
    double complex to_voltage = voltage[to] * cexp(I * angle[to]);
    double complex from_current = (series + charging) * from_voltage / (tap * tap) -
        series * to_voltage / tap;
    double complex to_current = (series + charging) * to_voltage -
        series * from_voltage / tap;
    double complex from_power = POWER_GRID_BASE_MVA * from_voltage * conj(from_current);
    double complex to_power = POWER_GRID_BASE_MVA * to_voltage * conj(to_current);
    *from_p = creal(from_power);
    *from_q = cimag(from_power);
    *to_p = creal(to_power);
    *to_q = cimag(to_power);
}

PowerGridACStatus power_grid_ac_solve(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridACSolveResult* result) {
    memset(result, 0, sizeof(*result));
    result->min_voltage_pu = INFINITY;
    result->topology_status = power_grid_validate_topology(topology, &result->component_count,
        &result->active_node_count);
    if (result->topology_status != POWER_GRID_SOLVE_OK) {
        return result->status = POWER_GRID_AC_TOPOLOGY_FAILURE;
    }

    ACNetwork ac = {0};
    ac_build_network(topology, &ac);
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        ac.voltage[node] = ac.active[node] ? 1.0 : 0.0;
    }

    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        if (!isfinite(point->load_mw[load]) || point->load_mw[load] < 0.0) {
            return result->status = POWER_GRID_AC_INVALID_INPUT;
        }
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        double reactive = power_grid_ac_load_q_mvar(point, load);
        ac.p_spec[node] -= point->load_mw[load] / POWER_GRID_BASE_MVA;
        ac.q_spec[node] -= reactive / POWER_GRID_BASE_MVA;
        ac.load_q[node] += reactive;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        if (!isfinite(point->generator_mw[gen]) || point->generator_mw[gen] < 0.0) {
            return result->status = POWER_GRID_AC_INVALID_INPUT;
        }
        ac.generator_node[gen] = power_grid_terminal_node(topology,
            POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        ac.voltage[ac.generator_node[gen]] = AC_GEN_V_SETPOINT[gen];
        if (gen > 0) {
            ac.p_spec[ac.generator_node[gen]] +=
                point->generator_mw[gen] / POWER_GRID_BASE_MVA;
            ac.pv[ac.generator_node[gen]] = 1;
        }
    }
    ac.slack = ac.generator_node[0];

    /* DC angles provide a much better initial guess for stressed/reconfigured cases. */
    PowerGridSolveResult dc = {0};
    if (power_grid_solve(topology, point, &dc) == POWER_GRID_SOLVE_OK) {
        memcpy(ac.angle, dc.node_angle, sizeof(ac.angle));
    }

    for (;;) {
        PowerGridACStatus status = ac_newton_raphson(&ac, &result->iterations);
        if (status != POWER_GRID_AC_OK) {
            /* A second, independent flat start distinguishes a poor DC-derived
             * initial guess from a topology that genuinely will not converge. */
            for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
                ac.voltage[node] = ac.active[node] ? 1.0 : 0.0;
                ac.angle[node] = 0.0;
            }
            ac.voltage[ac.slack] = AC_GEN_V_SETPOINT[0];
            for (int gen = 1; gen < POWER_GRID_NUM_GENERATORS; gen++) {
                if (ac.pv[ac.generator_node[gen]]) {
                    ac.voltage[ac.generator_node[gen]] = AC_GEN_V_SETPOINT[gen];
                }
            }
            status = ac_newton_raphson(&ac, &result->iterations);
        }
        if (status != POWER_GRID_AC_OK) return result->status = status;
        double p[POWER_GRID_NUM_NODES], q[POWER_GRID_NUM_NODES];
        ac_calculate_power(&ac, p, q);
        int limited = -1;
        double limited_q = 0.0;
        for (int gen = 1; gen < POWER_GRID_NUM_GENERATORS; gen++) {
            int node = ac.generator_node[gen];
            if (!ac.pv[node]) continue;
            double generator_q = q[node] * POWER_GRID_BASE_MVA + ac.load_q[node];
            if (generator_q < AC_GEN_Q_MIN[gen] - 1e-7) {
                limited = gen; limited_q = AC_GEN_Q_MIN[gen]; break;
            }
            if (generator_q > AC_GEN_Q_MAX[gen] + 1e-7) {
                limited = gen; limited_q = AC_GEN_Q_MAX[gen]; break;
            }
        }
        if (limited < 0) break;
        int node = ac.generator_node[limited];
        ac.pv[node] = 0;
        ac.q_spec[node] += limited_q / POWER_GRID_BASE_MVA;
        result->q_limit_count++;
    }

    double p[POWER_GRID_NUM_NODES], q[POWER_GRID_NUM_NODES];
    ac_calculate_power(&ac, p, q);
    result->converged = 1;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        result->node_voltage_pu[node] = ac.voltage[node];
        result->node_angle_rad[node] = ac.angle[node];
        if (!ac.active[node]) continue;
        if (!isfinite(ac.voltage[node]) || !isfinite(ac.angle[node])) {
            return result->status = POWER_GRID_AC_NONFINITE;
        }
        if (ac.voltage[node] < result->min_voltage_pu) {
            result->min_voltage_pu = ac.voltage[node];
        }
        if (ac.voltage[node] > result->max_voltage_pu) {
            result->max_voltage_pu = ac.voltage[node];
        }
        double violation = fmax(POWER_GRID_AC_VOLTAGE_MIN - ac.voltage[node],
            ac.voltage[node] - POWER_GRID_AC_VOLTAGE_MAX);
        if (violation > 0.0) {
            result->voltage_violation_count++;
            result->voltage_violation_cost += 100.0 * violation * violation;
        }
    }

    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = ac.generator_node[gen];
        result->generator_p_mw[gen] = gen == 0 ?
            p[node] * POWER_GRID_BASE_MVA : point->generator_mw[gen];
        /* IEEE-14 has at most one aggregated load and generator per bus. */
        double local_load_p = 0.0;
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
            int load_node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
                POWER_GRID_LOAD_BUSES[load]);
            if (load_node == node) local_load_p += point->load_mw[load];
        }
        if (gen == 0) result->generator_p_mw[gen] += local_load_p;
        result->generator_q_mvar[gen] = q[node] * POWER_GRID_BASE_MVA + ac.load_q[node];
        double p_violation = result->generator_p_mw[gen] - AC_GEN_P_MAX[gen];
        if (p_violation > 1e-7) {
            result->generator_p_violation_count++;
            result->generator_p_violation_mw += p_violation;
        }
        result->substation_injection_mw[POWER_GRID_GENERATOR_BUSES[gen]] +=
            result->generator_p_mw[gen];
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        result->substation_injection_mw[POWER_GRID_LOAD_BUSES[load]] -= point->load_mw[load];
    }

    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!topology->line_closed[line]) continue;
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
            branch->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
            branch->to_bus);
        ac_branch_flow(line, from, to, ac.voltage, ac.angle,
            &result->branch_from_p_mw[line],
            &result->branch_from_q_mvar[line], &result->branch_to_p_mw[line],
            &result->branch_to_q_mvar[line]);
        result->branch_from_mva[line] = hypot(result->branch_from_p_mw[line],
            result->branch_from_q_mvar[line]);
        result->branch_to_mva[line] = hypot(result->branch_to_p_mw[line],
            result->branch_to_q_mvar[line]);
        result->branch_rho[line] = fmax(result->branch_from_mva[line],
            result->branch_to_mva[line]) / power_grid_ac_branch_rating_mva(line);
        result->total_p_loss_mw += result->branch_from_p_mw[line] + result->branch_to_p_mw[line];
        result->total_q_loss_mvar += result->branch_from_q_mvar[line] + result->branch_to_q_mvar[line];
        if (!isfinite(result->branch_rho[line])) return result->status = POWER_GRID_AC_NONFINITE;
        if (result->branch_rho[line] > result->max_rho) result->max_rho = result->branch_rho[line];
        if (result->branch_rho[line] > 1.0) {
            double overload = result->branch_rho[line] - 1.0;
            result->congestion_cost += overload * overload;
        }
    }
    return result->status = POWER_GRID_AC_OK;
}

void power_grid_ac_to_compatible(const PowerGridACSolveResult* ac,
        PowerGridSolveResult* compatible) {
    memset(compatible, 0, sizeof(*compatible));
    compatible->component_count = ac->component_count;
    compatible->active_node_count = ac->active_node_count;
    if (ac->status == POWER_GRID_AC_TOPOLOGY_FAILURE) compatible->status = ac->topology_status;
    else if (ac->status == POWER_GRID_AC_OK) compatible->status = POWER_GRID_SOLVE_OK;
    else if (ac->status == POWER_GRID_AC_INVALID_INPUT) compatible->status = POWER_GRID_INVALID_INPUT;
    else if (ac->status == POWER_GRID_AC_NONFINITE) compatible->status = POWER_GRID_NONFINITE;
    else compatible->status = POWER_GRID_SINGULAR;
    compatible->slack_generation_mw = ac->generator_p_mw[0];
    compatible->congestion_cost = ac->congestion_cost;
    compatible->max_rho = ac->max_rho;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        compatible->overloaded_branch_count += ac->branch_rho[line] > 1.0;
    memcpy(compatible->node_angle, ac->node_angle_rad, sizeof(compatible->node_angle));
    memcpy(compatible->branch_flow_mw, ac->branch_from_p_mw, sizeof(compatible->branch_flow_mw));
    memcpy(compatible->branch_rho, ac->branch_rho, sizeof(compatible->branch_rho));
    memcpy(compatible->substation_injection_mw, ac->substation_injection_mw,
        sizeof(compatible->substation_injection_mw));
}

const char* power_grid_ac_status_name(PowerGridACStatus status) {
    static const char* names[] = {
        "ok", "topology_failure", "invalid_input", "singular", "diverged", "nonfinite",
    };
    return names[status];
}
