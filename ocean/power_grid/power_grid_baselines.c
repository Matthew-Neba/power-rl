#include "power_grid_baselines.h"

#include <float.h>
#include <math.h>
#include <string.h>

int power_grid_random_action(unsigned int* rng) {
    if (rng == NULL) return 0;
    *rng = *rng * 1664525u + 1013904223u;
    return (int)(*rng % POWER_GRID_NUM_ACTIONS);
}

int power_grid_greedy_action(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridSolveResult* selected_result) {
    int best_action = 0;
    double best_score = DBL_MAX;
    PowerGridSolveResult best_result = {0};
    if (topology == NULL || point == NULL) return 0;
    for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++) {
        PowerGridTopology candidate = *topology;
        PowerGridSolveResult result;
        power_grid_apply_action(&candidate, action);
        if (power_grid_solve(&candidate, point, &result) != POWER_GRID_SOLVE_OK) continue;
        double score = result.congestion_cost + (action != 0 ? 0.01 : 0.0) -
            (result.max_rho <= 1.0 ? 0.2 : 0.0);
        if (score < best_score) {
            best_score = score;
            best_action = action;
            best_result = result;
        }
    }
    if (selected_result != NULL) *selected_result = best_result;
    return best_action;
}

static int search_at_depth(const PowerGridTopology* topology, const PowerGridOperatingPoint* point,
        double rating_scale, int remaining, int previous_action, int* path, int path_length,
        PowerGridSearchResult* result) {
    PowerGridSolveResult current;
    result->topologies_evaluated++;
    if (power_grid_solve(topology, point, &current) != POWER_GRID_SOLVE_OK) return 0;
    if (current.max_rho <= rating_scale) {
        result->found = 1;
        result->depth = path_length;
        memcpy(result->actions, path, (size_t)path_length * sizeof(int));
        result->final_result = current;
        return 1;
    }
    if (remaining == 0) return 0;

    for (int action = 1; action < POWER_GRID_NUM_ACTIONS; action++) {
        if (action == previous_action) continue; /* Immediate reversal cannot improve a shortest path. */
        PowerGridTopology candidate = *topology;
        power_grid_apply_action(&candidate, action);
        path[path_length] = action;
        if (search_at_depth(&candidate, point, rating_scale, remaining - 1, action,
                path, path_length + 1, result)) {
            return 1;
        }
    }
    return 0;
}

PowerGridSearchResult power_grid_search_safe_topology(
        const PowerGridTopology* topology, const PowerGridOperatingPoint* point,
        double rating_scale, int max_depth) {
    PowerGridSearchResult result = {0};
    int path[POWER_GRID_MAX_SEARCH_DEPTH] = {0};
    if (topology == NULL || point == NULL || !isfinite(rating_scale) || rating_scale <= 0.0)
        return result;
    if (max_depth < 0) max_depth = 0;
    if (max_depth > POWER_GRID_MAX_SEARCH_DEPTH) max_depth = POWER_GRID_MAX_SEARCH_DEPTH;
    for (int depth = 0; depth <= max_depth; depth++) {
        if (search_at_depth(topology, point, rating_scale, depth, -1, path, 0, &result))
            return result;
    }
    return result;
}
