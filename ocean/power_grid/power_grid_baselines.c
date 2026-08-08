#include "power_grid_baselines.h"

#include <float.h>
#include <math.h>

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
