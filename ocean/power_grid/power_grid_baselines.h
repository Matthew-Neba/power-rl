#ifndef POWER_GRID_BASELINES_H
#define POWER_GRID_BASELINES_H

#include "power_grid_solver.h"

#define POWER_GRID_MAX_SEARCH_DEPTH 4

typedef struct {
    int found;
    int depth;
    int actions[POWER_GRID_MAX_SEARCH_DEPTH];
    int topologies_evaluated;
    PowerGridSolveResult final_result;
} PowerGridSearchResult;

int power_grid_random_action(unsigned int* rng);
int power_grid_greedy_action(
    const PowerGridTopology* topology,
    const PowerGridOperatingPoint* point,
    PowerGridSolveResult* selected_result
);
PowerGridSearchResult power_grid_search_safe_topology(
    const PowerGridTopology* topology,
    const PowerGridOperatingPoint* point,
    int max_depth
);

#endif
