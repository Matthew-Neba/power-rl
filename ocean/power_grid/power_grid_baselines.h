#ifndef POWER_GRID_BASELINES_H
#define POWER_GRID_BASELINES_H

#include "power_grid_solver.h"

int power_grid_random_action(unsigned int* rng);
int power_grid_greedy_action(
    const PowerGridTopology* topology,
    const PowerGridOperatingPoint* point,
    PowerGridSolveResult* selected_result
);
#endif
