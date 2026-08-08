#include "power_grid_baselines.h"

int power_grid_random_action(unsigned int* rng) {
    *rng = *rng * 1664525u + 1013904223u;
    return (int)(*rng % POWER_GRID_NUM_ACTIONS);
}
