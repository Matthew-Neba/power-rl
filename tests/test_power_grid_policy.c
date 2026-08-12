#define POWER_GRID_NO_RENDER

#include "power_grid_solver.c"
#include "power_grid.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "power_grid_policy.h"
#pragma clang diagnostic pop

#include <stdio.h>

int main(void)
{
    int action_sizes[1] = {POWER_GRID_NUM_ACTIONS};
    Weights *weights = load_weights("resources/power_grid/policy.bin");
    if (weights == NULL)
        return 1;
    PufferNet *policy = make_puffernet(
        weights, 1, POWER_GRID_OBS_SIZE, POWER_GRID_POLICY_HIDDEN_SIZE,
        POWER_GRID_POLICY_NUM_LAYERS, action_sizes, 1);
    if (policy == NULL)
        return 1;
    free_puffernet(policy);
    free(weights);

    weights = load_weights("resources/power_grid/policy.bin");
    if (weights == NULL)
        return 1;
    policy = make_puffernet(
        weights, 1, POWER_GRID_OBS_SIZE, 2 * POWER_GRID_POLICY_HIDDEN_SIZE,
        POWER_GRID_POLICY_NUM_LAYERS, action_sizes, 1);
    free(weights);
    if (policy != NULL)
    {
        free_puffernet(policy);
        return 1;
    }
    puts("Power-grid deployed policy architecture test passed");
    return 0;
}
