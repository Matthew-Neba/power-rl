#include "power_grid_baselines.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)

static void test_baseline_actions(void) {
    unsigned int rng_a = 1234, rng_b = 1234;
    PowerGridTopology topology;
    power_grid_topology_normal(&topology);
    PowerGridTopology unchanged = topology;
    CHECK(power_grid_apply_action(&topology, 0) == POWER_GRID_ACTION_NONE);
    CHECK(memcmp(&topology, &unchanged, sizeof(topology)) == 0);
    CHECK(power_grid_apply_action(&topology, POWER_GRID_NUM_ACTIONS) ==
        POWER_GRID_ACTION_INVALID);
    for (int i = 0; i < 100; i++) {
        int a = power_grid_random_action(&rng_a);
        int b = power_grid_random_action(&rng_b);
        CHECK(a == b);
        CHECK(a >= 0 && a < POWER_GRID_NUM_ACTIONS);
    }
}

static void test_intended_reconfiguration_sequence(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult result;
    power_grid_topology_normal(&topology);

    power_grid_operating_point_profile(&point, POWER_GRID_PROFILE_P1_HIGH);
    CHECK(power_grid_apply_action(&topology, 27) == POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    CHECK(power_grid_apply_action(&topology, 29) == POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    CHECK(result.max_rho < 1.0);

    power_grid_operating_point_profile(&point, POWER_GRID_PROFILE_P3_RESTORATION_SHIFT);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    CHECK(result.max_rho > 1.0);
    CHECK(power_grid_apply_action(&topology, 27) == POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_apply_action(&topology, 29) == POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    CHECK(result.max_rho < 1.0);

    power_grid_operating_point_profile(&point, POWER_GRID_PROFILE_P2_EAST_SHIFT);
    CHECK(power_grid_apply_action(&topology, 8) == POWER_GRID_ACTION_LINE);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    CHECK(result.max_rho < 1.0);

    power_grid_operating_point_profile(&point, POWER_GRID_PROFILE_P0_NOMINAL);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    CHECK(result.max_rho < 1.0);
}

int main(void) {
    test_baseline_actions();
    test_intended_reconfiguration_sequence();
    if (failures) return 1;
    puts("power-grid baseline tests passed");
    return 0;
}
