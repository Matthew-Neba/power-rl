#include "power_grid_solver.c"
#include "power_grid_ac.c"
#include "power_grid.h"

int main(void) {
    PowerGrid env = {0};
    env.ac_power_flow = 1;
    env.evaluation_scenarios = 1;
    power_grid_allocate(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        /* Standalone playback is observation-only; puffer eval supplies the policy action. */
        env.actions[0] = 0.0f;
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
