#include "power_grid_solver.c"
#include "power_grid_ac.c"
#include "power_grid.h"

int main(void) {
    PowerGrid env = {0};
    env.ac_power_flow = 1;
    env.offline_scenarios = 1;
    env.offline_scenario_validation = 1;
    env.random_events = 1;
    env.random_event_probability = 0.25;
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
