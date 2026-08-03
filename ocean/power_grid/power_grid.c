#include "power_grid_solver.c"
#include "power_grid.h"

int main(void) {
    PowerGrid env = {0};
    power_grid_allocate(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_RIGHT)) {
            env.client->selected_action = (env.client->selected_action + 1) % POWER_GRID_NUM_ACTIONS;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            env.client->selected_action = (env.client->selected_action + POWER_GRID_NUM_ACTIONS - 1) %
                POWER_GRID_NUM_ACTIONS;
        }
        if (IsKeyPressed(KEY_DOWN)) {
            env.client->selected_action = (env.client->selected_action + 10) % POWER_GRID_NUM_ACTIONS;
        }
        if (IsKeyPressed(KEY_UP)) {
            env.client->selected_action = (env.client->selected_action + POWER_GRID_NUM_ACTIONS - 10) %
                POWER_GRID_NUM_ACTIONS;
        }
        if (IsKeyPressed(KEY_R)) c_reset(&env);
        if (IsKeyPressed(KEY_SPACE)) {
            env.actions[0] = (float)env.client->selected_action;
            c_step(&env);
        }
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
