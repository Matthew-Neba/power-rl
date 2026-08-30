#define POWER_GRID_NO_RENDER

#include "power_grid_solver.c"
#include "power_grid.h"
#include "power_grid_user.h"

#include <math.h>
#include <stdio.h>

static int failures;

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)

int main(void)
{
    int bridge = -1, first = -1, second = -1, third = -1;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        PowerGridTopology topology;
        power_grid_topology_normal(&topology);
        topology.line_closed[line] = 0;
        if (power_grid_validate_topology(&topology, NULL, NULL) != POWER_GRID_SOLVE_OK)
        {
            bridge = line;
            break;
        }
    }
    for (int a = 0; a < POWER_GRID_NUM_BRANCHES && first < 0; a++)
        for (int b = a + 1; b < POWER_GRID_NUM_BRANCHES; b++)
        {
            PowerGridTopology topology;
            power_grid_topology_normal(&topology);
            topology.line_closed[a] = 0;
            topology.line_closed[b] = 0;
            if (power_grid_validate_topology(&topology, NULL, NULL) == POWER_GRID_SOLVE_OK)
            {
                first = a;
                second = b;
                break;
            }
        }
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        if (line != first && line != second)
        {
            third = line;
            break;
        }
    CHECK(bridge >= 0 && first >= 0 && second >= 0 && third >= 0);

    PowerGrid env = {
        .offline_scenarios = 1,
        .offline_scenario_validation = 1,
    };
    power_grid_allocate(&env);
    env.max_episode_steps = 0;
    c_reset(&env);

    PowerGridUserSession user;
    power_grid_user_init(&user, 2);
    float bridge_available_before =
        env.observations[POWER_GRID_LINE_OBS_OFFSET +
                         bridge * POWER_GRID_LINE_OBS_FEATURES + 3];
    CHECK(power_grid_user_set_line_outage(&user, &env, bridge, 1) !=
          POWER_GRID_SOLVE_OK);
    CHECK(!env.line_available[bridge]);
    CHECK(!env.topology.line_closed[bridge]);
    CHECK(user.line_outage[bridge]);
    CHECK(bridge_available_before == 1.0f);
    CHECK(env.observations[POWER_GRID_LINE_OBS_OFFSET +
                           bridge * POWER_GRID_LINE_OBS_FEATURES + 2] == 0.0f);
    CHECK(env.observations[POWER_GRID_LINE_OBS_OFFSET +
                           bridge * POWER_GRID_LINE_OBS_FEATURES + 3] == 0.0f);

    c_reset(&env);
    power_grid_user_init(&user, 2);

    CHECK(power_grid_user_set_line_outage(&user, &env, first, 1) ==
          POWER_GRID_SOLVE_OK);
    CHECK(power_grid_user_set_line_outage(&user, &env, second, 1) ==
          POWER_GRID_SOLVE_OK);
    power_grid_user_set_line_outage(&user, &env, third, 1);
    CHECK(power_grid_user_outage_count(&user) == 2);
    CHECK(env.line_available[first] && env.topology.line_closed[first]);
    CHECK(!user.line_outage[first]);
    CHECK(!env.line_available[second] && user.line_outage[second]);
    CHECK(!env.line_available[third] && user.line_outage[third]);
    CHECK(user.outage_order[0] == second && user.outage_order[1] == third);

    CHECK(power_grid_user_set_line_outage(&user, &env, third, 0) ==
          POWER_GRID_SOLVE_OK);
    CHECK(power_grid_user_outage_count(&user) == 1);
    CHECK(user.outage_order[0] == second);
    CHECK(env.line_available[third] && !env.topology.line_closed[third]);

    PowerGridTopology persistent = env.topology;
    for (int step = 0; step < POWER_GRID_EPISODE_STEPS + 1; step++)
    {
        env.actions[0] = POWER_GRID_ACTION_NONE;
        c_step(&env);
        CHECK(env.terminals[0] == 0.0f);
    }
    CHECK(env.episode_step == POWER_GRID_EPISODE_STEPS + 1);
    CHECK(env.current_period == 0);
    CHECK(memcmp(&env.topology, &persistent, sizeof(persistent)) == 0);
    CHECK(env.log.n == 0.0f);

    CHECK(power_grid_user_set_line_outage(&user, &env, second, 0) ==
          POWER_GRID_SOLVE_OK);
    CHECK(env.line_available[second]);
    c_close(&env);

    if (failures)
        return 1;
    puts("Power-grid user controller tests passed");
    return 0;
}
