#ifndef POWER_GRID_USER_H
#define POWER_GRID_USER_H

#include "power_grid.h"

/* Interactive controls are deliberately outside PowerGrid. Training and
 * evaluation do not own a user session or inherit its two-outage policy. */
typedef struct
{
    PowerGridTopology topology_before_outage;
    int topology_before_outage_valid;
    int max_outages;
} PowerGridUserSession;

static inline void power_grid_user_init(PowerGridUserSession *user, int max_outages)
{
    memset(user, 0, sizeof(*user));
    user->max_outages = max_outages;
}

static inline int power_grid_user_outage_count(const PowerGrid *env)
{
    int count = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        count += !env->line_available[line];
    return count;
}

/* Apply a user request transactionally. Invalid N-1/N-2 combinations leave
 * both the solved environment and the user session exactly as they were. */
static inline PowerGridSolveStatus power_grid_user_set_line_outage(
    PowerGridUserSession *user, PowerGrid *env, int line, int unavailable)
{
    if (line < 0 || line >= POWER_GRID_NUM_BRANCHES)
        return POWER_GRID_INVALID_INPUT;

    unavailable = unavailable != 0;
    if (unavailable == !env->line_available[line])
        return env->solution.status;

    int outage_count = power_grid_user_outage_count(env);
    if (unavailable && outage_count >= user->max_outages)
        return POWER_GRID_INVALID_INPUT;

    PowerGridTopology previous_topology = env->topology;
    PowerGridSolveResult previous_solution = env->solution;
    PowerGridACSolveResult previous_ac_solution = env->ac_solution;
    unsigned char previous_available = env->line_available[line];
    PowerGridTopology previous_base = user->topology_before_outage;
    int previous_base_valid = user->topology_before_outage_valid;

    if (unavailable && outage_count == 0)
    {
        user->topology_before_outage = env->topology;
        user->topology_before_outage_valid = 1;
    }

    env->line_available[line] = !unavailable;
    if (!unavailable && outage_count == 1 && user->topology_before_outage_valid)
        env->topology = user->topology_before_outage;
    else
        env->topology.line_closed[line] = !unavailable;

    PowerGridSolveStatus status = power_grid_solve_environment(env);
    if (status != POWER_GRID_SOLVE_OK)
    {
        power_grid_topology_normal(&env->topology);
        for (int index = 0; index < POWER_GRID_NUM_BRANCHES; index++)
            if (!env->line_available[index])
                env->topology.line_closed[index] = 0;
        status = power_grid_solve_environment(env);
    }
    if (status != POWER_GRID_SOLVE_OK)
    {
        env->line_available[line] = previous_available;
        env->topology = previous_topology;
        env->solution = previous_solution;
        env->ac_solution = previous_ac_solution;
        user->topology_before_outage = previous_base;
        user->topology_before_outage_valid = previous_base_valid;
        return status;
    }

    if (!unavailable && outage_count == 1)
        user->topology_before_outage_valid = 0;
    power_grid_compute_observations(env);
    return POWER_GRID_SOLVE_OK;
}

#ifndef POWER_GRID_NO_RENDER
static inline void power_grid_user_handle_input(PowerGridUserSession *user, PowerGrid *env)
{
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        return;

    Vector2 mouse = GetMousePosition();
    int selected = -1;
    float best_distance = 14.0f;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        const PowerGridBranch *branch = &POWER_GRID_BRANCHES[line];
        Vector2 from_station = POWER_GRID_STATION_POSITIONS[branch->from_bus];
        Vector2 to_station = POWER_GRID_STATION_POSITIONS[branch->to_bus];
        Vector2 from = power_grid_branch_endpoint(from_station, to_station,
            env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 0)]);
        Vector2 to = power_grid_branch_endpoint(to_station, from_station,
            env->topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(line, 1)]);
        float dx = to.x - from.x;
        float dy = to.y - from.y;
        float length_squared = dx * dx + dy * dy;
        float position = length_squared > 0.0f ?
            ((mouse.x - from.x) * dx + (mouse.y - from.y) * dy) /
                length_squared : 0.0f;
        position = fminf(1.0f, fmaxf(0.0f, position));
        float nearest_x = from.x + position * dx;
        float nearest_y = from.y + position * dy;
        float distance = hypotf(mouse.x - nearest_x, mouse.y - nearest_y);
        if (distance < best_distance)
        {
            best_distance = distance;
            selected = line;
        }
    }
    if (selected >= 0)
        power_grid_user_set_line_outage(
            user, env, selected, env->line_available[selected]);
}
#endif

#endif
