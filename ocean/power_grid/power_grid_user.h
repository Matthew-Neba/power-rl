#ifndef POWER_GRID_USER_H
#define POWER_GRID_USER_H

#include "power_grid.h"

/* Interactive controls are deliberately outside PowerGrid. Training and
 * evaluation do not own a user session or inherit its two-outage policy. */
typedef struct
{
    int max_outages;
    unsigned char line_outage[POWER_GRID_NUM_BRANCHES];
    int last_line;
    PowerGridSolveStatus last_status;
} PowerGridUserSession;

static inline void power_grid_user_init(PowerGridUserSession *user, int max_outages)
{
    memset(user, 0, sizeof(*user));
    user->max_outages = max_outages;
    user->last_line = -1;
    user->last_status = POWER_GRID_SOLVE_OK;
}

static inline int power_grid_user_outage_count(const PowerGridUserSession *user)
{
    int count = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        count += user->line_outage[line];
    return count;
}

/* User outages are exogenous events, not agent actions. Apply them exactly as
 * requested so an infeasible contingency is visible and counted as a failure. */
static inline PowerGridSolveStatus power_grid_user_set_line_outage(
    PowerGridUserSession *user, PowerGrid *env, int line, int unavailable)
{
    if (line < 0 || line >= POWER_GRID_NUM_BRANCHES)
        return POWER_GRID_INVALID_INPUT;

    unavailable = unavailable != 0;
    user->last_line = line;
    if (unavailable == user->line_outage[line])
        return env->solution.status;

    int outage_count = power_grid_user_outage_count(user);
    if (unavailable && outage_count >= user->max_outages)
        return user->last_status = POWER_GRID_INVALID_INPUT;
    /* An automatic outage is owned by the environment and cannot be restored
     * or reclassified by clicking it in the user controller. */
    if (unavailable && !env->line_available[line])
        return user->last_status = POWER_GRID_INVALID_INPUT;

    env->line_available[line] = !unavailable;
    user->line_outage[line] = unavailable;
    if (unavailable)
        env->topology.line_closed[line] = 0;
    /* Clearing an outage makes the line operable again without overriding a
     * topology choice made by the policy while it was unavailable. */

    PowerGridSolveStatus status = power_grid_solve_environment(env);
    if (unavailable && status != POWER_GRID_SOLVE_OK)
        env->emergency_recovery_steps = POWER_GRID_EMERGENCY_RECOVERY_STEPS;
    /* An infeasible click is precisely when the policy most needs the new
     * topology. Solver outputs are cleared on failure, while line availability
     * and switch state remain valid observation features. */
    power_grid_compute_observations(env);
    return user->last_status = status;
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
            user, env, selected, !user->line_outage[selected]);
}
#endif

#endif
