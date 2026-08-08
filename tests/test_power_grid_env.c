#define POWER_GRID_NO_RENDER

#include "power_grid_solver.c"
#include "power_grid_ac.c"
#include "power_grid.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)

#define CHECK_CLOSE(actual, expected) \
    CHECK(fabs((double)(actual) - (double)(expected)) < 1e-6)

static void test_action_application_and_bookkeeping(void)
{
    PowerGrid env = {0};
    power_grid_topology_normal(&env.topology);
    memset(env.line_available, 1, sizeof(env.line_available));

    PowerGridAppliedAction no_op = apply_agent_action(&env, 0.0f);
    CHECK(no_op.value == 0);
    CHECK(no_op.type == POWER_GRID_ACTION_NONE);
    CHECK(!no_op.switched);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 0);

    PowerGridAppliedAction line = apply_agent_action(&env, 1.0f);
    CHECK(line.type == POWER_GRID_ACTION_LINE);
    CHECK(line.switched);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 1);
    CHECK(env.episode.switches[POWER_GRID_ACTION_LINE] == 1);

    PowerGridAppliedAction invalid = apply_agent_action(&env, NAN);
    CHECK(invalid.type == POWER_GRID_ACTION_INVALID);
    CHECK(!invalid.switched);
    CHECK(env.episode.no_op_actions == 1);
    CHECK(env.episode.switches[0] == 1);
}

static void test_safety_cost_and_reward(void)
{
    PowerGridSolveResult solution = {0};
    PowerGridACSolveResult ac_solution = {0};
    solution.congestion_cost = 0.04;
    ac_solution.voltage_violation_cost = 0.09;

    CHECK_CLOSE(calculate_constraint_cost(&solution, &ac_solution, 0, 3), 0.04);
    CHECK_CLOSE(calculate_constraint_cost(&solution, &ac_solution, 1, 3), 3.13);

    CHECK_CLOSE(calculate_reward(POWER_GRID_INVALID_TOPOLOGY, 0.0, 0, 0),
                POWER_GRID_FAILURE_REWARD);
    CHECK_CLOSE(calculate_reward(POWER_GRID_SOLVE_OK, 0.0, 0, 1), 0.2);
    CHECK_CLOSE(calculate_reward(POWER_GRID_SOLVE_OK, 0.0, 1, 1), 0.199);
    CHECK_CLOSE(calculate_reward(POWER_GRID_SOLVE_OK, 0.04, 0, 0), -0.04);
}

static void test_episode_metrics(void)
{
    PowerGrid env = {0};
    power_grid_allocate(&env);
    env.rendering = 1;
    env.episode_step = 10;
    env.episode.safe_steps = 7;
    env.episode.no_op_actions = 8;
    env.solution.status = POWER_GRID_SOLVE_OK;
    power_grid_finish_episode(&env);
    CHECK_CLOSE(env.log.perf, 0.7);
    CHECK_CLOSE(env.log.score, 0.8);

    memset(&env.log, 0, sizeof(env.log));
    env.pending_reset = 0;
    env.solution.status = POWER_GRID_ISLANDED;
    power_grid_finish_episode(&env);
    CHECK_CLOSE(env.log.perf, 0.0);
    CHECK_CLOSE(env.log.score, 0.8);
    c_close(&env);
}

int main(void)
{
    test_action_application_and_bookkeeping();
    test_safety_cost_and_reward();
    test_episode_metrics();
    if (failures)
        return 1;
    puts("power-grid environment tests passed");
    return 0;
}
