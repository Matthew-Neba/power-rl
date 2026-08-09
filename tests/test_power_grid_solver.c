#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>

static int failures;

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)

static void check_near(double actual, double expected, double tolerance)
{
    if (!isfinite(actual) || fabs(actual - expected) > tolerance)
    {
        fprintf(stderr, "FAIL %.12f != %.12f (tol %.3g)\n", actual, expected, tolerance);
        failures++;
    }
}

static double total_load(const PowerGridOperatingPoint *point)
{
    double result = 0.0;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
        result += point->load_mw[load];
    return result;
}

static void check_kirchhoff(const PowerGridTopology *topology,
                            const PowerGridOperatingPoint *point,
                            const PowerGridSolveResult *result)
{
    double injection[POWER_GRID_NUM_NODES] = {0};
    double outgoing[POWER_GRID_NUM_NODES] = {0};
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
    {
        int node = power_grid_terminal_node(topology,
            POWER_GRID_GENERATOR_TERMINAL(generator),
            POWER_GRID_GENERATOR_BUSES[generator]);
        injection[node] += generator == 0 ? result->slack_generation_mw :
                                            point->generator_mw[generator];
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
    {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
                                            POWER_GRID_LOAD_BUSES[load]);
        injection[node] -= point->load_mw[load];
    }
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        if (!topology->line_closed[line])
            continue;
        const PowerGridBranch *branch = &POWER_GRID_BRANCHES[line];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
                                            branch->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
                                          branch->to_bus);
        outgoing[from] += result->branch_flow_mw[line];
        outgoing[to] -= result->branch_flow_mw[line];
        check_near(result->node_angle[from] - result->node_angle[to],
            result->branch_flow_mw[line] * branch->reactance * branch->tap_ratio /
                POWER_GRID_BASE_MVA, 1e-9);
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
        check_near(injection[node], outgoing[node], 2e-7);
}

static void test_dimensions_and_nominal_solution(void)
{
    CHECK(POWER_GRID_NUM_SUBSTATIONS == 118);
    CHECK(POWER_GRID_NUM_BRANCHES == 186);
    CHECK(POWER_GRID_NUM_GENERATORS == 54);
    CHECK(POWER_GRID_NUM_LOADS == 99);
    CHECK(POWER_GRID_NUM_TERMINALS == 525);
    CHECK(POWER_GRID_NUM_ACTIONS == 830);
    CHECK(POWER_GRID_GENERATOR_BUSES[0] == POWER_GRID_IEEE118_SLACK_BUS);

    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    check_near(total_load(&point), 4242.0, 1e-9);
    check_near(result.slack_generation_mw, 381.0, 1e-9);
    CHECK(result.component_count == 1);
    CHECK(result.active_node_count == 118);
    CHECK(result.max_rho < 1.0);
    check_kirchhoff(&topology, &point, &result);
}

static void test_profiles(void)
{
    PowerGridTopology topology;
    power_grid_topology_normal(&topology);
    double nominal_load = 0.0;
    for (int profile = 0; profile < POWER_GRID_NUM_PROFILES; profile++)
    {
        PowerGridOperatingPoint point;
        PowerGridSolveResult result;
        power_grid_operating_point_profile(&point, (PowerGridProfile)profile);
        CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
        CHECK(power_grid_profile_name((PowerGridProfile)profile) != NULL);
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            CHECK(point.load_mw[load] >= 0.0);
        if (profile == POWER_GRID_PROFILE_P0_NOMINAL)
            nominal_load = total_load(&point);
        if (profile == POWER_GRID_PROFILE_P1_HIGH)
            check_near(total_load(&point), 1.15 * nominal_load, 1e-8);
        if (profile == POWER_GRID_PROFILE_P2_LOW)
            check_near(total_load(&point), 0.95 * nominal_load, 1e-8);
    }
}

static void test_contingency_eligibility(void)
{
    PowerGridOperatingPoint point;
    power_grid_operating_point_nominal(&point);
    int eligible = 0, bridges = 0;
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        PowerGridTopology topology;
        PowerGridSolveResult result;
        power_grid_topology_normal(&topology);
        topology.line_closed[line] = 0;
        PowerGridSolveStatus status = power_grid_solve(&topology, &point, &result);
        if (power_grid_random_event_eligible(line))
        {
            eligible++;
            CHECK(status == POWER_GRID_SOLVE_OK);
        }
        else
        {
            bridges++;
            CHECK(status != POWER_GRID_SOLVE_OK);
        }
    }
    CHECK(eligible == POWER_GRID_NUM_BRANCHES - POWER_GRID_IEEE118_BRIDGE_COUNT);
    CHECK(bridges == POWER_GRID_IEEE118_BRIDGE_COUNT);
}

static void test_busbar_actions(void)
{
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult before, after;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    CHECK(power_grid_solve(&topology, &point, &before) == POWER_GRID_SOLVE_OK);

    int load_bus = POWER_GRID_LOAD_BUSES[0];
    topology.terminal_busbar[POWER_GRID_LOAD_TERMINAL(0)] = 1;
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_DISCONNECTED_LOAD);
    topology.coupler_closed[load_bus] = 1;
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
        check_near(after.branch_flow_mw[line], before.branch_flow_mw[line], 1e-8);

    CHECK(power_grid_apply_action(&topology, -1) == POWER_GRID_ACTION_INVALID);
    CHECK(power_grid_apply_action(&topology, POWER_GRID_NUM_ACTIONS) ==
          POWER_GRID_ACTION_INVALID);
}

int main(void)
{
    test_dimensions_and_nominal_solution();
    test_profiles();
    test_contingency_eligibility();
    test_busbar_actions();
    if (failures)
        return 1;
    puts("IEEE-118 DC solver tests passed");
    return 0;
}
