#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

static void check_near(double actual, double expected, double tolerance)
{
    if (!isfinite(actual) || fabs(actual - expected) > tolerance)
    {
        fprintf(stderr, "FAIL %.12f != %.12f (tol %.3g)\n", actual, expected, tolerance);
        exit(1);
    }
}

static void check_complex_balance(const PowerGridTopology *topology,
                                  const PowerGridOperatingPoint *point,
                                  const PowerGridACSolveResult *result)
{
    double injection_p[POWER_GRID_NUM_NODES] = {0};
    double injection_q[POWER_GRID_NUM_NODES] = {0};
    double outgoing_p[POWER_GRID_NUM_NODES] = {0};
    double outgoing_q[POWER_GRID_NUM_NODES] = {0};
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++)
    {
        int node = power_grid_terminal_node(topology,
            POWER_GRID_GENERATOR_TERMINAL(generator),
            POWER_GRID_GENERATOR_BUSES[generator]);
        injection_p[node] += result->generator_p_mw[generator];
        injection_q[node] += result->generator_q_mvar[generator];
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
    {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
                                            POWER_GRID_LOAD_BUSES[load]);
        injection_p[node] -= point->load_mw[load];
        injection_q[node] -= power_grid_ac_load_q_mvar(point, load);
    }
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        int node = 2 * bus;
        injection_q[node] += POWER_GRID_BUS_SHUNT_B_MVAR[bus] *
                             result->node_voltage_pu[node] * result->node_voltage_pu[node];
    }
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        if (!topology->line_closed[line])
            continue;
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
                                            POWER_GRID_BRANCHES[line].from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
                                          POWER_GRID_BRANCHES[line].to_bus);
        outgoing_p[from] += result->branch_from_p_mw[line];
        outgoing_q[from] += result->branch_from_q_mvar[line];
        outgoing_p[to] += result->branch_to_p_mw[line];
        outgoing_q[to] += result->branch_to_q_mvar[line];
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
    {
        check_near(injection_p[node], outgoing_p[node], 3e-5);
        check_near(injection_q[node], outgoing_q[node], 3e-5);
    }
}

static void test_nominal_reference(void)
{
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridACSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
    CHECK(result.converged);
    CHECK(result.active_node_count == 14);
    CHECK(result.voltage_violation_count == 0);
    CHECK(result.max_rho < 1.0);
    CHECK(result.total_p_loss_mw > 12.0 && result.total_p_loss_mw < 15.0);
    double angle_shift = result.node_angle_rad[2 * POWER_GRID_IEEE14_SLACK_BUS] *
                         180.0 / 3.141592653589793 -
                         POWER_GRID_BUS_ANGLE_REFERENCE_DEG[POWER_GRID_IEEE14_SLACK_BUS];
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        check_near(result.node_voltage_pu[2 * bus], POWER_GRID_BUS_V_REFERENCE[bus], 0.02);
        check_near(result.node_angle_rad[2 * bus] * 180.0 / 3.141592653589793 -
                   angle_shift, POWER_GRID_BUS_ANGLE_REFERENCE_DEG[bus], 0.35);
    }
    check_complex_balance(&topology, &point, &result);
}

static void test_topology_and_contingencies(void)
{
    PowerGridOperatingPoint point;
    power_grid_operating_point_nominal(&point);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
    {
        PowerGridTopology topology;
        PowerGridACSolveResult result;
        power_grid_topology_normal(&topology);
        topology.line_closed[line] = 0;
        PowerGridACStatus status = power_grid_ac_solve(&topology, &point, &result);
        CHECK(power_grid_random_event_eligible(line));
        if (result.topology_status == POWER_GRID_SOLVE_OK)
        {
            if (status == POWER_GRID_AC_OK)
                check_complex_balance(&topology, &point, &result);
            else
                CHECK(status == POWER_GRID_AC_DIVERGED ||
                      status == POWER_GRID_AC_SINGULAR);
        }
        else
        {
            CHECK(status == POWER_GRID_AC_TOPOLOGY_FAILURE);
        }
    }

    PowerGridTopology topology;
    PowerGridACSolveResult result;
    power_grid_topology_normal(&topology);
    int load_bus = POWER_GRID_LOAD_BUSES[0];
    topology.terminal_busbar[POWER_GRID_LOAD_TERMINAL(0)] = 1;
    CHECK(power_grid_ac_solve(&topology, &point, &result) ==
          POWER_GRID_AC_TOPOLOGY_FAILURE);
    topology.coupler_closed[load_bus] = 1;
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
}

static void test_profiles_and_thermal_model(void)
{
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridACSolveResult result;
    power_grid_topology_normal(&topology);
    for (int profile = 0; profile < POWER_GRID_NUM_PROFILES; profile++)
    {
        power_grid_operating_point_profile(&point, (PowerGridProfile)profile);
        CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
        CHECK(result.iterations > 0);
    }
    double stress = power_grid_ac_thermal_step(0.0, 2.0);
    check_near(stress, 1.0, 1e-12);
    stress = 0.0;
    for (int step = 0; step < 4; step++)
        stress = power_grid_ac_thermal_step(stress, 1.5);
    check_near(stress, 1.0, 1e-12);
    check_near(power_grid_ac_thermal_step(stress, 0.8), 0.98, 1e-12);
}

int main(void)
{
    test_nominal_reference();
    test_topology_and_contingencies();
    test_profiles_and_thermal_model();
    puts("IEEE-14 AC solver tests passed");
    return 0;
}
