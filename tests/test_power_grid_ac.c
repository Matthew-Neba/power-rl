#include "power_grid_ac.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void check_near(double actual, double expected, double tolerance) {
    if (!isfinite(actual) || fabs(actual - expected) > tolerance) {
        fprintf(stderr, "expected %.12f, got %.12f (tolerance %.3g)\n",
            expected, actual, tolerance);
        exit(1);
    }
}

static void check_complex_power_balance(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, const PowerGridACSolveResult* result) {
    double injection_p[POWER_GRID_NUM_NODES] = {0};
    double injection_q[POWER_GRID_NUM_NODES] = {0};
    double outgoing_p[POWER_GRID_NUM_NODES] = {0};
    double outgoing_q[POWER_GRID_NUM_NODES] = {0};
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        injection_p[node] += result->generator_p_mw[gen];
        injection_q[node] += result->generator_q_mvar[gen];
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        injection_p[node] -= point->load_mw[load];
        injection_q[node] -= power_grid_ac_load_q_mvar(point, load);
    }
    int bus9_node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(5), 8);
    injection_q[bus9_node] += 19.0 * result->node_voltage_pu[bus9_node] *
        result->node_voltage_pu[bus9_node];
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!topology->line_closed[line]) continue;
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
            POWER_GRID_BRANCHES[line].from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
            POWER_GRID_BRANCHES[line].to_bus);
        outgoing_p[from] += result->branch_from_p_mw[line];
        outgoing_q[from] += result->branch_from_q_mvar[line];
        outgoing_p[to] += result->branch_to_p_mw[line];
        outgoing_q[to] += result->branch_to_q_mvar[line];
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        check_near(injection_p[node], outgoing_p[node], 2e-5);
        check_near(injection_q[node], outgoing_q[node], 2e-5);
    }
}

static void test_nominal_matpower_reference(void) {
    static const double voltage_reference[POWER_GRID_NUM_SUBSTATIONS] = {
        1.060, 1.045, 1.010, 1.019, 1.020, 1.070, 1.062,
        1.090, 1.056, 1.051, 1.057, 1.055, 1.050, 1.036,
    };
    static const double angle_reference_deg[POWER_GRID_NUM_SUBSTATIONS] = {
        0, -4.98, -12.72, -10.33, -8.78, -14.22, -13.37,
        -13.36, -14.94, -15.10, -14.79, -15.07, -15.16, -16.04,
    };
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridACSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
    CHECK(result.converged);
    CHECK(result.active_node_count == 14);
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++) {
        check_near(result.node_voltage_pu[2 * bus], voltage_reference[bus], 0.0025);
        check_near(result.node_angle_rad[2 * bus] * 180.0 / 3.141592653589793,
            angle_reference_deg[bus], 0.12);
    }
    check_near(result.generator_p_mw[0], 232.4, 0.5);
    static const double generator_q_reference[POWER_GRID_NUM_GENERATORS] = {
        -16.55, 43.56, 25.08, 12.73, 17.62,
    };
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        check_near(result.generator_q_mvar[gen], generator_q_reference[gen], 0.15);
    }
    check_near(result.branch_from_p_mw[0], 156.9, 0.3);
    check_near(result.branch_from_p_mw[1], 75.5, 0.3);
    CHECK(result.total_p_loss_mw > 12.0 && result.total_p_loss_mw < 15.0);
    check_near(result.generator_p_mw[0] + 40.0, 259.0 + result.total_p_loss_mw, 1e-5);
    CHECK(result.min_voltage_pu > 1.0);
    CHECK(result.max_voltage_pu <= 1.10);
    CHECK(result.voltage_violation_count == 0);
    CHECK(result.max_rho <= 1.0);
    check_complex_power_balance(&topology, &point, &result);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        CHECK(isfinite(result.branch_from_p_mw[line]));
        CHECK(isfinite(result.branch_from_q_mvar[line]));
        check_near(result.branch_rho[line], fmax(result.branch_from_mva[line],
            result.branch_to_mva[line]) / power_grid_ac_branch_rating_mva(line), 1e-12);
    }
}

static void test_open_line_and_busbar_topology(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridACSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    topology.line_closed[16] = 0;
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
    check_near(result.branch_from_mva[16], 0.0, 0.0);
    check_near(result.branch_to_mva[16], 0.0, 0.0);

    power_grid_topology_normal(&topology);
    topology.terminal_busbar[POWER_GRID_LOAD_TERMINAL(0)] = 1;
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_TOPOLOGY_FAILURE);
    CHECK(result.topology_status == POWER_GRID_DISCONNECTED_LOAD);

    power_grid_topology_normal(&topology);
    topology.coupler_closed[1] = 1;
    topology.terminal_busbar[POWER_GRID_LOAD_TERMINAL(0)] = 1;
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
    check_complex_power_balance(&topology, &point, &result);

    /* The known DC P1 split creates a genuine fifteenth electrical node in AC too. */
    power_grid_topology_normal(&topology);
    power_grid_operating_point_profile(&point, POWER_GRID_PROFILE_P1_HIGH);
    topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(3, 0)] = 1;
    topology.terminal_busbar[POWER_GRID_LINE_TERMINAL(4, 0)] = 1;
    CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
    CHECK(result.active_node_count == 15);
    check_complex_power_balance(&topology, &point, &result);
}

static void test_profiles_report_real_ac_constraints(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridACSolveResult result;
    power_grid_topology_normal(&topology);
    for (int profile = 0; profile < POWER_GRID_NUM_PROFILES; profile++) {
        power_grid_operating_point_profile(&point, (PowerGridProfile)profile);
        CHECK(power_grid_ac_solve(&topology, &point, &result) == POWER_GRID_AC_OK);
        CHECK(result.converged);
        CHECK(result.iterations > 0);
        CHECK(result.total_p_loss_mw >= 0.0);
        if (profile == POWER_GRID_PROFILE_P9_BUS3_GENERATION) {
            CHECK(result.generator_p_violation_count >= 1);
            CHECK(result.generator_p_violation_mw >= 69.9);
        }
    }
}

static void test_inverse_time_thermal_protection(void) {
    double stress = 0.0;
    stress = power_grid_ac_thermal_step(stress, 2.0);
    check_near(stress, 1.0, 1e-12);

    stress = 0.0;
    for (int step = 0; step < 4; step++) stress = power_grid_ac_thermal_step(stress, 1.5);
    check_near(stress, 1.0, 1e-12);
    stress = power_grid_ac_thermal_step(stress, 0.8);
    check_near(stress, 0.98, 1e-12);
}

static void test_connected_nominal_line_contingencies(void) {
    PowerGridOperatingPoint point;
    power_grid_operating_point_nominal(&point);
    for (int outage = 0; outage < POWER_GRID_NUM_BRANCHES; outage++) {
        PowerGridTopology topology;
        PowerGridACSolveResult result;
        power_grid_topology_normal(&topology);
        topology.line_closed[outage] = 0;
        PowerGridSolveStatus topology_status = power_grid_validate_topology(&topology, NULL, NULL);
        PowerGridACStatus ac_status = power_grid_ac_solve(&topology, &point, &result);
        if (topology_status == POWER_GRID_SOLVE_OK) {
            if (outage == 0) {
                /* An unconstrained AC solve converges, but all four non-slack PV
                 * generators exceed Q limits. Enforcing those limits does not. */
                CHECK(ac_status == POWER_GRID_AC_DIVERGED);
                CHECK(result.q_limit_count == 4);
            } else {
                if (ac_status != POWER_GRID_AC_OK) fprintf(stderr,
                    "connected outage %s AC status %s\n", POWER_GRID_BRANCH_NAMES[outage],
                    power_grid_ac_status_name(ac_status));
                CHECK(ac_status == POWER_GRID_AC_OK);
                check_complex_power_balance(&topology, &point, &result);
            }
        } else {
            CHECK(ac_status == POWER_GRID_AC_TOPOLOGY_FAILURE);
            CHECK(result.topology_status == topology_status);
        }
    }
}

int main(void) {
    test_nominal_matpower_reference();
    test_open_line_and_busbar_topology();
    test_profiles_report_real_ac_constraints();
    test_inverse_time_thermal_protection();
    test_connected_nominal_line_contingencies();
    puts("power-grid AC solver tests passed");
    return 0;
}
