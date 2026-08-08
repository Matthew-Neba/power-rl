#include "power_grid_solver.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failures++; \
    } \
} while (0)

static void check_near(double actual, double expected, double tolerance) {
    if (fabs(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL: %.12f != %.12f (tolerance %.3g)\n", actual, expected, tolerance);
        failures++;
    }
}

static double total_load(const PowerGridOperatingPoint* point) {
    double total = 0.0;
    for (int i = 0; i < POWER_GRID_NUM_LOADS; i++) total += point->load_mw[i];
    return total;
}

static double total_non_slack_generation(const PowerGridOperatingPoint* point) {
    double total = 0.0;
    for (int i = 1; i < POWER_GRID_NUM_GENERATORS; i++) total += point->generator_mw[i];
    return total;
}

static void check_kirchhoff_laws(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, const PowerGridSolveResult* result) {
    double node_injection[POWER_GRID_NUM_NODES] = {0};
    double outgoing_flow[POWER_GRID_NUM_NODES] = {0};
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        node_injection[node] += gen == 0 ? result->slack_generation_mw : point->generator_mw[gen];
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        node_injection[node] -= point->load_mw[load];
    }
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!topology->line_closed[line]) {
            check_near(result->branch_flow_mw[line], 0.0, 0.0);
            continue;
        }
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
            branch->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
            branch->to_bus);
        outgoing_flow[from] += result->branch_flow_mw[line];
        outgoing_flow[to] -= result->branch_flow_mw[line];

        /* Branch angle drop is the DC form of Kirchhoff's voltage law. */
        double angle_drop_from_flow = result->branch_flow_mw[line] * branch->reactance *
            branch->tap_ratio / POWER_GRID_BASE_MVA;
        check_near(result->node_angle[from] - result->node_angle[to], angle_drop_from_flow, 1e-10);
    }
    /* Kirchhoff's current law: injection equals net outgoing flow at every busbar. */
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        check_near(node_injection[node], outgoing_flow[node], 1e-8);
    }
}

static void test_nominal_balance_and_flow(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);

    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    check_near(total_load(&point), 259.0, 1e-9);
    check_near(result.slack_generation_mw, 219.0, 1e-9);
    check_near(result.branch_flow_mw[0] + result.branch_flow_mw[1], 219.0, 1e-8);
    CHECK(result.component_count == 1);
    CHECK(result.active_node_count == 14);
    check_kirchhoff_laws(&topology, &point, &result);
    for (int i = 0; i < POWER_GRID_NUM_NODES; i++) CHECK(isfinite(result.node_angle[i]));
    for (int i = 0; i < POWER_GRID_NUM_BRANCHES; i++) {
        CHECK(isfinite(result.branch_flow_mw[i]));
        check_near(result.branch_rho[i], fabs(result.branch_flow_mw[i]) /
            POWER_GRID_BRANCHES[i].thermal_limit_mw, 1e-12);
    }

    /* Values independently calculated from the MATPOWER makeBdc formulation. */
    static const double reference_flows[POWER_GRID_NUM_BRANCHES] = {
        147.838595558910, 71.161404441091, 70.014635958331, 55.151852703177,
        40.972106897401, -24.185364041669, -61.746490650773, 28.361152787811,
        16.551826524471, 42.787020687719, 6.728345804468, 7.607358142264,
        17.251316740986, 0.0, 28.361152787811, 5.771654195532,
        9.641325116750, -3.228345804468, 1.507358142264, 5.258674883250,
    };
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        check_near(result.branch_flow_mw[line], reference_flows[line], 1e-8);
    }
}

static void test_profiles_are_balanced_by_slack(void) {
    PowerGridTopology topology;
    power_grid_topology_normal(&topology);
    for (int profile = 0; profile < POWER_GRID_NUM_PROFILES; profile++) {
        PowerGridOperatingPoint point;
        PowerGridSolveResult result;
        power_grid_operating_point_profile(&point, (PowerGridProfile)profile);
        CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
        check_near(result.slack_generation_mw,
            total_load(&point) - total_non_slack_generation(&point), 1e-9);
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) CHECK(point.load_mw[load] >= 0.0);
    }
}

static void test_stress_profile_locations(void) {
    static const int expected_branch[] = {9, 7, 15, 17, 16, 5, 6, 4, 10, 12, 0};
    PowerGridTopology topology;
    power_grid_topology_normal(&topology);
    for (int profile = POWER_GRID_PROFILE_P4_BUS6_LOAD_SHIFT;
            profile <= POWER_GRID_PROFILE_P14_BUS2_LOCAL_PEAK; profile++) {
        PowerGridOperatingPoint point;
        PowerGridSolveResult result;
        power_grid_operating_point_profile(&point, (PowerGridProfile)profile);
        CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
        CHECK(result.max_rho > 1.0);
        int branch = expected_branch[profile - POWER_GRID_PROFILE_P4_BUS6_LOAD_SHIFT];
        CHECK(result.branch_rho[branch] == result.max_rho);
        CHECK(power_grid_profile_name((PowerGridProfile)profile) != NULL);
    }
}

static void test_open_line_and_island_detection(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);

    CHECK(power_grid_apply_action(&topology, POWER_GRID_LINE_ACTION_OFFSET + 13) ==
        POWER_GRID_ACTION_LINE); /* 7-8 is a bridge to generator bus 8. */
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_DISCONNECTED_GENERATOR);

    power_grid_topology_normal(&topology);
    CHECK(power_grid_apply_action(&topology, POWER_GRID_LINE_ACTION_OFFSET + 16) ==
        POWER_GRID_ACTION_LINE); /* 9-14 has a parallel path through 13-14. */
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    check_near(result.branch_flow_mw[16], 0.0, 0.0);
    check_near(result.branch_rho[16], 0.0, 0.0);
}

static void test_unsafe_terminal_moves(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);

    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_LOAD_TERMINAL(0)) ==
        POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_DISCONNECTED_LOAD);

    power_grid_topology_normal(&topology);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_GENERATOR_TERMINAL(0)) ==
        POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_DISCONNECTED_LOAD);
}

static void test_busbar_reconfiguration(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult before;
    PowerGridSolveResult after;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_profile(&point, POWER_GRID_PROFILE_P1_HIGH);
    CHECK(power_grid_solve(&topology, &point, &before) == POWER_GRID_SOLVE_OK);

    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_LINE_TERMINAL(3, 0)) ==
        POWER_GRID_ACTION_TERMINAL); /* bus-2 end 2-4 */
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_LINE_TERMINAL(4, 0)) ==
        POWER_GRID_ACTION_TERMINAL); /* bus-2 end 2-5 */
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    check_kirchhoff_laws(&topology, &point, &after);
    CHECK(before.congestion_cost > 0.0);
    check_near(after.congestion_cost, 0.0, 0.0);
    CHECK(after.max_rho < 1.0);
    check_near(after.branch_rho[0], 0.6648744943885827, 1e-10);
}

static void test_coupler_allows_safe_radial_transfer(void) {
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult before;
    PowerGridSolveResult after;
    const int substation_8 = 7;
    const int line_7_8_bus_8_terminal = POWER_GRID_LINE_TERMINAL(13, 1);
    const int generator_8_terminal = POWER_GRID_GENERATOR_TERMINAL(4);
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    CHECK(power_grid_solve(&topology, &point, &before) == POWER_GRID_SOLVE_OK);

    /* Equipment on opposite bars remains on the same electrical node while coupled. */
    CHECK(power_grid_apply_action(&topology, POWER_GRID_COUPLER_ACTION_OFFSET + 1) ==
        POWER_GRID_ACTION_COUPLER);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_LOAD_TERMINAL(0)) ==
        POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        check_near(after.branch_flow_mw[line], before.branch_flow_mw[line], 1e-8);
    }
    CHECK(power_grid_apply_action(&topology, POWER_GRID_COUPLER_ACTION_OFFSET + 1) ==
        POWER_GRID_ACTION_COUPLER);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_DISCONNECTED_LOAD);
    CHECK(power_grid_apply_action(&topology, POWER_GRID_COUPLER_ACTION_OFFSET + 1) ==
        POWER_GRID_ACTION_COUPLER);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_LOAD_TERMINAL(0)) ==
        POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_apply_action(&topology, POWER_GRID_COUPLER_ACTION_OFFSET + 1) ==
        POWER_GRID_ACTION_COUPLER);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);

    /* Moving either radial terminal alone would disconnect generator 8. */
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + generator_8_terminal) == POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_DISCONNECTED_GENERATOR);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + generator_8_terminal) == POWER_GRID_ACTION_TERMINAL);

    /* The closed coupler keeps both bars one electrical node during the transfer. */
    CHECK(power_grid_apply_action(&topology, POWER_GRID_COUPLER_ACTION_OFFSET + substation_8) ==
        POWER_GRID_ACTION_COUPLER);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + generator_8_terminal) == POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    CHECK(power_grid_apply_action(&topology,
        POWER_GRID_TERMINAL_ACTION_OFFSET + line_7_8_bus_8_terminal) ==
        POWER_GRID_ACTION_TERMINAL);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    CHECK(topology.terminal_busbar[generator_8_terminal] == 1);
    CHECK(topology.terminal_busbar[line_7_8_bus_8_terminal] == 1);

    /* Opening the coupler completes an energized, topology-equivalent transfer to BB2. */
    CHECK(power_grid_apply_action(&topology, POWER_GRID_COUPLER_ACTION_OFFSET + substation_8) ==
        POWER_GRID_ACTION_COUPLER);
    CHECK(power_grid_solve(&topology, &point, &after) == POWER_GRID_SOLVE_OK);
    check_kirchhoff_laws(&topology, &point, &after);
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        check_near(after.branch_flow_mw[line], before.branch_flow_mw[line], 1e-8);
    }
}

static void test_stable_names(void) {
    char name[128];
    CHECK(strcmp(power_grid_action_name(0, name, sizeof(name)), "do-nothing") == 0);
    CHECK(strcmp(power_grid_action_name(1, name, sizeof(name)), "toggle-line-1-2") == 0);
    CHECK(strstr(power_grid_action_name(21, name, sizeof(name)), "line-1-2-from-end") != NULL);
    CHECK(strstr(power_grid_action_name(61, name, sizeof(name)), "generator-0-bus-1") != NULL);
    CHECK(strstr(power_grid_action_name(66, name, sizeof(name)), "load-0-bus-2") != NULL);
    CHECK(strcmp(power_grid_action_name(77, name, sizeof(name)),
        "toggle-substation-1-coupler") == 0);
    CHECK(strcmp(power_grid_action_name(90, name, sizeof(name)),
        "toggle-substation-14-coupler") == 0);
    CHECK(strstr(power_grid_action_name(91, name, sizeof(name)), "invalid-action") != NULL);
}

int main(void) {
    test_nominal_balance_and_flow();
    test_profiles_are_balanced_by_slack();
    test_stress_profile_locations();
    test_open_line_and_island_detection();
    test_unsafe_terminal_moves();
    test_busbar_reconfiguration();
    test_coupler_allows_safe_radial_transfer();
    test_stable_names();
    if (failures != 0) {
        fprintf(stderr, "%d power-grid solver test(s) failed\n", failures);
        return 1;
    }
    puts("power-grid solver tests passed");
    return 0;
}
