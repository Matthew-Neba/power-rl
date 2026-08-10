#include "power_grid_solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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

/* Independent reference implementation: assemble the reduced DC nodal
 * admittance matrix densely and solve it with pivoted Gaussian elimination.
 * This intentionally does not use the production ordering, sparse Cholesky,
 * symbolic pattern, or factor cache. */
static PowerGridSolveStatus solve_dense_reference(
        const PowerGridTopology *topology, const PowerGridOperatingPoint *point,
        PowerGridSolveResult *result, double rating_scale)
{
    unsigned char active[POWER_GRID_NUM_NODES] = {0};
    int row_for_node[POWER_GRID_NUM_NODES];
    double injection[POWER_GRID_NUM_NODES] = {0};
    double matrix[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES] = {{0}};
    double rhs[POWER_GRID_NUM_NODES] = {0};
    double angles[POWER_GRID_NUM_NODES] = {0};
    memset(result, 0, sizeof(*result));
    result->status = power_grid_validate_topology(topology,
        &result->component_count, &result->active_node_count);
    if (result->status != POWER_GRID_SOLVE_OK)
        return result->status;

    double load_total = 0.0;
    double non_slack_generation = 0.0;
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        if (!isfinite(point->load_mw[load]) || point->load_mw[load] < 0.0)
            return result->status = POWER_GRID_INVALID_INPUT;
        load_total += point->load_mw[load];
    }
    for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++) {
        if (!isfinite(point->generator_mw[generator]) ||
                point->generator_mw[generator] < 0.0)
            return result->status = POWER_GRID_INVALID_INPUT;
        non_slack_generation += point->generator_mw[generator];
    }
    result->slack_generation_mw = load_total - non_slack_generation;
    if (!isfinite(result->slack_generation_mw) ||
            result->slack_generation_mw < 0.0)
        return result->status = POWER_GRID_INVALID_INPUT;

    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch *line = &POWER_GRID_BRANCHES[branch];
        active[power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus)] = 1;
        active[power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus)] = 1;
    }
    for (int generator = 0; generator < POWER_GRID_NUM_GENERATORS; generator++) {
        int node = power_grid_terminal_node(topology,
            POWER_GRID_GENERATOR_TERMINAL(generator),
            POWER_GRID_GENERATOR_BUSES[generator]);
        active[node] = 1;
        double output = generator == 0 ? result->slack_generation_mw :
                                         point->generator_mw[generator];
        injection[node] += output;
        result->substation_injection_mw[POWER_GRID_GENERATOR_BUSES[generator]] += output;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
                                            POWER_GRID_LOAD_BUSES[load]);
        active[node] = 1;
        injection[node] -= point->load_mw[load];
        result->substation_injection_mw[POWER_GRID_LOAD_BUSES[load]] -= point->load_mw[load];
    }

    int reference = power_grid_terminal_node(topology,
        POWER_GRID_GENERATOR_TERMINAL(0), POWER_GRID_GENERATOR_BUSES[0]);
    int dimensions = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        row_for_node[node] = active[node] && node != reference ? dimensions++ : -1;
    }
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch *line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus);
        int to = power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus);
        int from_row = row_for_node[from];
        int to_row = row_for_node[to];
        double susceptance = POWER_GRID_BASE_MVA /
            (line->reactance * line->tap_ratio);
        if (from_row >= 0) matrix[from_row][from_row] += susceptance;
        if (to_row >= 0) matrix[to_row][to_row] += susceptance;
        if (from_row >= 0 && to_row >= 0) {
            matrix[from_row][to_row] -= susceptance;
            matrix[to_row][from_row] -= susceptance;
        }
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
        if (row_for_node[node] >= 0)
            rhs[row_for_node[node]] = injection[node];
    if (!power_grid_solve_dense(&matrix[0][0], rhs, angles, dimensions,
                                POWER_GRID_NUM_NODES))
        return result->status = POWER_GRID_SINGULAR;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
        if (row_for_node[node] >= 0)
            result->node_angle[node] = angles[row_for_node[node]];

    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        if (!topology->line_closed[branch]) continue;
        const PowerGridBranch *line = &POWER_GRID_BRANCHES[branch];
        int from = power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 0), line->from_bus);
        int to = power_grid_terminal_node(topology,
            POWER_GRID_LINE_TERMINAL(branch, 1), line->to_bus);
        double flow = POWER_GRID_BASE_MVA *
            (result->node_angle[from] - result->node_angle[to]) /
            (line->reactance * line->tap_ratio);
        double rho = fabs(flow) / (line->thermal_limit_mw * rating_scale);
        if (!isfinite(flow) || !isfinite(rho))
            return result->status = POWER_GRID_NONFINITE;
        result->branch_flow_mw[branch] = flow;
        result->branch_rho[branch] = rho;
        if (rho > result->max_rho) result->max_rho = rho;
        if (rho > 1.0) {
            double overload = rho - 1.0;
            result->congestion_cost += overload * overload;
            result->overloaded_branch_count++;
        }
    }
    return result->status = POWER_GRID_SOLVE_OK;
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

static PowerGridSolveStatus compare_with_dense_reference(const PowerGridTopology *topology,
        const PowerGridOperatingPoint *point, double rating_scale)
{
    PowerGridSolveResult optimized, reference;
    PowerGridSolveStatus optimized_status = power_grid_solve_scaled(
        topology, point, &optimized, rating_scale);
    PowerGridSolveStatus reference_status = solve_dense_reference(
        topology, point, &reference, rating_scale);
    CHECK(optimized_status == reference_status);
    CHECK(optimized.component_count == reference.component_count);
    CHECK(optimized.active_node_count == reference.active_node_count);
    if (optimized_status != POWER_GRID_SOLVE_OK) return optimized_status;

    check_near(optimized.slack_generation_mw, reference.slack_generation_mw, 1e-9);
    check_near(optimized.max_rho, reference.max_rho, 2e-7);
    check_near(optimized.congestion_cost, reference.congestion_cost, 2e-7);
    CHECK(optimized.overloaded_branch_count == reference.overloaded_branch_count);
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++)
        check_near(optimized.node_angle[node], reference.node_angle[node], 2e-7);
    for (int branch = 0; branch < POWER_GRID_NUM_BRANCHES; branch++) {
        check_near(optimized.branch_flow_mw[branch],
                   reference.branch_flow_mw[branch], 2e-5);
        check_near(optimized.branch_rho[branch],
                   reference.branch_rho[branch], 2e-7);
    }
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
        check_near(optimized.substation_injection_mw[bus],
                   reference.substation_injection_mw[bus], 1e-9);
    check_kirchhoff(topology, point, &optimized);
    return optimized_status;
}

static void test_dimensions_and_nominal_solution(void)
{
    CHECK(POWER_GRID_NUM_SUBSTATIONS == 14);
    CHECK(POWER_GRID_NUM_BRANCHES == 20);
    CHECK(POWER_GRID_NUM_GENERATORS == 5);
    CHECK(POWER_GRID_NUM_LOADS == 11);
    CHECK(POWER_GRID_NUM_TERMINALS == 56);
    CHECK(POWER_GRID_NUM_ACTIONS == 91);
    CHECK(POWER_GRID_GENERATOR_BUSES[0] == POWER_GRID_IEEE14_SLACK_BUS);

    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    PowerGridSolveResult result;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    CHECK(power_grid_solve(&topology, &point, &result) == POWER_GRID_SOLVE_OK);
    check_near(total_load(&point), 259.0, 1e-9);
    check_near(result.slack_generation_mw, 219.0, 1e-9);
    CHECK(result.component_count == 1);
    CHECK(result.active_node_count == 14);
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
        compare_with_dense_reference(&topology, &point, 1.0);
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
    CHECK(eligible == POWER_GRID_NUM_BRANCHES - POWER_GRID_IEEE14_BRIDGE_COUNT);
    CHECK(bridges == POWER_GRID_IEEE14_BRIDGE_COUNT);
}

static void test_randomized_dense_parity(void)
{
    PowerGridTopology topology;
    power_grid_topology_normal(&topology);
    unsigned int random = 0x738014u;
    int valid_transitions = 0;
    int invalid_transitions = 0;

    for (int trial = 0; trial < 128; trial++) {
        random = random * 1664525u + 1013904223u;
        int action = 1 + (int)(random % (POWER_GRID_NUM_ACTIONS - 1));
        PowerGridTopology candidate = topology;
        CHECK(power_grid_apply_action(&candidate, action) != POWER_GRID_ACTION_INVALID);

        PowerGridOperatingPoint point;
        power_grid_operating_point_profile(&point,
            (PowerGridProfile)(trial % POWER_GRID_NUM_PROFILES));
        /* Exercise changing right-hand sides without violating the model's
         * nonnegative input contract. */
        double demand_scale = 0.82 + 0.003 * (trial % 61);
        double generation_scale = 0.88 + 0.004 * (trial % 47);
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++)
            point.load_mw[load] *= demand_scale;
        for (int generator = 1; generator < POWER_GRID_NUM_GENERATORS; generator++)
            point.generator_mw[generator] *= generation_scale;
        double rating_scale = 0.85 + 0.005 * (trial % 81);

        PowerGridSolveStatus status = compare_with_dense_reference(
            &candidate, &point, rating_scale);
        if (status == POWER_GRID_SOLVE_OK) {
            topology = candidate;
            valid_transitions++;
        } else {
            invalid_transitions++;
        }
    }
    CHECK(valid_transitions > 0);
    CHECK(invalid_transitions > 0);
}

static void test_invalid_input_parity(void)
{
    PowerGridTopology topology;
    PowerGridOperatingPoint point;
    power_grid_topology_normal(&topology);
    power_grid_operating_point_nominal(&point);
    point.load_mw[0] = NAN;
    compare_with_dense_reference(&topology, &point, 1.0);
    power_grid_operating_point_nominal(&point);
    point.generator_mw[1] = -1.0;
    compare_with_dense_reference(&topology, &point, 1.0);
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
    test_randomized_dense_parity();
    test_invalid_input_parity();
    test_busbar_actions();
    if (failures)
        return 1;
    puts("IEEE-14 DC solver tests passed");
    return 0;
}
