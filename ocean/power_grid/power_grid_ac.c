#include "power_grid_ac.h"

#include <complex.h>
#include <math.h>
#include <string.h>

#define AC_STATE_SIZE (2 * POWER_GRID_NUM_NODES)
#define AC_TOLERANCE 1e-8

/* Electrical data comes from canonical MATPOWER case118. The source case has
 * unlimited ratings, so the generated data supplies documented voltage-class
 * ratings for both the DC training proxy and AC validation. */
#define AC_BRANCH_R POWER_GRID_BRANCH_R
#define AC_BRANCH_B POWER_GRID_BRANCH_B
#define AC_LOAD_Q_NOMINAL POWER_GRID_LOAD_Q_NOMINAL
#define AC_LOAD_P_NOMINAL POWER_GRID_LOAD_P_NOMINAL
#define AC_GEN_V_SETPOINT POWER_GRID_GENERATOR_V_SETPOINT
#define AC_GEN_Q_MIN POWER_GRID_GENERATOR_Q_MIN
#define AC_GEN_Q_MAX POWER_GRID_GENERATOR_Q_MAX
#define AC_GEN_P_MAX POWER_GRID_GENERATOR_P_MAX

typedef struct {
    unsigned char active[POWER_GRID_NUM_NODES];
    unsigned char pv[POWER_GRID_NUM_NODES];
    double g[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    double b[POWER_GRID_NUM_NODES][POWER_GRID_NUM_NODES];
    double p_spec[POWER_GRID_NUM_NODES];
    double q_spec[POWER_GRID_NUM_NODES];
    double load_q[POWER_GRID_NUM_NODES];
    double voltage[POWER_GRID_NUM_NODES];
    double angle[POWER_GRID_NUM_NODES];
    int generator_node[POWER_GRID_NUM_GENERATORS];
    int slack;
} ACNetwork;

double power_grid_ac_load_q_mvar(const PowerGridOperatingPoint* point, int load) {
    return AC_LOAD_Q_NOMINAL[load] * point->load_mw[load] / AC_LOAD_P_NOMINAL[load];
}

double power_grid_ac_branch_rating_mva(int branch) {
    return POWER_GRID_BRANCHES[branch].thermal_limit_mw;
}

double power_grid_ac_thermal_step(double previous_stress, double rho) {
    if (rho <= 1.0) return fmax(0.0, previous_stress - POWER_GRID_THERMAL_COOLING_PER_STEP);
    double overload = rho - 1.0;
    return previous_stress + overload * overload;
}

static void ac_build_network(const PowerGridTopology* topology, ACNetwork* ac) {
    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!topology->line_closed[line]) continue;
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
            branch->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
            branch->to_bus);
        ac->active[from] = ac->active[to] = 1;
        double r = AC_BRANCH_R[line], x = branch->reactance;
        double denominator = r * r + x * x;
        double series_g = r / denominator;
        double series_b = -x / denominator;
        double tap = branch->tap_ratio;
        ac->g[from][from] += series_g / (tap * tap);
        ac->b[from][from] += (series_b + AC_BRANCH_B[line] / 2.0) / (tap * tap);
        ac->g[to][to] += series_g;
        ac->b[to][to] += series_b + AC_BRANCH_B[line] / 2.0;
        ac->g[from][to] -= series_g / tap;
        ac->b[from][to] -= series_b / tap;
        ac->g[to][from] -= series_g / tap;
        ac->b[to][from] -= series_b / tap;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        ac->active[node] = 1;
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        ac->active[node] = 1;
    }
    for (int bus = 0; bus < POWER_GRID_NUM_SUBSTATIONS; bus++)
    {
        int node = 2 * bus;
        ac->b[node][node] += POWER_GRID_BUS_SHUNT_B_MVAR[bus] / POWER_GRID_BASE_MVA;
    }
}

static void ac_calculate_power(const ACNetwork* ac, double* p, double* q) {
    memset(p, 0, POWER_GRID_NUM_NODES * sizeof(*p));
    memset(q, 0, POWER_GRID_NUM_NODES * sizeof(*q));
    for (int i = 0; i < POWER_GRID_NUM_NODES; i++) {
        if (!ac->active[i]) continue;
        for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
            if (!ac->active[j]) continue;
            double difference = ac->angle[i] - ac->angle[j];
            double cosine = cos(difference), sine = sin(difference);
            p[i] += ac->voltage[i] * ac->voltage[j] *
                (ac->g[i][j] * cosine + ac->b[i][j] * sine);
            q[i] += ac->voltage[i] * ac->voltage[j] *
                (ac->g[i][j] * sine - ac->b[i][j] * cosine);
        }
    }
}

static PowerGridACStatus ac_newton_raphson(ACNetwork* ac, int* total_iterations) {
    int angle_index[POWER_GRID_NUM_NODES], voltage_index[POWER_GRID_NUM_NODES];
    int dimensions = 0;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        angle_index[node] = voltage_index[node] = -1;
        if (ac->active[node] && node != ac->slack) angle_index[node] = dimensions++;
    }
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        if (ac->active[node] && node != ac->slack && !ac->pv[node]) {
            voltage_index[node] = dimensions++;
        }
    }

    for (int iteration = 0; iteration < POWER_GRID_AC_MAX_ITERATIONS; iteration++) {
        double p[POWER_GRID_NUM_NODES], q[POWER_GRID_NUM_NODES];
        double jacobian[AC_STATE_SIZE][AC_STATE_SIZE] = {{0}};
        double mismatch[AC_STATE_SIZE] = {0}, update[AC_STATE_SIZE] = {0};
        ac_calculate_power(ac, p, q);
        double largest = 0.0;
        for (int i = 0; i < POWER_GRID_NUM_NODES; i++) {
            if (angle_index[i] >= 0) {
                mismatch[angle_index[i]] = ac->p_spec[i] - p[i];
                if (fabs(mismatch[angle_index[i]]) > largest) largest = fabs(mismatch[angle_index[i]]);
            }
            if (voltage_index[i] >= 0) {
                mismatch[voltage_index[i]] = ac->q_spec[i] - q[i];
                if (fabs(mismatch[voltage_index[i]]) > largest) largest = fabs(mismatch[voltage_index[i]]);
            }
        }
        (*total_iterations)++;
        if (largest < AC_TOLERANCE) return POWER_GRID_AC_OK;

        for (int i = 0; i < POWER_GRID_NUM_NODES; i++) {
            if (!ac->active[i] || i == ac->slack) continue;
            int pi = angle_index[i];
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (!ac->active[j] || j == ac->slack) continue;
                int tj = angle_index[j];
                if (i == j) {
                    jacobian[pi][tj] = -q[i] - ac->b[i][i] *
                        ac->voltage[i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[pi][tj] = ac->voltage[i] * ac->voltage[j] *
                        (ac->g[i][j] * sin(d) - ac->b[i][j] * cos(d));
                }
            }
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (voltage_index[j] < 0) continue;
                int vj = voltage_index[j];
                if (i == j) {
                    jacobian[pi][vj] = p[i] / ac->voltage[i] +
                        ac->g[i][i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[pi][vj] = ac->voltage[i] *
                        (ac->g[i][j] * cos(d) + ac->b[i][j] * sin(d));
                }
            }
            if (voltage_index[i] < 0) continue;
            int qi = voltage_index[i];
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (!ac->active[j] || j == ac->slack) continue;
                int tj = angle_index[j];
                if (i == j) {
                    jacobian[qi][tj] = p[i] - ac->g[i][i] *
                        ac->voltage[i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[qi][tj] = -ac->voltage[i] * ac->voltage[j] *
                        (ac->g[i][j] * cos(d) + ac->b[i][j] * sin(d));
                }
            }
            for (int j = 0; j < POWER_GRID_NUM_NODES; j++) {
                if (voltage_index[j] < 0) continue;
                int vj = voltage_index[j];
                if (i == j) {
                    jacobian[qi][vj] = q[i] / ac->voltage[i] -
                        ac->b[i][i] * ac->voltage[i];
                } else {
                    double d = ac->angle[i] - ac->angle[j];
                    jacobian[qi][vj] = ac->voltage[i] *
                        (ac->g[i][j] * sin(d) - ac->b[i][j] * cos(d));
                }
            }
        }
        if (!power_grid_solve_dense(&jacobian[0][0], mismatch, update, dimensions,
                AC_STATE_SIZE)) {
            return POWER_GRID_AC_SINGULAR;
        }
        double scale = 1.0;
        for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
            if (angle_index[node] >= 0 && fabs(update[angle_index[node]]) > 0.5) scale = 0.5;
            if (voltage_index[node] >= 0 && fabs(update[voltage_index[node]]) > 0.2) scale = 0.5;
        }
        for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
            if (angle_index[node] >= 0) ac->angle[node] += scale * update[angle_index[node]];
            if (voltage_index[node] >= 0) {
                ac->voltage[node] += scale * update[voltage_index[node]];
                if (!isfinite(ac->voltage[node]) || ac->voltage[node] < 0.2 ||
                        ac->voltage[node] > 2.0) {
                    return POWER_GRID_AC_DIVERGED;
                }
            }
        }
    }
    return POWER_GRID_AC_DIVERGED;
}

static void ac_branch_flow(int line, int from, int to, const double* voltage,
        const double* angle, double* from_p, double* from_q, double* to_p, double* to_q) {
    const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
    double tap = branch->tap_ratio;
    double complex series = 1.0 / (AC_BRANCH_R[line] + I * branch->reactance);
    double complex charging = I * AC_BRANCH_B[line] / 2.0;
    double complex from_voltage = voltage[from] * cexp(I * angle[from]);
    double complex to_voltage = voltage[to] * cexp(I * angle[to]);
    double complex from_current = (series + charging) * from_voltage / (tap * tap) -
        series * to_voltage / tap;
    double complex to_current = (series + charging) * to_voltage -
        series * from_voltage / tap;
    double complex from_power = POWER_GRID_BASE_MVA * from_voltage * conj(from_current);
    double complex to_power = POWER_GRID_BASE_MVA * to_voltage * conj(to_current);
    *from_p = creal(from_power);
    *from_q = cimag(from_power);
    *to_p = creal(to_power);
    *to_q = cimag(to_power);
}

PowerGridACStatus power_grid_ac_solve(const PowerGridTopology* topology,
        const PowerGridOperatingPoint* point, PowerGridACSolveResult* result) {
    memset(result, 0, sizeof(*result));
    result->min_voltage_pu = INFINITY;
    result->topology_status = power_grid_validate_topology(topology, &result->component_count,
        &result->active_node_count);
    if (result->topology_status != POWER_GRID_SOLVE_OK) {
        return result->status = POWER_GRID_AC_TOPOLOGY_FAILURE;
    }

    ACNetwork ac = {0};
    ac_build_network(topology, &ac);
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        ac.voltage[node] = ac.active[node] ? 1.0 : 0.0;
    }

    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        if (!isfinite(point->load_mw[load]) || point->load_mw[load] < 0.0) {
            return result->status = POWER_GRID_AC_INVALID_INPUT;
        }
        int node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
            POWER_GRID_LOAD_BUSES[load]);
        double reactive = power_grid_ac_load_q_mvar(point, load);
        ac.p_spec[node] -= point->load_mw[load] / POWER_GRID_BASE_MVA;
        ac.q_spec[node] -= reactive / POWER_GRID_BASE_MVA;
        ac.load_q[node] += reactive;
    }
    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        if (!isfinite(point->generator_mw[gen]) || point->generator_mw[gen] < 0.0) {
            return result->status = POWER_GRID_AC_INVALID_INPUT;
        }
        ac.generator_node[gen] = power_grid_terminal_node(topology,
            POWER_GRID_GENERATOR_TERMINAL(gen),
            POWER_GRID_GENERATOR_BUSES[gen]);
        ac.voltage[ac.generator_node[gen]] = AC_GEN_V_SETPOINT[gen];
        if (gen > 0) {
            ac.p_spec[ac.generator_node[gen]] +=
                point->generator_mw[gen] / POWER_GRID_BASE_MVA;
            ac.pv[ac.generator_node[gen]] = 1;
        }
    }
    ac.slack = ac.generator_node[0];

    /* DC angles provide a much better initial guess for stressed/reconfigured cases. */
    PowerGridSolveResult dc = {0};
    if (power_grid_solve(topology, point, &dc) == POWER_GRID_SOLVE_OK) {
        memcpy(ac.angle, dc.node_angle, sizeof(ac.angle));
    }

    for (;;) {
        PowerGridACStatus status = ac_newton_raphson(&ac, &result->iterations);
        if (status != POWER_GRID_AC_OK) {
            /* A second, independent flat start distinguishes a poor DC-derived
             * initial guess from a topology that genuinely will not converge. */
            for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
                ac.voltage[node] = ac.active[node] ? 1.0 : 0.0;
                ac.angle[node] = 0.0;
            }
            ac.voltage[ac.slack] = AC_GEN_V_SETPOINT[0];
            for (int gen = 1; gen < POWER_GRID_NUM_GENERATORS; gen++) {
                if (ac.pv[ac.generator_node[gen]]) {
                    ac.voltage[ac.generator_node[gen]] = AC_GEN_V_SETPOINT[gen];
                }
            }
            status = ac_newton_raphson(&ac, &result->iterations);
        }
        if (status != POWER_GRID_AC_OK) return result->status = status;
        double p[POWER_GRID_NUM_NODES], q[POWER_GRID_NUM_NODES];
        ac_calculate_power(&ac, p, q);
        int limited = -1;
        double limited_q = 0.0;
        for (int gen = 1; gen < POWER_GRID_NUM_GENERATORS; gen++) {
            int node = ac.generator_node[gen];
            if (!ac.pv[node]) continue;
            double generator_q = q[node] * POWER_GRID_BASE_MVA + ac.load_q[node];
            if (generator_q < AC_GEN_Q_MIN[gen] - 1e-7) {
                limited = gen; limited_q = AC_GEN_Q_MIN[gen]; break;
            }
            if (generator_q > AC_GEN_Q_MAX[gen] + 1e-7) {
                limited = gen; limited_q = AC_GEN_Q_MAX[gen]; break;
            }
        }
        if (limited < 0) break;
        int node = ac.generator_node[limited];
        ac.pv[node] = 0;
        ac.q_spec[node] += limited_q / POWER_GRID_BASE_MVA;
        result->q_limit_count++;
    }

    double p[POWER_GRID_NUM_NODES], q[POWER_GRID_NUM_NODES];
    ac_calculate_power(&ac, p, q);
    result->converged = 1;
    for (int node = 0; node < POWER_GRID_NUM_NODES; node++) {
        result->node_voltage_pu[node] = ac.voltage[node];
        result->node_angle_rad[node] = ac.angle[node];
        if (!ac.active[node]) continue;
        if (!isfinite(ac.voltage[node]) || !isfinite(ac.angle[node])) {
            return result->status = POWER_GRID_AC_NONFINITE;
        }
        if (ac.voltage[node] < result->min_voltage_pu) {
            result->min_voltage_pu = ac.voltage[node];
        }
        if (ac.voltage[node] > result->max_voltage_pu) {
            result->max_voltage_pu = ac.voltage[node];
        }
        double violation = fmax(POWER_GRID_AC_VOLTAGE_MIN - ac.voltage[node],
            ac.voltage[node] - POWER_GRID_AC_VOLTAGE_MAX);
        if (violation > 0.0) {
            result->voltage_violation_count++;
            result->voltage_violation_cost += 100.0 * violation * violation;
        }
    }

    for (int gen = 0; gen < POWER_GRID_NUM_GENERATORS; gen++) {
        int node = ac.generator_node[gen];
        result->generator_p_mw[gen] = gen == 0 ?
            p[node] * POWER_GRID_BASE_MVA : point->generator_mw[gen];
        /* Canonical IEEE-118 has at most one aggregated load and generator per bus. */
        double local_load_p = 0.0;
        for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
            int load_node = power_grid_terminal_node(topology, POWER_GRID_LOAD_TERMINAL(load),
                POWER_GRID_LOAD_BUSES[load]);
            if (load_node == node) local_load_p += point->load_mw[load];
        }
        if (gen == 0) result->generator_p_mw[gen] += local_load_p;
        result->generator_q_mvar[gen] = q[node] * POWER_GRID_BASE_MVA + ac.load_q[node];
        double p_violation = result->generator_p_mw[gen] - AC_GEN_P_MAX[gen];
        if (p_violation > 1e-7) {
            result->generator_p_violation_count++;
            result->generator_p_violation_mw += p_violation;
        }
        result->substation_injection_mw[POWER_GRID_GENERATOR_BUSES[gen]] +=
            result->generator_p_mw[gen];
    }
    for (int load = 0; load < POWER_GRID_NUM_LOADS; load++) {
        result->substation_injection_mw[POWER_GRID_LOAD_BUSES[load]] -= point->load_mw[load];
    }

    for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++) {
        if (!topology->line_closed[line]) continue;
        const PowerGridBranch* branch = &POWER_GRID_BRANCHES[line];
        int from = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 0),
            branch->from_bus);
        int to = power_grid_terminal_node(topology, POWER_GRID_LINE_TERMINAL(line, 1),
            branch->to_bus);
        ac_branch_flow(line, from, to, ac.voltage, ac.angle,
            &result->branch_from_p_mw[line],
            &result->branch_from_q_mvar[line], &result->branch_to_p_mw[line],
            &result->branch_to_q_mvar[line]);
        result->branch_from_mva[line] = hypot(result->branch_from_p_mw[line],
            result->branch_from_q_mvar[line]);
        result->branch_to_mva[line] = hypot(result->branch_to_p_mw[line],
            result->branch_to_q_mvar[line]);
        result->branch_rho[line] = fmax(result->branch_from_mva[line],
            result->branch_to_mva[line]) / power_grid_ac_branch_rating_mva(line);
        result->total_p_loss_mw += result->branch_from_p_mw[line] + result->branch_to_p_mw[line];
        result->total_q_loss_mvar += result->branch_from_q_mvar[line] + result->branch_to_q_mvar[line];
        if (!isfinite(result->branch_rho[line])) return result->status = POWER_GRID_AC_NONFINITE;
        if (result->branch_rho[line] > result->max_rho) result->max_rho = result->branch_rho[line];
        if (result->branch_rho[line] > 1.0) {
            double overload = result->branch_rho[line] - 1.0;
            result->congestion_cost += overload * overload;
        }
    }
    return result->status = POWER_GRID_AC_OK;
}

void power_grid_ac_to_compatible(const PowerGridACSolveResult* ac,
        PowerGridSolveResult* compatible) {
    memset(compatible, 0, sizeof(*compatible));
    compatible->component_count = ac->component_count;
    compatible->active_node_count = ac->active_node_count;
    if (ac->status == POWER_GRID_AC_TOPOLOGY_FAILURE) compatible->status = ac->topology_status;
    else if (ac->status == POWER_GRID_AC_OK) compatible->status = POWER_GRID_SOLVE_OK;
    else if (ac->status == POWER_GRID_AC_INVALID_INPUT) compatible->status = POWER_GRID_INVALID_INPUT;
    else if (ac->status == POWER_GRID_AC_NONFINITE) compatible->status = POWER_GRID_NONFINITE;
    else compatible->status = POWER_GRID_SINGULAR;
    compatible->slack_generation_mw = ac->generator_p_mw[0];
    compatible->congestion_cost = ac->congestion_cost;
    compatible->max_rho = ac->max_rho;
    memcpy(compatible->node_angle, ac->node_angle_rad, sizeof(compatible->node_angle));
    memcpy(compatible->branch_flow_mw, ac->branch_from_p_mw, sizeof(compatible->branch_flow_mw));
    memcpy(compatible->branch_rho, ac->branch_rho, sizeof(compatible->branch_rho));
    memcpy(compatible->substation_injection_mw, ac->substation_injection_mw,
        sizeof(compatible->substation_injection_mw));
}

const char* power_grid_ac_status_name(PowerGridACStatus status) {
    static const char* names[] = {
        "ok", "topology_failure", "invalid_input", "singular", "diverged", "nonfinite",
    };
    return names[status];
}
