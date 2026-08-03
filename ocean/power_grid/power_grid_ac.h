#ifndef POWER_GRID_AC_H
#define POWER_GRID_AC_H

#include "power_grid_solver.h"

#define POWER_GRID_AC_MAX_ITERATIONS 50
#define POWER_GRID_AC_VOLTAGE_MIN 0.90
#define POWER_GRID_AC_VOLTAGE_MAX 1.10
#define POWER_GRID_THERMAL_TRIP_THRESHOLD 1.0
#define POWER_GRID_THERMAL_COOLING_PER_STEP 0.02

typedef enum {
    POWER_GRID_AC_OK = 0,
    POWER_GRID_AC_TOPOLOGY_FAILURE,
    POWER_GRID_AC_INVALID_INPUT,
    POWER_GRID_AC_SINGULAR,
    POWER_GRID_AC_DIVERGED,
    POWER_GRID_AC_NONFINITE,
} PowerGridACStatus;

typedef struct {
    PowerGridACStatus status;
    PowerGridSolveStatus topology_status;
    int converged;
    int iterations;
    int component_count;
    int active_node_count;
    int q_limit_count;
    int voltage_violation_count;
    int generator_p_violation_count;

    double node_voltage_pu[POWER_GRID_NUM_NODES];
    double node_angle_rad[POWER_GRID_NUM_NODES];
    double branch_from_p_mw[POWER_GRID_NUM_BRANCHES];
    double branch_from_q_mvar[POWER_GRID_NUM_BRANCHES];
    double branch_to_p_mw[POWER_GRID_NUM_BRANCHES];
    double branch_to_q_mvar[POWER_GRID_NUM_BRANCHES];
    double branch_from_mva[POWER_GRID_NUM_BRANCHES];
    double branch_to_mva[POWER_GRID_NUM_BRANCHES];
    double branch_rho[POWER_GRID_NUM_BRANCHES];
    double generator_p_mw[POWER_GRID_NUM_GENERATORS];
    double generator_q_mvar[POWER_GRID_NUM_GENERATORS];
    double substation_injection_mw[POWER_GRID_NUM_SUBSTATIONS];

    double congestion_cost;
    double max_rho;
    double min_voltage_pu;
    double max_voltage_pu;
    double voltage_violation_cost;
    double generator_p_violation_mw;
    double total_p_loss_mw;
    double total_q_loss_mvar;
} PowerGridACSolveResult;

PowerGridACStatus power_grid_ac_solve(
    const PowerGridTopology* topology,
    const PowerGridOperatingPoint* point,
    PowerGridACSolveResult* result
);
const char* power_grid_ac_status_name(PowerGridACStatus status);
double power_grid_ac_load_q_mvar(const PowerGridOperatingPoint* point, int load);
double power_grid_ac_branch_rating_mva(int branch);
double power_grid_ac_thermal_step(double previous_stress, double rho);

/* Copy policy-compatible active-power fields while retaining AC details separately. */
void power_grid_ac_to_compatible(
    const PowerGridACSolveResult* ac,
    PowerGridSolveResult* compatible
);

#endif
