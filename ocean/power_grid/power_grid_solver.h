#ifndef POWER_GRID_SOLVER_H
#define POWER_GRID_SOLVER_H

#include <stddef.h>

#define POWER_GRID_NUM_SUBSTATIONS 14
#define POWER_GRID_NUM_BUSBARS 2
#define POWER_GRID_NUM_NODES (POWER_GRID_NUM_SUBSTATIONS * POWER_GRID_NUM_BUSBARS)
#define POWER_GRID_NUM_BRANCHES 20
#define POWER_GRID_NUM_GENERATORS 5
#define POWER_GRID_NUM_LOADS 11
#define POWER_GRID_NUM_TERMINALS 56
#define POWER_GRID_BASE_MVA 100.0

#define POWER_GRID_LINE_ACTION_OFFSET 1
#define POWER_GRID_TERMINAL_ACTION_OFFSET \
    (POWER_GRID_LINE_ACTION_OFFSET + POWER_GRID_NUM_BRANCHES)
#define POWER_GRID_COUPLER_ACTION_OFFSET \
    (POWER_GRID_TERMINAL_ACTION_OFFSET + POWER_GRID_NUM_TERMINALS)
#define POWER_GRID_NUM_ACTIONS \
    (POWER_GRID_COUPLER_ACTION_OFFSET + POWER_GRID_NUM_SUBSTATIONS)

#define POWER_GRID_LINE_TERMINAL(branch, side) (2 * (branch) + (side))
#define POWER_GRID_GENERATOR_TERMINAL(generator) (40 + (generator))
#define POWER_GRID_LOAD_TERMINAL(load) (45 + (load))

typedef struct {
    int from_bus;
    int to_bus;
    double reactance;
    double tap_ratio;
    double thermal_limit_mw;
} PowerGridBranch;

typedef struct {
    unsigned char line_closed[POWER_GRID_NUM_BRANCHES];
    unsigned char terminal_busbar[POWER_GRID_NUM_TERMINALS];
    unsigned char coupler_closed[POWER_GRID_NUM_SUBSTATIONS];
} PowerGridTopology;

typedef struct {
    double load_mw[POWER_GRID_NUM_LOADS];
    double generator_mw[POWER_GRID_NUM_GENERATORS];
} PowerGridOperatingPoint;

typedef enum {
    POWER_GRID_SOLVE_OK = 0,
    POWER_GRID_INVALID_TOPOLOGY,
    POWER_GRID_DISCONNECTED_LOAD,
    POWER_GRID_DISCONNECTED_GENERATOR,
    POWER_GRID_ISLANDED,
    POWER_GRID_SINGULAR,
    POWER_GRID_NONFINITE,
    POWER_GRID_INVALID_INPUT,
} PowerGridSolveStatus;

typedef struct {
    PowerGridSolveStatus status;
    int component_count;
    int active_node_count;
    double slack_generation_mw;
    double node_angle[POWER_GRID_NUM_NODES];
    double branch_flow_mw[POWER_GRID_NUM_BRANCHES];
    double branch_rho[POWER_GRID_NUM_BRANCHES];
    double substation_injection_mw[POWER_GRID_NUM_SUBSTATIONS];
    double congestion_cost;
    double max_rho;
} PowerGridSolveResult;

typedef enum {
    POWER_GRID_PROFILE_P0_NOMINAL = 0,
    POWER_GRID_PROFILE_P1_HIGH,
    POWER_GRID_PROFILE_P2_EAST_SHIFT,
    POWER_GRID_PROFILE_P3_RESTORATION_SHIFT,
    POWER_GRID_PROFILE_P4_BUS6_LOAD_SHIFT,
    POWER_GRID_PROFILE_P5_BUS9_LOAD_SHIFT,
    POWER_GRID_PROFILE_P6_BUS10_LOAD_SHIFT,
    POWER_GRID_PROFILE_P7_BUS6_GENERATION,
    POWER_GRID_PROFILE_P8_BUS13_LOAD_SHIFT,
    POWER_GRID_PROFILE_P9_BUS3_GENERATION,
    POWER_GRID_PROFILE_P10_BUS4_LOAD_GENERATION,
    POWER_GRID_PROFILE_P11_BUS5_LOAD_GENERATION,
    POWER_GRID_PROFILE_P12_BUS11_LOAD_GENERATION,
    POWER_GRID_PROFILE_P13_BUS12_LOAD_GENERATION,
    POWER_GRID_PROFILE_P14_BUS2_LOCAL_PEAK,
    POWER_GRID_NUM_PROFILES,
} PowerGridProfile;

typedef enum {
    POWER_GRID_ACTION_INVALID = -1,
    POWER_GRID_ACTION_NONE = 0,
    POWER_GRID_ACTION_LINE,
    POWER_GRID_ACTION_TERMINAL,
    POWER_GRID_ACTION_COUPLER,
} PowerGridActionType;

extern const PowerGridBranch POWER_GRID_BRANCHES[POWER_GRID_NUM_BRANCHES];
extern const int POWER_GRID_GENERATOR_BUSES[POWER_GRID_NUM_GENERATORS];
extern const int POWER_GRID_LOAD_BUSES[POWER_GRID_NUM_LOADS];
extern const char* const POWER_GRID_BRANCH_NAMES[POWER_GRID_NUM_BRANCHES];

void power_grid_topology_normal(PowerGridTopology* topology);
PowerGridActionType power_grid_apply_action(PowerGridTopology* topology, int action);
int power_grid_terminal_node(const PowerGridTopology* topology, int terminal, int substation);

void power_grid_operating_point_nominal(PowerGridOperatingPoint* point);
int power_grid_operating_point_profile(PowerGridOperatingPoint* point, PowerGridProfile profile);
const char* power_grid_profile_name(PowerGridProfile profile);

PowerGridSolveStatus power_grid_validate_topology(
    const PowerGridTopology* topology,
    int* component_count,
    int* active_node_count
);
PowerGridSolveStatus power_grid_solve(
    const PowerGridTopology* topology,
    const PowerGridOperatingPoint* point,
    PowerGridSolveResult* result
);
const char* power_grid_solve_status_name(PowerGridSolveStatus status);
const char* power_grid_action_name(int action, char* buffer, size_t size);

/* Shared pivoted dense solve used by the small DC and AC nodal systems. */
int power_grid_solve_dense(double* matrix, double* rhs, double* solution,
    int dimensions, int stride);

#endif
