/* Canonical MATPOWER case14 electrical data with the environment's documented
 * synthetic thermal ratings. The cached AESO renewable traces are normalized
 * from their source nameplates before they are applied to this smaller case. */
#ifndef POWER_GRID_IEEE14_DATA_H
#define POWER_GRID_IEEE14_DATA_H

#define POWER_GRID_SLACK_BUS 0
#define POWER_GRID_IEEE14_SLACK_BUS POWER_GRID_SLACK_BUS
#define POWER_GRID_IEEE14_BRIDGE_COUNT 1
#define POWER_GRID_SOLAR_GENERATOR_BUS 5
#define POWER_GRID_WIND_GENERATOR_BUS 2
#define POWER_GRID_SOLAR_NAMEPLATE_MW 30.0
#define POWER_GRID_WIND_NAMEPLATE_MW 45.0
#define POWER_GRID_SCENARIO_SOLAR_SOURCE_NAMEPLATE_MW 500.0
#define POWER_GRID_SCENARIO_WIND_SOURCE_NAMEPLATE_MW 700.0

static const PowerGridBranch POWER_GRID_BRANCHES[POWER_GRID_NUM_BRANCHES] = {
    {0, 1, .05917, 1.000, 160}, {0, 4, .22304, 1.000, 145},
    {1, 2, .19797, 1.000, 130}, {1, 3, .17632, 1.000, 80},
    {1, 4, .17388, 1.000, 70},  {2, 3, .17103, 1.000, 50},
    {3, 4, .04211, 1.000, 80},  {3, 6, .20912, .978, 50},
    {3, 8, .55618, .969, 35},   {4, 5, .25202, .932, 65},
    {5, 10, .19890, 1.000, 25}, {5, 11, .25581, 1.000, 25},
    {5, 12, .13027, 1.000, 30}, {6, 7, .17615, 1.000, 25},
    {6, 8, .11001, 1.000, 50},  {8, 9, .08450, 1.000, 20},
    {8, 13, .27038, 1.000, 15}, {9, 10, .19207, 1.000, 20},
    {11, 12, .19988, 1.000, 20},{12, 13, .34802, 1.000, 25},
};

static const char *const POWER_GRID_BRANCH_NAMES[POWER_GRID_NUM_BRANCHES] = {
    "1-2", "1-5", "2-3", "2-4", "2-5", "3-4", "4-5", "4-7", "4-9", "5-6",
    "6-11", "6-12", "6-13", "7-8", "7-9", "9-10", "9-14", "10-11", "12-13", "13-14",
};

static const double POWER_GRID_BRANCH_R[POWER_GRID_NUM_BRANCHES] = {
    .01938, .05403, .04699, .05811, .05695, .06701, .01335, 0, 0, 0,
    .09498, .12291, .06615, 0, 0, .03181, .12711, .08205, .22092, .17093,
};

static const double POWER_GRID_BRANCH_B[POWER_GRID_NUM_BRANCHES] = {
    .0528, .0492, .0438, .0340, .0346, .0128, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

/* Line 7-8 (index 13) is the sole bridge in the normal IEEE-14 graph. */
static const unsigned char POWER_GRID_RANDOM_EVENT_ELIGIBLE[POWER_GRID_NUM_BRANCHES] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
};

static const int POWER_GRID_GENERATOR_BUSES[POWER_GRID_NUM_GENERATORS] = {0, 1, 2, 5, 7};
static const double POWER_GRID_GENERATOR_P_NOMINAL[POWER_GRID_NUM_GENERATORS] = {
    232.4, 40.0, 0.0, 0.0, 0.0,
};
static const double POWER_GRID_GENERATOR_P_MAX[POWER_GRID_NUM_GENERATORS] = {
    332.4, 140.0, 100.0, 100.0, 100.0,
};
static const double POWER_GRID_GENERATOR_Q_MIN[POWER_GRID_NUM_GENERATORS] = {
    0.0, -40.0, 0.0, -6.0, -6.0,
};
static const double POWER_GRID_GENERATOR_Q_MAX[POWER_GRID_NUM_GENERATORS] = {
    10.0, 50.0, 40.0, 24.0, 24.0,
};
static const double POWER_GRID_GENERATOR_V_SETPOINT[POWER_GRID_NUM_GENERATORS] = {
    1.060, 1.045, 1.010, 1.070, 1.090,
};

static const int POWER_GRID_LOAD_BUSES[POWER_GRID_NUM_LOADS] = {
    1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13,
};
static const double POWER_GRID_LOAD_P_NOMINAL[POWER_GRID_NUM_LOADS] = {
    21.7, 94.2, 47.8, 7.6, 11.2, 29.5, 9.0, 3.5, 6.1, 13.5, 14.9,
};
static const double POWER_GRID_LOAD_Q_NOMINAL[POWER_GRID_NUM_LOADS] = {
    12.7, 19.0, -3.9, 1.6, 7.5, 16.6, 5.8, 1.8, 1.6, 5.8, 5.0,
};

static const double POWER_GRID_BUS_SHUNT_B_MVAR[POWER_GRID_NUM_SUBSTATIONS] = {
    0, 0, 0, 0, 0, 0, 0, 0, 19.0, 0, 0, 0, 0, 0,
};
static const double POWER_GRID_BUS_BASE_KV[POWER_GRID_NUM_SUBSTATIONS] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
static const double POWER_GRID_BUS_V_REFERENCE[POWER_GRID_NUM_SUBSTATIONS] = {
    1.060, 1.045, 1.010, 1.019, 1.020, 1.070, 1.062,
    1.090, 1.056, 1.051, 1.057, 1.055, 1.050, 1.036,
};
static const double POWER_GRID_BUS_ANGLE_REFERENCE_DEG[POWER_GRID_NUM_SUBSTATIONS] = {
    0.00, -4.98, -12.72, -10.33, -8.78, -14.22, -13.37,
    -13.36, -14.94, -15.10, -14.79, -15.07, -15.16, -16.04,
};

#endif
