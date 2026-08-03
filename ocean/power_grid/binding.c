#include "power_grid_solver.c"
#include "power_grid.h"

#define OBS_SIZE POWER_GRID_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {POWER_GRID_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env PowerGrid
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    env->owns_buffers = 0;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "total_failure", log->total_failure);
    dict_set(out, "topology_failure", log->topology_failure);
    dict_set(out, "solver_failure", log->solver_failure);
    dict_set(out, "total_switches", log->total_switches);
    dict_set(out, "line_switches", log->line_switches);
    dict_set(out, "busbar_switches", log->busbar_switches);
    dict_set(out, "coupler_switches", log->coupler_switches);
}
