#include "power_grid_solver.c"
#include "power_grid_ac.c"
#include "power_grid.h"

#define OBS_SIZE POWER_GRID_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {POWER_GRID_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env PowerGrid
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->owns_buffers = 0;
    env->ac_power_flow = (int)dict_get(kwargs, "ac_power_flow")->value;
    env->offline_scenarios = (int)dict_get(kwargs, "offline_scenarios")->value;
    env->offline_scenario_validation =
        (int)dict_get(kwargs, "offline_scenario_validation")->value;
    env->offline_scenario_probability = dict_get(kwargs, "offline_scenario_probability")->value;
    env->random_events = (int)dict_get(kwargs, "random_events")->value;
    env->random_event_probability = dict_get(kwargs, "random_event_probability")->value;
    env->single_episode_evaluation =
        (int)dict_get(kwargs, "single_episode_evaluation")->value;
}

void my_log(Log* log, Dict* out) {
#define POWER_GRID_LOG(field) dict_set(out, #field, log->field)
    POWER_GRID_LOG(perf);
    POWER_GRID_LOG(score);
    POWER_GRID_LOG(episode_return);
    POWER_GRID_LOG(episode_length);
    POWER_GRID_LOG(total_failure);
    POWER_GRID_LOG(connectivity_failure);
    POWER_GRID_LOG(solver_failure);
    POWER_GRID_LOG(event_failure);
    POWER_GRID_LOG(total_switches);
    POWER_GRID_LOG(line_switches);
    POWER_GRID_LOG(busbar_switches);
    POWER_GRID_LOG(coupler_switches);
    POWER_GRID_LOG(overload_free_steps);
    POWER_GRID_LOG(random_events);
    POWER_GRID_LOG(demand_fulfilled);
    POWER_GRID_LOG(outage_completion);
    POWER_GRID_LOG(all_outages_survived);
    POWER_GRID_LOG(thermal_trips);
    POWER_GRID_LOG(thermal_trip_episode);
    POWER_GRID_LOG(peak_thermal_stress);
    POWER_GRID_LOG(peak_line_loading);
    POWER_GRID_LOG(overloaded_line_fraction);
#undef POWER_GRID_LOG
}
