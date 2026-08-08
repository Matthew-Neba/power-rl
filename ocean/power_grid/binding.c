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
    env->evaluation_scenarios = (int)dict_get(kwargs, "evaluation_scenarios")->value;
    env->offline_scenarios = (int)dict_get(kwargs, "offline_scenarios")->value;
    env->offline_scenario_validation =
        (int)dict_get(kwargs, "offline_scenario_validation")->value;
    env->offline_scenario_probability =
        dict_get(kwargs, "offline_scenario_probability")->value;
    if (!isfinite(env->offline_scenario_probability))
        env->offline_scenario_probability = 0.75;
    if (env->offline_scenario_probability < 0.0)
        env->offline_scenario_probability = 0.0;
    if (env->offline_scenario_probability > 1.0)
        env->offline_scenario_probability = 1.0;
    env->random_events = (int)dict_get(kwargs, "random_events")->value;
    env->random_event_probability = dict_get(kwargs, "random_event_probability")->value;
    if (!isfinite(env->random_event_probability))
        env->random_event_probability = 0.25;
    if (env->random_event_probability < 0.0)
        env->random_event_probability = 0.0;
    if (env->random_event_probability > 1.0)
        env->random_event_probability = 1.0;
}

void my_log(Log* log, Dict* out) {
#define POWER_GRID_LOG(field) dict_set(out, #field, log->field)
    POWER_GRID_LOG(perf);
    POWER_GRID_LOG(score);
    POWER_GRID_LOG(episode_return);
    POWER_GRID_LOG(episode_length);
    POWER_GRID_LOG(total_failure);
    POWER_GRID_LOG(topology_failure);
    POWER_GRID_LOG(solver_failure);
    POWER_GRID_LOG(event_failure);
    POWER_GRID_LOG(total_switches);
    POWER_GRID_LOG(line_switches);
    POWER_GRID_LOG(busbar_switches);
    POWER_GRID_LOG(coupler_switches);
    POWER_GRID_LOG(overload_free_steps);
    POWER_GRID_LOG(ac_voltage_violation_steps);
    POWER_GRID_LOG(ac_generator_p_violation_steps);
    POWER_GRID_LOG(ac_q_limit_events);
    POWER_GRID_LOG(ac_mean_active_loss_mw);
    POWER_GRID_LOG(ac_nonconvergence);
    POWER_GRID_LOG(ac_thermal_trips);
    POWER_GRID_LOG(ac_peak_thermal_stress);
    POWER_GRID_LOG(maintenance_events);
    POWER_GRID_LOG(random_events);
#undef POWER_GRID_LOG
}
