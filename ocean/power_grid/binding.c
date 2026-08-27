/* Training uses the refined float DC factorization for throughput. Standalone
 * tools select float explicitly when they need to match this training path. */
#define POWER_GRID_DC_FLOAT
#include "power_grid_solver.c"
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
    env->max_episode_steps = (int)dict_get(kwargs, "max_episode_steps")->value;
    env->ac_power_flow = (int)dict_get(kwargs, "ac_power_flow")->value;
    env->configured_ac_power_flow = env->ac_power_flow;
    env->ac_power_flow_probability =
        dict_get(kwargs, "ac_power_flow_probability")->value;
    env->curriculum_ac_power_flow_probability =
        dict_get(kwargs, "curriculum_ac_power_flow_probability")->value;
    env->offline_scenarios = (int)dict_get(kwargs, "offline_scenarios")->value;
    env->offline_scenario_validation =
        (int)dict_get(kwargs, "offline_scenario_validation")->value;
    env->offline_scenario_probability = dict_get(kwargs, "offline_scenario_probability")->value;
    env->random_events = (int)dict_get(kwargs, "random_events")->value;
    env->random_event_probability = dict_get(kwargs, "random_event_probability")->value;
    env->random_outage_count = (int)dict_get(kwargs, "random_outage_count")->value;
    env->random_outage_count_min =
        (int)dict_get(kwargs, "random_outage_count_min")->value;
    env->random_outages_at_reset =
        (int)dict_get(kwargs, "random_outages_at_reset")->value;
    env->reset_outage_probability =
        dict_get(kwargs, "reset_outage_probability")->value;
    env->timed_outages_when_not_reset =
        (int)dict_get(kwargs, "timed_outages_when_not_reset")->value;
    env->sequential_outages =
        (int)dict_get(kwargs, "sequential_outages")->value;
    env->curriculum_steps =
        (int)dict_get(kwargs, "curriculum_steps")->value;
    env->curriculum_recovery_probability =
        dict_get(kwargs, "curriculum_recovery_probability")->value;
    env->curriculum_recovery_probability_final =
        dict_get(kwargs, "curriculum_recovery_probability_final")->value;
    env->curriculum_safe_probability =
        dict_get(kwargs, "curriculum_safe_probability")->value;
    env->curriculum_sequence_probability =
        dict_get(kwargs, "curriculum_sequence_probability")->value;
    env->curriculum_sequence_max_period =
        (int)dict_get(kwargs, "curriculum_sequence_max_period")->value;
    env->curriculum_trial_steps =
        (int)dict_get(kwargs, "curriculum_trial_steps")->value;
    env->curriculum_one_step_fraction =
        dict_get(kwargs, "curriculum_one_step_fraction")->value;
    env->curriculum_outage_line =
        (int)dict_get(kwargs, "curriculum_outage_line")->value;
    env->randomize_reset_operating_period =
        (int)dict_get(kwargs, "randomize_reset_operating_period")->value;
    env->initial_outage_requires_overload =
        (int)dict_get(kwargs, "initial_outage_requires_overload")->value;
    env->initial_outage_requires_one_step_recovery =
        (int)dict_get(kwargs,
                      "initial_outage_requires_one_step_recovery")->value;
    env->end_episode_on_recovery =
        (int)dict_get(kwargs, "end_episode_on_recovery")->value;
    env->single_episode_evaluation =
        (int)dict_get(kwargs, "single_episode_evaluation")->value;
    env->reward_failure = (float)dict_get(kwargs, "reward_failure")->value;
    env->reward_thermal_trip =
        (float)dict_get(kwargs, "reward_thermal_trip")->value;
    env->reward_alive = (float)dict_get(kwargs, "reward_alive")->value;
    env->reward_warning_threshold =
        (float)dict_get(kwargs, "reward_warning_threshold")->value;
    env->reward_worst_line_cost_weight =
        (float)dict_get(kwargs, "reward_worst_line_cost_weight")->value;
    env->reward_congestion_cost_weight =
        (float)dict_get(kwargs, "reward_congestion_cost_weight")->value;
    env->reward_congestion_progress_weight =
        (float)dict_get(kwargs, "reward_congestion_progress_weight")->value;
    env->reward_switch_penalty =
        (float)dict_get(kwargs, "reward_switch_penalty")->value;
    if (env->offline_scenarios &&
        ((env->initial_outage_requires_one_step_recovery &&
          env->offline_scenario_probability >= 1.0) ||
         env->curriculum_steps > 0))
        power_grid_prepare_one_step_recovery_cache();
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
    POWER_GRID_LOG(curriculum_recovery_trial);
    POWER_GRID_LOG(curriculum_recovery_success);
#undef POWER_GRID_LOG
}
