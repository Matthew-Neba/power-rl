#define POWER_GRID_NO_RENDER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "power_grid_policy.h"
#pragma clang diagnostic pop
#include "power_grid_solver.c"
#include "power_grid.h"
#include "power_grid_user.h"

typedef struct
{
    float observations[POWER_GRID_OBS_SIZE];
    float hidden[POWER_GRID_POLICY_HIDDEN_SIZE];
    int32_t action;
    uint8_t acceptable[POWER_GRID_NUM_ACTIONS];
    uint8_t sequence_start;
    uint8_t train;
} RecoveryExample;

_Static_assert(sizeof(RecoveryExample) ==
                   POWER_GRID_OBS_SIZE * sizeof(float) +
                       POWER_GRID_POLICY_HIDDEN_SIZE * sizeof(float) +
                       sizeof(int32_t) + POWER_GRID_NUM_ACTIONS + 2,
               "recovery dataset layout must match fit_recovery_decoder.py");

static void reset_policy(PufferNet *policy)
{
    memset(policy->mingru->state, 0,
           (size_t)policy->mingru->num_layers * policy->mingru->hidden_size *
               sizeof(float));
}

static int policy_action_and_hidden(PufferNet *policy, const float *observations,
                                    float *hidden)
{
    float action = 0.0f;
    linear(policy->encoder, (float *)observations);
    mingru(policy->mingru, policy->encoder->output);
    memcpy(hidden, policy->mingru->output,
           POWER_GRID_POLICY_HIDDEN_SIZE * sizeof(float));
    linear(policy->decoder, policy->mingru->output);
    argmax_multidiscrete(policy->multidiscrete, policy->decoder->output, &action);
    return (int)action;
}

static int expert_action(const PowerGrid *env, uint8_t *acceptable)
{
    int best_action = POWER_GRID_ACTION_NONE;
    double best_value = -INFINITY;
    memset(acceptable, 0, POWER_GRID_NUM_ACTIONS);
    for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++)
    {
        PowerGrid probe = *env;
        power_grid_apply_action(&probe.topology, action);
        for (int line = 0; line < POWER_GRID_NUM_BRANCHES; line++)
            if (!probe.line_available[line])
                probe.topology.line_closed[line] = 0;
        if (power_grid_solve_environment(&probe) != POWER_GRID_SOLVE_OK)
            continue;

        double served = power_grid_served_load_fraction(&probe);
        int secure = probe.solution.max_rho <= 1.0 &&
                     probe.ac_solution.voltage_violation_count == 0;
        if (secure && served >= 0.90)
            acceptable[action] = 1;
        double value = 1000000.0 * (secure && served >= 0.90) +
                       100000.0 * (served >= 0.90) + served -
                       1000.0 * fmax(probe.solution.max_rho - 1.0, 0.0) -
                       100.0 * probe.ac_solution.voltage_violation_count;
        if (value > best_value + 1e-12)
        {
            best_value = value;
            best_action = action;
        }
    }
    int acceptable_count = 0;
    for (int action = 0; action < POWER_GRID_NUM_ACTIONS; action++)
        acceptable_count += acceptable[action];
    if (acceptable_count == 0)
        acceptable[best_action] = 1;
    return best_action;
}

static void write_example(FILE *dataset, const float *observations,
                          const float *hidden, int action,
                          const uint8_t *acceptable, int sequence_start, int train,
                          size_t *count)
{
    RecoveryExample example = {0};
    memcpy(example.observations, observations, sizeof(example.observations));
    memcpy(example.hidden, hidden, sizeof(example.hidden));
    example.action = action;
    memcpy(example.acceptable, acceptable, sizeof(example.acceptable));
    example.sequence_start = sequence_start;
    example.train = train;
    if (fwrite(&example, sizeof(example), 1, dataset) != 1)
    {
        perror("write recovery dataset");
        exit(1);
    }
    (*count)++;
}

static void combination_at(int combination, int *first, int *second)
{
    if (combination < POWER_GRID_NUM_BRANCHES)
    {
        *first = combination;
        *second = -1;
        return;
    }
    int pair = combination - POWER_GRID_NUM_BRANCHES;
    for (*first = 0; *first < POWER_GRID_NUM_BRANCHES; (*first)++)
        for (*second = *first + 1; *second < POWER_GRID_NUM_BRANCHES; (*second)++)
            if (pair-- == 0)
                return;
    fprintf(stderr, "invalid combination %d\n", combination);
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        fprintf(stderr, "usage: %s CHECKPOINT DATASET [CONTEXTS]\n", argv[0]);
        return 2;
    }
    int contexts = argc == 4 ? atoi(argv[3]) : 16;
    if (contexts <= 0)
        return 2;

    Weights *weights = load_weights(argv[1]);
    int action_sizes[1] = {POWER_GRID_NUM_ACTIONS};
    PufferNet *policy = weights == NULL ? NULL : make_puffernet(
        weights, 1, POWER_GRID_OBS_SIZE, POWER_GRID_POLICY_HIDDEN_SIZE,
        POWER_GRID_POLICY_NUM_LAYERS, action_sizes, 1);
    FILE *dataset = fopen(argv[2], "wb");
    if (policy == NULL || dataset == NULL)
    {
        perror("open recovery dataset");
        return 1;
    }

    size_t examples = 0;
    int combinations = POWER_GRID_NUM_BRANCHES *
                       (POWER_GRID_NUM_BRANCHES + 1) / 2;
    for (int context = 0; context < contexts; context++)
        for (int combination = 0; combination < combinations; combination++)
        {
            int first, second;
            combination_at(combination, &first, &second);
            int case_id = first * POWER_GRID_NUM_BRANCHES +
                          (second >= 0 ? second : first);
            PowerGrid env = {
                .rng = UINT32_C(0x91e10da5) ^
                       (uint32_t)(context * 421 + case_id),
                .ac_power_flow = 1,
                .offline_scenarios = 1,
                .offline_scenario_probability = 1.0,
                .random_events = 0,
                .single_episode_evaluation = 1,
            };
            power_grid_allocate(&env);
            c_reset(&env);
            reset_policy(policy);
            PowerGridUserSession user;
            power_grid_user_init(&user, 2);
            int outage_step = POWER_GRID_STEPS_PER_PERIOD *
                              (1 + (context + case_id) % 9);

            while (env.log.n == 0.0f && env.episode_step <= outage_step + 4)
            {
                int final_outage = 0;
                if (env.episode_step == outage_step)
                {
                    int line = context % 2 == 0 || second < 0 ? first : second;
                    power_grid_user_set_line_outage(&user, &env, line, 1);
                    final_outage = second < 0;
                }
                if (second >= 0 && env.episode_step == outage_step + 1)
                {
                    int line = context % 2 == 0 ? second : first;
                    power_grid_user_set_line_outage(&user, &env, line, 1);
                    final_outage = 1;
                }

                float hidden[POWER_GRID_POLICY_HIDDEN_SIZE];
                int deployed_action = policy_action_and_hidden(
                    policy, env.observations, hidden);
                int target = POWER_GRID_ACTION_NONE;
                uint8_t acceptable[POWER_GRID_NUM_ACTIONS] = {0};
                acceptable[POWER_GRID_ACTION_NONE] = 1;
                if (env.episode_step >= outage_step)
                    target = expert_action(&env, acceptable);
                int train = env.episode_step == outage_step - 1 ||
                            env.episode_step >= outage_step;
                write_example(dataset, env.observations, hidden, target, acceptable,
                              env.episode_step == 0, train, &examples);

                /* Keep the first-click state on the deployed policy's data
                 * distribution. After the final click, follow the teacher so
                 * the dataset also contains legitimate multi-step recoveries. */
                int final_request_seen = second < 0 ?
                    env.episode_step >= outage_step :
                    env.episode_step >= outage_step + 1;
                env.actions[0] = final_outage || final_request_seen ?
                                     (float)target : (float)deployed_action;
                c_step(&env);
            }
            c_close(&env);
        }

    fclose(dataset);
    free_puffernet(policy);
    free(weights);
    printf("wrote %zu training examples from %d 2019 contexts to %s\n",
           examples, contexts, argv[2]);
    return 0;
}
