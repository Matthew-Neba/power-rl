#ifndef POWER_GRID_POLICY_H
#define POWER_GRID_POLICY_H

#include <math.h>
#include <stdlib.h>

#include "../../src/puffernet.h"

#ifndef POWER_GRID_POLICY_HIDDEN_SIZE
#define POWER_GRID_POLICY_HIDDEN_SIZE 512
#endif
#ifndef POWER_GRID_POLICY_NUM_LAYERS
#define POWER_GRID_POLICY_NUM_LAYERS 3
#endif

typedef enum
{
    POWER_GRID_INFERENCE_STOCHASTIC,
    POWER_GRID_INFERENCE_ARGMAX,
} PowerGridInferenceMode;

/* PufferNet's generic demo sampler exponentiates raw logits. Trained policies
 * can produce logits large enough to overflow, so deployment uses the standard
 * max-subtracted categorical sampler used by PPO implementations. */
static inline void power_grid_sample_multidiscrete(
    Multidiscrete *layer, const float *input, float *output)
{
    int action_width = 0;
    for (int head = 0; head < layer->num_actions; head++)
        action_width += layer->logit_sizes[head];

    for (int batch = 0; batch < layer->batch_size; batch++)
    {
        int input_index = batch * (action_width + 1);
        for (int head = 0; head < layer->num_actions; head++)
        {
            int size = layer->logit_sizes[head];
            float maximum = input[input_index];
            for (int action = 1; action < size; action++)
                maximum = fmaxf(maximum, input[input_index + action]);
            double total = 0.0;
            for (int action = 0; action < size; action++)
                total += exp((double)input[input_index + action] - maximum);
            double draw = ((double)rand() / ((double)RAND_MAX + 1.0)) * total;
            double cumulative = 0.0;
            int selected = size - 1;
            for (int action = 0; action < size; action++)
            {
                cumulative += exp((double)input[input_index + action] - maximum);
                if (draw < cumulative)
                {
                    selected = action;
                    break;
                }
            }
            output[batch * layer->num_actions + head] = (float)selected;
            input_index += size;
        }
    }
}

static inline void power_grid_policy_action(
    PufferNet *policy, const float *observations, float *actions,
    PowerGridInferenceMode mode)
{
    linear(policy->encoder, (float *)observations);
    mingru(policy->mingru, policy->encoder->output);
    linear(policy->decoder, policy->mingru->output);
    if (mode == POWER_GRID_INFERENCE_ARGMAX)
        argmax_multidiscrete(policy->multidiscrete, policy->decoder->output, actions);
    else
        power_grid_sample_multidiscrete(
            policy->multidiscrete, policy->decoder->output, actions);
}

#endif
