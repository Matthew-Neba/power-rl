#ifndef POWER_GRID_MLP_POLICY_H
#define POWER_GRID_MLP_POLICY_H

#include <stdlib.h>

#ifndef POWER_GRID_MLP_NUM_LAYERS
#define POWER_GRID_MLP_NUM_LAYERS 1
#endif

#ifndef POWER_GRID_MLP_HIDDEN_SIZE
#define POWER_GRID_MLP_HIDDEN_SIZE 256
#endif

typedef struct
{
    float *encoder_weight;
    float *encoder_bias;
    float *hidden_weight;
    float *hidden_bias;
#if POWER_GRID_MLP_NUM_LAYERS == 2
    float *hidden_2_weight;
    float *hidden_2_bias;
#endif
    float *decoder_weight;
    float *decoder_bias;
    float encoded[POWER_GRID_MLP_HIDDEN_SIZE];
    float hidden[POWER_GRID_MLP_HIDDEN_SIZE];
    float hidden_2[POWER_GRID_MLP_HIDDEN_SIZE];
    float logits[POWER_GRID_NUM_ACTIONS];
} PowerGridMlpPolicy;

static inline PowerGridMlpPolicy *power_grid_make_mlp_policy(Weights *weights)
{
    PowerGridMlpPolicy *policy = calloc(1, sizeof(*policy));
    if (policy == NULL)
        return NULL;
    policy->encoder_weight = get_weights(
        weights, POWER_GRID_MLP_HIDDEN_SIZE * POWER_GRID_OBS_SIZE);
    policy->encoder_bias = get_weights(weights, POWER_GRID_MLP_HIDDEN_SIZE);
    policy->hidden_weight = get_weights(
        weights, POWER_GRID_MLP_HIDDEN_SIZE * POWER_GRID_MLP_HIDDEN_SIZE);
    policy->hidden_bias = get_weights(weights, POWER_GRID_MLP_HIDDEN_SIZE);
#if POWER_GRID_MLP_NUM_LAYERS == 2
    policy->hidden_2_weight = get_weights(
        weights, POWER_GRID_MLP_HIDDEN_SIZE * POWER_GRID_MLP_HIDDEN_SIZE);
    policy->hidden_2_bias = get_weights(weights, POWER_GRID_MLP_HIDDEN_SIZE);
#endif
    policy->decoder_weight = get_weights(
        weights, POWER_GRID_NUM_ACTIONS * POWER_GRID_MLP_HIDDEN_SIZE);
    policy->decoder_bias = get_weights(weights, POWER_GRID_NUM_ACTIONS);
    return policy;
}

static inline void power_grid_mlp_policy_action(
    PowerGridMlpPolicy *policy, const float *observations, float *actions,
    PowerGridInferenceMode mode)
{
    (void)mode;
    _linear((float *)observations, policy->encoder_weight,
            policy->encoder_bias, policy->encoded, 1,
            POWER_GRID_OBS_SIZE, POWER_GRID_MLP_HIDDEN_SIZE);
    _linear(policy->encoded, policy->hidden_weight, policy->hidden_bias,
            policy->hidden, 1, POWER_GRID_MLP_HIDDEN_SIZE,
            POWER_GRID_MLP_HIDDEN_SIZE);
    _gelu(policy->hidden, policy->hidden, POWER_GRID_MLP_HIDDEN_SIZE);
#if POWER_GRID_MLP_NUM_LAYERS == 2
    _linear(policy->hidden, policy->hidden_2_weight, policy->hidden_2_bias,
            policy->hidden_2, 1, POWER_GRID_MLP_HIDDEN_SIZE,
            POWER_GRID_MLP_HIDDEN_SIZE);
    _gelu(policy->hidden_2, policy->hidden_2, POWER_GRID_MLP_HIDDEN_SIZE);
    float *decoded_input = policy->hidden_2;
#else
    float *decoded_input = policy->hidden;
#endif
    _linear(decoded_input, policy->decoder_weight, policy->decoder_bias,
            policy->logits, 1, POWER_GRID_MLP_HIDDEN_SIZE,
            POWER_GRID_NUM_ACTIONS);
    int selected = 0;
    for (int action = 1; action < POWER_GRID_NUM_ACTIONS; action++)
        if (policy->logits[action] > policy->logits[selected])
            selected = action;
    actions[0] = (float)selected;
}

#endif
