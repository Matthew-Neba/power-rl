#include "drone.h"
#include "puffernet.h"
#include "render.h"
#include "task_hover.h"
#include "task_race.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

static void setup_task(DroneEnv* env, int task) {
    if (env->task != NULL) env->task->close(env);

    if (task == 1) {
        env->task = &TASK_RACE;
        RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
        cfg->max_rings = 10;
        env->task_config = cfg;
    } else {
        env->task = &TASK_HOVER;
        HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
        cfg->target_dist = 5.0f;
        env->task_config = cfg;
    }
    env->task->init(env);
    c_reset(env);
}

static void toggle_task(DroneEnv* env) {
    int current = (env->task == &TASK_RACE) ? 1 : 0;
    setup_task(env, (current + 1) % 2);
}

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    PufferNet* net;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    if (IsKeyPressed(KEY_SPACE)) toggle_task(args->env);
    forward_puffernet(args->net, args->env->observations, args->env->actions);
    c_step(args->env);
    c_render(args->env);
}
#endif

int main(int argc, char** argv) {
    srand(time(NULL));

    int task = argc > 1 ? atoi(argv[1]) : 0;

    DroneEnv* env = calloc(1, sizeof(DroneEnv));
    env->num_agents = 64;

    env->observations = (float*)calloc(env->num_agents * DRONE_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    init(env);
    setup_task(env, task);

    Weights* weights = load_weights("resources/drone/drone_weights.bin");
    int logit_sizes[4] = {1, 1, 1, 1};
    PufferNet* net = make_puffernet(weights, env->num_agents, DRONE_OBS_SIZE, 64, 2, logit_sizes, 4);

#ifdef __EMSCRIPTEN__
    WebRenderArgs args = {.env = env, .net = net};
    emscripten_set_main_loop_arg(emscriptenStep, &args, 0, true);
#else
    c_render(env);
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) toggle_task(env);
        forward_puffernet(net, env->observations, env->actions);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free_puffernet(net);
    free(weights);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
#endif

    return 0;
}