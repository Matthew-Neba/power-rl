#include "drone.h"
#include "puffernet.h"
#include "render.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

static void setup_task(DroneEnv* env, int task) {
    task_close(env);
    env->task = task;

    if (task == TASK_RACE) {
        RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
        cfg->max_rings = 10;
        cfg->horizon = 2048;
        env->task_config = cfg;
    } else {
        HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
        cfg->target_dist = 5.0f;
        cfg->sphere_radius = 4.0f;
        cfg->horizon = 1024;
        env->task_config = cfg;
    }
    task_init(env);
    c_reset(env);
}

// we render at 60Hz, but drone frames are 100Hz
static void step_realtime(DroneEnv* env, PufferNet* net) {
    static double accum = 0.0;
    accum += GetFrameTime();
    if (accum > 0.25) accum = 0.25;
    while (accum >= ACTION_DT) {
        forward_puffernet(net, env->observations, env->actions);
        c_step(env);
        accum -= ACTION_DT;
    }
}

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    PufferNet* net;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    if (IsKeyPressed(KEY_SPACE)) {
        setup_task(args->env, (args->env->task + 1) % NUM_TASKS);
    }
    step_realtime(args->env, args->net);
    c_render(args->env);
}
#endif

int main(int argc, char** argv) {
    srand(time(NULL));

    int task = argc > 1 ? atoi(argv[1]) : TASK_RACE;

    DroneEnv* env = calloc(1, sizeof(DroneEnv));
    env->num_agents = 64;

    env->observations = (float*)calloc(env->num_agents * DRONE_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    init(env);
    adr_init(0.05f, 0.0f, 0.0f, 0.0f, 0.0f, 0);  // static 5% DR for the demo (no probing)
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
        if (IsKeyPressed(KEY_SPACE)) setup_task(env, (env->task + 1) % NUM_TASKS);
        step_realtime(env, net);
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