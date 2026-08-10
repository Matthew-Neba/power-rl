#define _POSIX_C_SOURCE 200809L
#define POWER_GRID_NO_RENDER
#define POWER_GRID_DC_FLOAT

#include "power_grid_solver.c"
#include "power_grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Build and run from the repository root:
 * clang -O3 -march=native -Iocean/power_grid \
 *   tests/benchmark_power_grid_throughput.c -lm -o /tmp/power-grid-bench
 * /tmp/power-grid-bench mixed 128 4096 */
static double elapsed_seconds(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec) +
           1e-9 * (double)(end.tv_nsec - start.tv_nsec);
}

int main(int argc, char **argv)
{
    const char *workload = argc > 1 ? argv[1] : "mixed";
    int environment_count = argc > 2 ? atoi(argv[2]) : 128;
    int rounds = argc > 3 ? atoi(argv[3]) : 4096;
    int mixed_actions = strcmp(workload, "mixed") == 0;
    int random_actions = strcmp(workload, "random") == 0;
    if ((strcmp(workload, "noop") != 0 && !mixed_actions && !random_actions) ||
        environment_count <= 0 || rounds <= 0)
    {
        fprintf(stderr, "usage: %s [noop|mixed|random] [environments] [rounds]\n",
                argv[0]);
        return 2;
    }

    PowerGrid *environments = calloc((size_t)environment_count, sizeof(*environments));
    if (environments == NULL)
        return 2;
    for (int index = 0; index < environment_count; index++)
    {
        PowerGrid *env = &environments[index];
        env->rng = 1234u + (unsigned int)index;
        env->offline_scenarios = 1;
        env->offline_scenario_probability = 0.75;
        env->random_events = 1;
        env->random_event_probability = 0.50;
        env->random_outage_count = 3;
        power_grid_allocate(env);
        c_reset(env);
    }

    /* Warm caches and page mappings before timing. */
    for (int index = 0; index < environment_count; index++)
    {
        environments[index].actions[0] = POWER_GRID_ACTION_NONE;
        c_step(&environments[index]);
    }

    unsigned int random = 0x118738u;
    double checksum = 0.0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int round = 0; round < rounds; round++)
    {
        for (int index = 0; index < environment_count; index++)
        {
            int action = POWER_GRID_ACTION_NONE;
            if (mixed_actions || random_actions)
            {
                random = random * 1664525u + 1013904223u;
                if (random_actions || (random & 1u) != 0)
                    action = (int)(random % POWER_GRID_NUM_ACTIONS);
            }
            environments[index].actions[0] = (float)action;
            c_step(&environments[index]);
            checksum += environments[index].rewards[0] +
                        environments[index].observations[random % POWER_GRID_OBS_SIZE];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    long long steps = (long long)environment_count * rounds;
    double seconds = elapsed_seconds(start, end);
    printf("workload=%s environments=%d steps=%lld seconds=%.6f "
           "steps_per_second=%.2f checksum=%.9f\n",
           workload, environment_count, steps, seconds, steps / seconds, checksum);
    for (int index = 0; index < environment_count; index++)
        c_close(&environments[index]);
    free(environments);
    return 0;
}
