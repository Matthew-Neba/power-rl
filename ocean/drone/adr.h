#pragma once

#include <math.h>
#include <stdlib.h>

#include "dronelib.h"

// Automatic Domain Randomization (ADR): per-parameter randomization widths that grow
// only as fast as the policy stays competent at the boundary. One process-global
// frontier shared by every env instance, so all experience pools into it.

#define ADR_BUF 32      // perf samples per boundary before an expand/contract decision
#define ADR_W_MAX 0.9f  // cap so the (1 - w) factor stays positive (mass, inertia > 0)

typedef struct {
    float w;                // current half-width: factor ~ U(1-w, 1+w)
    float buf_lo[ADR_BUF];  // perf samples collected with this param pinned low
    float buf_hi[ADR_BUF];  // ... pinned high
    int n_lo, n_hi;
} AdrParam;

static struct {
    AdrParam p[NUM_DR_PARAMS];
    float w_floor;  // always-on baseline half-width; ADR may grow above it, never below
    float p_probe;  // probability an agent is a boundary probe this episode
    float t_lo;     // contract a boundary whose mean perf falls below this
    float t_hi;     // expand a boundary whose mean perf rises above this
    float step;     // width delta per update
    int on;         // 0 => static DR at the seed widths (no probing, no updates)
} g_adr;

// dict_set stores the key pointer (no copy), so these must outlive the log call.
static const char* ADR_LOG_KEYS[NUM_DR_PARAMS] = {
    "adr/arm_len", "adr/mass", "adr/ixx", "adr/iyy", "adr/izz",
    "adr/k_thrust", "adr/k_ang_damp", "adr/k_drag", "adr/b_drag", "adr/k_mot",
};

static inline void adr_init(float seed_w, float p_probe, float t_lo, float t_hi,
                            float step, int on) {
    // seed_w is the always-on floor: every param starts here and ADR may only grow above it.
    for (int i = 0; i < NUM_DR_PARAMS; i++) {
        g_adr.p[i].w = seed_w;
        g_adr.p[i].n_lo = 0;
        g_adr.p[i].n_hi = 0;
    }
    g_adr.w_floor = seed_w;
    g_adr.p_probe = p_probe;
    g_adr.t_lo = t_lo;
    g_adr.t_hi = t_hi;
    g_adr.step = step;
    g_adr.on = on;
}

// Draw multiplicative factors for one agent. With prob p_probe (and on), pin a single
// param to a boundary and report it via *probe_param / *probe_side (0=low, 1=high);
// otherwise *probe_param = -1. The pinned boundary is the worst-case ADR measures.
static inline void adr_sample(unsigned int* rng, float* factors,
                              int* probe_param, int* probe_side) {
    for (int i = 0; i < NUM_DR_PARAMS; i++) {
        float w = g_adr.p[i].w;
        factors[i] = rndf(1.0f - w, 1.0f + w, rng);
    }
    *probe_param = -1;
    *probe_side = -1;
    if (!g_adr.on || rndf(0.0f, 1.0f, rng) >= g_adr.p_probe) return;

    int j = (int)(rand_r(rng) % NUM_DR_PARAMS);
    int side = (rndf(0.0f, 1.0f, rng) < 0.5f) ? 0 : 1;
    float w = g_adr.p[j].w;
    factors[j] = (side == 0) ? (1.0f - w) : (1.0f + w);
    *probe_param = j;
    *probe_side = side;
}

// Record an episode's perf for a boundary probe; expand/contract once a buffer fills.
static inline void adr_record(int probe_param, int probe_side, float perf) {
    if (probe_param < 0) return;
    AdrParam* p = &g_adr.p[probe_param];
    float* buf = (probe_side == 0) ? p->buf_lo : p->buf_hi;
    int* n = (probe_side == 0) ? &p->n_lo : &p->n_hi;

    buf[*n] = perf;
    if (++(*n) < ADR_BUF) return;

    float mean = 0.0f;
    for (int i = 0; i < ADR_BUF; i++) mean += buf[i];
    mean /= (float)ADR_BUF;
    *n = 0;

    if (mean > g_adr.t_hi) p->w = fminf(ADR_W_MAX, p->w + g_adr.step);
    else if (mean < g_adr.t_lo) p->w = fmaxf(g_adr.w_floor, p->w - g_adr.step);
}
