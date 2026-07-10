// Prompt auto-dismissal and the multi-key action macros (Elbereth, wear,
// eat). Included by nethack.h after the Nethack struct.
#pragma once

// message ends with '?': single-key prompts NLE doesn't expose via misc[]
static int nethack_msg_is_prompt(const Nethack* env) {
    const unsigned char* m = env->message;
    if (!m[0]) return 0;
    int e = 0;
    while (e < NLE_MESSAGE_SIZE && m[e]) e++;
    while (e > 0 && m[e-1] == ' ') e--;
    return e > 0 && m[e-1] == '?';
}

static int nethack_msg_contains(const Nethack* env, const char* needle) {
    char buf[NLE_MESSAGE_SIZE + 1];
    memcpy(buf, env->message, NLE_MESSAGE_SIZE);
    buf[NLE_MESSAGE_SIZE] = '\0';
    return strstr(buf, needle) != NULL;
}

// dismiss passive prompts (welcome, --More--, getline) until the game is back
// at the main command prompt
static void nethack_drain_prompts(Nethack* env) {
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]) break;
        env->obs.action = yn ? 27 : '\r';
        env->ctx = nle_step(env->ctx, &env->obs);
    }
}

// Answer sub-prompts the agent can't: yn prompts commit 'y' EXCEPT the
// no-return climb (ends the game as ESCAPED) and peaceful-attack confirms
// (hostilizes Minetown); those and everything else get ESC. Returns 1 if a
// sub-prompt fired (the illegal_penalty condition).
static int nethack_handle_prompts(Nethack* env) {
    // direction prompts stay live — the agent's next key answers them
    if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "n what direction"))
        return 0;
    // the pray confirm is a deliberate action's own prompt: commit, no penalty
    int praying = env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "to pray");
    int illegal = !praying && (env->misc[NETHACK_MISC_YN] || env->misc[NETHACK_MISC_GETLIN]
               || nethack_msg_is_prompt(env));
    if (!illegal && !praying) {
        if (env->misc[NETHACK_MISC_XWAIT]) nethack_drain_prompts(env);
        return 0;
    }
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]
            && !nethack_msg_is_prompt(env)) break;
        int commit = yn && !nethack_msg_contains(env, "no return")
                        && !nethack_msg_contains(env, "eally attack");
        env->obs.action = commit ? 'y' : 27;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
    if (illegal) env->stats.illegal_actions++;
    return illegal;
}

// Engrave Elbereth with a fingertip (E, '-', "Elbereth", RET) — the early
// game's strongest panic button. Each stage is gated on its expected prompt;
// aborts fall through to nethack_handle_prompts.
static void nethack_do_elbereth(Nethack* env) {
    env->obs.action = 'E';
    env->ctx = nle_step(env->ctx, &env->obs);
    if (env->obs.done || !env->misc[NETHACK_MISC_YN]
        || !nethack_msg_contains(env, "write with")) return;
    env->obs.action = '-';
    env->ctx = nle_step(env->ctx, &env->obs);
    // the dust --More-- raises xwait ALONGSIDE the getlin — clear it before
    // typing; decline "add to current engraving?" so the fresh text replaces it
    const char* c = "Elbereth\r";
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done && *c; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN]
                 && nethack_msg_contains(env, "current engraving")) env->obs.action = 'n';
        else if (env->misc[NETHACK_MISC_GETLIN]) env->obs.action = (unsigned char)*c++;
        else break;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
}

// corpses are excluded from the eat macro: age is invisible and old ones kill
static int nethack_letter_is_corpse(const Nethack* env, char letter) {
    for (int i = 0; i < NLE_INVENTORY_SIZE && env->inv_letters[i]; i++)
        if ((char)env->inv_letters[i] == letter) {
            int g = env->inv_glyphs[i];
            return g >= NETHACK_GLYPH_BODY_OFF
                && g < NETHACK_GLYPH_BODY_OFF + NETHACK_NUMMONS;
        }
    return 0;
}

// Random letter from a getobj bracket list ("[b-d f or ?*]"), 0 if none.
// Random, not first: a refused item stays listed and would be retried forever.
static int nethack_pick_candidate(Nethack* env) {
    const unsigned char* m = env->message;
    int i = 0;
    while (i < NLE_MESSAGE_SIZE && m[i] && m[i] != '[') i++;
    char cand[52];
    int n = 0;
    for (i++; i < NLE_MESSAGE_SIZE && m[i] && n < (int)sizeof(cand); i++) {
        unsigned char c = m[i];
        if (n == 0 && (c == '-' || c == ' ')) continue;   // allownone's leading "- "
        if (c == '-' && i + 1 < NLE_MESSAGE_SIZE) {       // compactified run
            for (char x = cand[n-1] + 1; x <= (char)m[i+1] && n < (int)sizeof(cand); x++)
                cand[n++] = x;
            i++;
            continue;
        }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) cand[n++] = (char)c;
        else break;   // ' ' before "or ?*", ']', '#', ...: end of the letter list
    }
    int k = 0;
    for (int j = 0; j < n; j++)
        if (!nethack_letter_is_corpse(env, cand[j])) cand[k++] = cand[j];
    n = k;
    if (n == 0) return 0;
    env->rng = env->rng * 1103515245u + 12345u;
    return cand[(env->rng >> 16) % (unsigned)n];
}

// Use a carried item through a getobj command (W = wear, e = eat): answer the
// prompt with a random listed letter. `decline` skips floor-item offers
// ("eat it?" — floor corpses kill). Nothing usable -> no prompt -> clean
// no-op. Returns 1 if an item letter was sent.
static int nethack_do_use_item(Nethack* env, int cmd, const char* gate, const char* decline) {
    env->obs.action = cmd;
    env->ctx = nle_step(env->ctx, &env->obs);
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN] && decline && nethack_msg_contains(env, decline))
            env->obs.action = 'n';
        else if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, gate)) {
            int c = nethack_pick_candidate(env);
            // no acceptable candidate: dismiss penalty-free
            env->obs.action = c ? c : 27;
            env->ctx = nle_step(env->ctx, &env->obs);
            return c != 0;
        }
        else return 0;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
    return 0;
}
