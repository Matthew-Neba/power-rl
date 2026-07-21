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
    // ring PUTON asks "Which ring-finger, Right or Left?" ("rl" choices — 'y' is
    // invalid and aborts the puton). The action's own prompt: answer 'r', no penalty.
    int ringq = env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "ight or Left");
    int illegal = !praying && !ringq && (env->misc[NETHACK_MISC_YN] || env->misc[NETHACK_MISC_GETLIN]
               || nethack_msg_is_prompt(env));
    if (!illegal && !praying && !ringq) {
        if (env->misc[NETHACK_MISC_XWAIT]) nethack_drain_prompts(env);
        return 0;
    }
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]
            && !nethack_msg_is_prompt(env)) break;
        int ring = yn && nethack_msg_contains(env, "ight or Left");
        int commit = yn && !nethack_msg_contains(env, "no return")
                        && !nethack_msg_contains(env, "eally attack");
        env->obs.action = ring ? 'r' : (commit ? 'y' : 27);
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

// Parse a getobj bracket list ("[b-d f or ?*]") into cand[]; returns count.
static int nethack_parse_candidates(const Nethack* env, char* cand, int cap) {
    const unsigned char* m = env->message;
    int i = 0;
    while (i < NLE_MESSAGE_SIZE && m[i] && m[i] != '[') i++;
    int n = 0;
    for (i++; i < NLE_MESSAGE_SIZE && m[i] && n < cap; i++) {
        unsigned char c = m[i];
        if (n == 0 && (c == '-' || c == ' ' || c == '$')) continue;   // leading "- " (allownone) / "$" (gold)
        if (c == '-' && i + 1 < NLE_MESSAGE_SIZE) {       // compactified run
            for (char x = cand[n-1] + 1; x <= (char)m[i+1] && n < cap; x++)
                cand[n++] = x;
            i++;
            continue;
        }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) cand[n++] = (char)c;
        else break;   // ' ' before "or ?*", ']', '#', ...: end of the letter list
    }
    return n;
}

// Use a carried item through a getobj command (W = wear, e = eat): answer the
// prompt with the letter of inventory slot `slot` (the item action head).
// Floor offers (`floor_offer` = "eat it?") are ACCEPTED regardless of slot —
// fresh kills are the sustainable food source now that pray is gone; the
// cockatrice family is declined (instant petrification, matches "cockatrice"
// and "chickatrice"). Carried corpses stay invalid for EAT (age invisible).
// Returns 1 if an item was eaten/worn, 0 for a clean no-op (no prompt), -1
// when the chosen slot wasn't a valid candidate (prompt ESC'd; illegal).
static int nethack_do_use_item(Nethack* env, int cmd, const char* gate, const char* floor_offer, int slot) {
    env->obs.action = cmd;
    env->ctx = nle_step(env->ctx, &env->obs);
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN] && floor_offer && nethack_msg_contains(env, floor_offer)) {
            // quaff: decline fountain/sink offers so the potion prompt follows;
            // eat: accept floor food except the cockatrice family
            if (cmd == 'q' || nethack_msg_contains(env, "atrice")) env->obs.action = 'n';
            else {
                env->stats.floor_eats++;
                env->obs.action = 'y';
                env->ctx = nle_step(env->ctx, &env->obs);
                return 1;
            }
        }
        else if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, gate)) {
            char cand[52];
            int n = nethack_parse_candidates(env, cand, (int)sizeof(cand));
            char want = (char)env->inv_letters[slot];
            int ok = 0;
            for (int j = 0; j < n; j++)
                if (cand[j] == want) { ok = 1; break; }
            if (ok && cmd == 'e' && nethack_letter_is_corpse(env, want)) ok = 0;
            env->obs.action = ok ? want : 27;
            env->ctx = nle_step(env->ctx, &env->obs);
            return ok ? 1 : -1;
        }
        else return 0;
        env->ctx = nle_step(env->ctx, &env->obs);
    }
    return 0;
}
