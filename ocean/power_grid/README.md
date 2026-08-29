# Alberta-driven IEEE-14 topology control

This environment trains a topology-control policy with fast DC power flow and evaluates the same
policy under AC physics. The network is canonical MATPOWER IEEE-14. Load and generator injections
remain at one nominal dispatch. Alberta weather trajectories randomize ambient conditions and
dynamic line ratings, so this is a synthetic benchmark, not a reconstruction or operational model
of Alberta's transmission grid.

## Fixed environment contract

- 14 substations, 28 busbars, 20 branches, 5 generators, and 11 loads.
- Exactly 91 unmasked actions: one no-op, 20 line toggles, 56 terminal transfers, and 14 coupler
  toggles.
- Exactly 221 observations: 100 line features, 56 terminal bits, 14 coupler bits, 14 bus
  injections, one rating scale, three weather values, 28 busbar voltage/activity values, and five
  generator reactive-power values.
- No load shedding, generator tripping, generator redispatch action, action mask, rollback,
  planner, lookahead, forced no-op, or runtime fallback.

All equipment must remain connected to the energized slack component. A user or scheduled outage
that islands equipment remains applied and is counted as a failure. It is never silently rejected,
rolled back, or converted into reduced demand.

`power_grid_solver.c` uses cached sparse DC solves for training. The AC Newton-Raphson path adds
resistance, charging, transformer taps, the bus-9 shunt, voltage/reactive limits, losses, worst-end
MVA loading, and inverse-time thermal protection.

## Data, normalization, and outages

Load and generator injections are static across all twelve periods and all episodes. The checked-in
cache supplies 2019 weather for training and 2020 weather for held-out validation: temperature,
wind speed, and irradiance vary the physical line-rating multiplier without changing load or
generation. Domain randomization also remains active through outage set/timing sampling and the
targeted AC/DC curriculum mixture. All 20 lines are eligible for persistent N-1/N-2 outages;
combinations are not filtered from final training or QA.

Observations are normalized in the environment with physical bases, rather than fitted statistics
that could leak validation data:

| Feature | Encoding |
|---|---|
| branch flow and loading | signed per-unit flow and `rho`, where 1 is the rating |
| branch/terminal/coupler state | binary 0/1 |
| bus injection and generator Q | divided by the IEEE 100 MVA base |
| dynamic rating | multiplier, normally 0.90–1.35 |
| temperature, wind, irradiance | divided by 50 C, 10 m/s, and 1000 W/m² |
| AC voltage | per unit; DC activity is binary |

Per-unit overload values are not clipped because their severity is decision-relevant. The policy
does not compute running observation statistics at deployment, so training and WebAssembly
inference use the identical transformation.

## Reward and metrics

The reward contains overload cost and a switch-action cost:

```text
-reward_congestion_cost_weight * sum(max(0, line_loading - 1)^2)
-reward_switch_penalty * valid_non_noop_switch_action
```

Invalid topology, an unpowered island, a disconnected load or generator, or solver failure terminates
with a fixed reward of `-1`. A survivable AC thermal-protection trip updates the topology and then uses
the same overload reward. Every reward is clamped to `[-1, 1]`.

There is no separate reward for unserved load, frequency, reference-topology distance, or losses:
this version has fixed injections and no load-shedding/frequency control, and infeasible demand service
already appears as a terminal connectivity failure. Both reward weights are exposed in
`config/power_grid.ini` and copied into each native environment by `binding.c`.
This exactly matches the unmodified PufferLib PPO trainer's reward range.
Reward-independent metrics include survival, secure-step fraction, outage completion, switching,
thermal trips, loading, and demand served. `perf` is the primary 0–1 score and is not rescaled when
reward coefficients change.

## Build, train, and evaluate

```sh
uv run --with pytest pytest -q \
  tests/test_power_grid_solver.py tests/test_power_grid_scenarios.py
PATH="$PWD/.venv/bin:$PATH" ./build.sh power_grid
PATH="$PWD/.venv/bin:$PATH" puffer train power_grid
```

`config/power_grid.ini` is the single source of truth for the from-scratch training recipe. Plain
`puffer train power_grid` uses the compiled native MinGRU backend; no launcher script, checkpoint
loading, fine-tuning, teacher, planner, action mask, rollback, fallback, or PufferLib modification is
involved. The defaults run 80M steps and cosine-anneal entropy from `0.15` to a `0.075` floor.
Every setting remains overrideable through the normal Puffer CLI, for example
`--env.curriculum-ac-power-flow-probability 0.03`.

Training samples unfiltered N-1/N-2 contingencies and consecutive-click timing throughout the run. No
historical checkpoint is loaded and there are no later fine-tuning or offline-teacher stages.
The optional curriculum also uses the same 91 unrestricted actions: it mixes overloaded N-1
starts, intact one-step starts, and short consecutive-click N-2 trials, but never supplies a target
action. Its reset probability may anneal to a nonzero rehearsal floor while ordinary unfiltered
N-1/N-2 episodes remain in the mixture. The intact contrast teaches the policy to wait before a
contingency; N-1 rehearsal and unfiltered N-2 sequences teach post-contingency response without
erasing either context. Contextual trials may continue for several actions so the policy must hold
a recovered topology instead of immediately undoing its switch.
The short outage-response trials retain targeted AC/DC domain randomization, exposing the policy to
AC voltage, reactive-power, loss, and loading consequences without varying load or generator
injections.

Training defaults intentionally enable the curriculum. Disable it when sampling ordinary AC
evaluation episodes for a native checkpoint:

```sh
PATH="$PWD/.venv/bin:$PATH" puffer eval power_grid \
  --load-model-path checkpoints/power_grid/<run>/<checkpoint>.bin \
  --env.curriculum-steps 0 \
  --env.ac-power-flow True \
  --env.ac-power-flow-probability 0 \
  --env.random-events True \
  --env.random-event-probability 1 \
  --env.random-outage-count 2 \
  --env.random-outage-count-min 1
```

`--load-model-path` only loads weights; it does not select AC physics or force outages. The config's
policy dimensions must match the native checkpoint. This command samples automatic N-1/N-2
episodes, rather than reproducing manual browser clicks exactly.

The exhaustive browser-contract QA enumerates all 20 N-1 and 190 N-2 line sets, alternates both
N-2 click orders, and rotates held-out context and outage time. For N-2, both user clicks are
applied in their selected order before the policy receives its first post-outage observation and
response:

```sh
cc -O3 -march=native -w \
  -DPOWER_GRID_POLICY_NUM_LAYERS=1 \
  -Iocean/power_grid -Isrc -Ivendor ocean/power_grid/qa_user_outages.c \
  -lm -o build/qa_user_outages
./build/qa_user_outages checkpoints/power_grid/<run>/<checkpoint>.bin 8 0 1 0
```

A handled trial completes without solver/connectivity failure, keeps at least 95% of post-click
steps thermally and voltage secure, and serves at least 90% of demand on average. Impossible
islanding combinations remain in the denominator. The 2020 validation pool must not be used for
training. Inspect multiple from-scratch checkpoints because the final PPO checkpoint is not
necessarily the best held-out policy.

Under this fixed contract, 95% overall handling is mathematically unreachable. Line 7-8 (index
13) is the sole bridge to bus 8, so its persistent outage disconnects the bus-8 generator and no
allowed line, terminal, or coupler switch can reconnect it. The exhaustive set contains that line
in one of 20 N-1 cases and 19 of 190 N-2 cases. With impossible cases retained, the absolute
topological ceiling is therefore `(210 - 20) / 210 = 90.476%` before AC voltage, thermal, or policy
errors. Reaching 95% would require changing the grid, excluding impossible cases, permitting
generation disconnection, or adding a non-switching remedy; all are outside the fixed rules. The
optimization target is consequently the highest honest score below this ceiling unless the metric
is revised explicitly.

The repository policy is currently the exact 221/91 historical baseline from commit `fcef678b`,
SHA-256 `48b48f0194a4b3a2e19b5cf28332cbf0431e1edb80abad6d9f35e7b43d5a82f8`. It is a temporary
compatibility baseline, not the result of the RL-only curriculum and not a production-ready model.
A two-context held-out AC smoke (420 trials) measured 60.0% survival and 29.5% handled overall.
The strongest allowed from-scratch candidate so far is not promoted into `resources`: across two
standardized 16-context AC blocks (6,720 trials) it handled 26.85%, survived 56.01%, and served
62.52% of demand. It is preserved under
`checkpoints/power_grid/rl-only-candidate-anneal50-20260826/`. These figures are far below the 95%
handled and 90% demand gates.

Controlled 80-million-step, random-initialization follow-ups kept the same environment and action
contract. V-trace clips of 1.0 reached 26.50% handled at about 0.48M SPS; gamma/GAE 0.995/0.9
reached 26.58%; value coefficient 0.05 and default Adam momentum failed checkpoint screening;
max-gradient norm 1.5 reached 26.06%; and two additional seeds of the selected recipe reached
26.56% and 26.31%. None replaced the selected candidate. Competitive runs of the selected recipe
sustained roughly 0.52-0.56M SPS after startup.

Failure profiling of the selected candidate found that about 39% of unsafe first responses were
no-op and that non-noop responses used only a small subset of the 90 switches. A 50/50 mixture of
one-step-filtered and unrestricted overloaded N-1 curriculum starts broadened the lessons but
reached only 26.74% handled, 54.69% survival, and 61.51% demand at roughly 0.51M SPS. Doubling
targeted AC rehearsal from 1% to 2% cost about 6-10% throughput and failed checkpoint screening.
Learning rate 0.0015 retained roughly 0.54-0.57M late-run SPS but also failed screening. All three
were rejected; the launcher remains at fully one-step N-1 rehearsal, 1% targeted AC, and learning
rate 0.001.

A reward-only penalty for choosing no-op in a solved but overloaded state was also tested at 0.05
and 0.20. The larger value reduced unsafe first-response no-op on the screen, but its exhaustive
checkpoint reached only 26.44% handled, 54.54% survival, and 61.71% demand because the additional
switching caused worse downstream outcomes. The experiment and its configuration surface were
fully removed rather than retaining unused reward complexity.

The selected candidate's held-out breakdown is strongly N-2 limited: on one standardized
16-context block it handled 58.75% of N-1 trials but only 23.16% of N-2 trials. Increasing N-2
sequence curriculum probability from 30% to 40% while preserving the 35% intact-state contrast
reduced both categories and was rejected. Checkpoint selection now uses the 420-trial smoke only
as a first pass, followed by a matched 1,680-trial, eight-context screen before the full gate; the
small screen repeatedly overestimated candidates that failed broader contexts.

One-layer MLP widths of 384 and 192 were also tested. Width 384 fell to roughly 0.46-0.48M SPS and
did not match handled performance. Width 192 reached roughly 0.56M SPS, but on the matched
1,680-trial screen it scored 26.85% handled, 53.10% survival, and 61.30% demand versus the selected
256-wide model's 27.02%, 55.00%, and 62.13%. Width 256 remains the measured speed/quality choice.

PPO batch geometry was tested without changing rollout data. Increasing minibatch size from 18,432
to 36,864 halved optimizer-step count, slowed training to roughly 0.46-0.48M SPS, and remained far
behind the selected learning curve, so the run was stopped early. Increasing prioritized replay
alpha from 0.194 to 0.5 retained high SPS but scored only 26.85% handled, 53.57% survival, and
61.70% demand on the matched 1,680-trial screen. The selected four-update minibatch geometry and
weak prioritization remain unchanged.

Uniform trajectory sampling (`prio_alpha=0`) was then tested as the other endpoint. Its best
checkpoint reached 1,798/6,720 handled (26.76%) and 3,754 survived, versus the selected model's
1,804 handled and 3,764 survived. Mean demand improved slightly to 62.64% and thermal trips fell,
but it lost both primary handled and survival counts, so the selected alpha of about 0.194 remains.

Reducing the entropy schedule from 0.15→0.075 to 0.10→0.05 preserved high throughput (roughly
0.55M SPS late in the run), but the best smoke checkpoint regressed on the matched 1,680-trial
screen: 26.25% handled, 53.87% survival, 62.25% demand, and 25.65% thermal trips versus the
selected model's 27.02%, 55.00%, 62.13%, and 20.48%. It was rejected without a full gate.
Increasing the schedule to 0.20→0.10 was also rejected: its best checkpoint reached 26.96%
handled, 53.21% survival, 61.35% demand, and 24.05% thermal trips on the same screen, while
sustaining only about 0.51M SPS. The selected 0.15→0.075 schedule is therefore bracketed by
controlled lower- and higher-entropy runs.

Lengthening short recovery-curriculum trials from four to six steps retained roughly 0.54-0.56M
late-run SPS and narrowly improved the matched-screen handled rate, so it received the full gate.
Across 6,720 trials it handled 1,796 (26.73%), survived 3,666, served 61.53% demand, and incurred
1,345 thermal-trip trials versus the selected model's 1,804 handled, 3,764 survived, 62.52%
demand, and 1,312 thermal-trip trials. Four-step curriculum trials remain selected.

Reducing the short-curriculum reset share from 99.5% to 95% increased exact unfiltered N-1/N-2
episode exposure while retaining roughly 0.55-0.57M late-run SPS. Its best checkpoint nevertheless
scored 26.67% handled, 53.10% survival, 61.14% demand, and 22.26% thermal trips on the matched
1,680-trial screen versus the selected model's 27.02%, 55.00%, 62.13%, and 20.48%. It was rejected
without a full gate.

Reducing the recovery bonus from 1.0 to 0.5 avoided saturating successful recovery rewards at the
trainer's +1 clip and retained roughly 0.53-0.56M SPS. A later checkpoint showed fewer thermal
trips on the smoke set, but on the matched screen it reached only 26.85% handled, 54.11% survival,
61.94% demand, and 20.18% thermal trips. The earlier checkpoint with the best smoke handled rate
also regressed broadly, so the 1.0 recovery bonus remains selected.

Increasing cached historical-day sampling from 65% to 80% produced a promising smoke thermal
profile at no material throughput cost, but regressed on the matched screen to 26.49% handled,
53.87% survival, 61.83% demand, and 20.71% thermal trips. It was rejected without a full gate.
Reducing historical sampling to 50% slightly improved handled count on the matched screen, so it
received the full gate. Across 6,720 trials it handled 1,786 (26.58%), survived 3,658, served about
62.00% demand, and incurred 1,419 thermal-trip trials versus the selected model's 1,804 handled,
3,764 survived, 62.52% demand, and 1,312 thermal-trip trials. The selected 65% historical / 35%
synthetic operating-context mixture is therefore bracketed on both sides.

Doubling PPO rollout horizon from 72 to 144 fit in about 1.1GB VRAM and retained roughly 0.55M SPS.
Its balanced checkpoint edged the matched-screen handled count by one case, so it received the full
gate. Across 6,720 trials it handled 1,785 (26.56%), survived 3,702, served about 62.24% demand,
and incurred 1,482 thermal-trip trials. Horizon 72 remains selected.
Halving horizon to 36 sustained only about 0.51-0.53M SPS during full PPO updates. Its matched
screen edged handled count by two cases but regressed every secondary metric, and the full gate
confirmed 1,780/6,720 handled (26.49%), 3,659 survived, about 61.89% demand, and 1,517 thermal-trip
trials. Horizon 72 is therefore bracketed on both sides for quality and throughput.

Short from-scratch systems probes bracketed the selected eight vector threads: both four and
sixteen threads reached only about 0.47M SPS around 12M steps. Neither justified a full quality
run, and eight threads remains the measured throughput setting.

Doubling parallel agents from 1,024 to 2,048 materially improved sustained training throughput to
roughly 0.62-0.63M SPS with about 1.1GB VRAM. Two competitive checkpoints received full gates.
The balanced checkpoint handled 1,783/6,720 (26.53%), survived 3,787, served about 63.22% demand,
and incurred 1,377 thermal-trip trials. The final checkpoint handled 1,778 (26.46%), survived
3,791, served about 63.23% demand, and incurred 1,368 thermal-trip trials. Both improved survival,
demand, and speed but missed the selected model's 1,804 handled primary count, so 1,024 agents
remains in the reproducible selected launcher.
An intermediate 1,536-agent run sustained roughly 0.60-0.61M SPS. Its two competitive checkpoints
also received full gates: the earlier handled 1,783, survived 3,745, and served about 62.97% demand;
the final handled 1,784, survived 3,724, and served about 62.69% demand. Neither preserved the
selected handled count, so the launcher remains at 1,024 agents pending stronger evidence.
A paired-seed 2,048-agent repeat reproduced roughly 0.63M SPS but failed its matched screen at
26.90% handled, 53.21% survival, 61.45% demand, and 22.86% thermal trips. These larger-agent runs
also revealed that fixed per-slot curriculum steps confound agent-count comparisons by halving
curriculum progress per slot; follow-ups must scale that schedule with agent count.
A corrected paired run combined 2,048 agents, horizon 36, and 32,768 per-slot curriculum steps,
preserving the selected 73,728-sample rollout, four minibatches, seed, and global curriculum
progress. It sustained roughly 0.62M SPS but scored only 26.79% handled, 54.05% survival, 61.99%
demand, and 23.27% thermal trips on the matched screen, so it was rejected without a full gate.

Reducing targeted curriculum AC rehearsal from 1% to 0.5% retained roughly 0.56M late-run SPS.
Its balanced checkpoint improved matched-screen handled count by two cases and reduced thermal
trips, so it received the full gate. Across 6,720 trials it handled 1,790 (26.64%), survived 3,701,
served about 61.85% demand, and incurred 1,250 thermal-trip trials. The thermal reduction did not
offset the handled, survival, and demand regressions, so 1% targeted AC remains selected.
A paired-seed 0.75% midpoint retained about 0.54-0.55M SPS during full PPO updates, but its balanced
checkpoint reached only 26.79% handled, 54.17% survival, 61.73% demand, and 20.95% thermal trips on
the matched 1,680-trial screen. The selected 1% checkpoint reached 27.02%, 55.00%, 62.13%, and
20.48% on the same screen, so the midpoint was rejected without an exhaustive gate.

Doubling the safe-state maintenance reward from 0.05 to 0.10 retained about 0.53-0.55M SPS and
improved the matched-screen handled count by four cases, so it received the exhaustive gate.
Across 6,720 trials it handled 1,798 (26.76%), survived 3,730, served about 62.07% demand, and
incurred 1,270 thermal-trip trials. The 42 fewer thermal-trip trials did not offset six fewer
handled trials, 34 fewer survivals, and lower demand than the selected 0.05-reward checkpoint.
Halving the same reward to 0.025 also retained about 0.53-0.55M SPS. Its matched-screen handled
count improved by two cases, so it received the exhaustive gate. It tied the selected checkpoint
at 1,804/6,720 handled (26.85%), but survived only 3,708 cases and served about 62.04% demand;
thermal-trip trials were effectively tied at 1,310 versus 1,312. The selected 0.05 reward therefore
remains the better balanced setting and is bracketed on both sides.

Doubling the penalty for ineffective switching or switching from an already secure state from
0.25 to 0.50 retained roughly 0.54-0.57M SPS, but collapsed policy entropy and nearly eliminated
coupler use. Its best 420-trial smoke checkpoint reached only 25.95% handled versus 26.43% for the
selected checkpoint. Later checkpoints reduced thermal trips to 17.86% but handled only 24.05%,
showing that the stronger penalty suppressed necessary control diversity along with churn. It was
rejected without a larger gate.
Halving the penalty to 0.125 caused the opposite failure mode: switch counts and coupler churn rose,
late-run throughput fell to roughly 0.50-0.52M SPS, and the best smoke checkpoint still handled
only 25.95%. It was also rejected without a larger gate, leaving 0.25 bracketed as the measured
speed/quality setting.

Reducing intact-state curriculum contrast from 35% to 25% reallocated ten percentage points to
one-step N-1 recovery while preserving 30% N-2 sequence rehearsal. The extra recovery-reset work
caused a mid-training throughput trough near 0.42M SPS before recovering to about 0.55M late in the
run. Its best smoke checkpoint handled only 26.19%, with lower survival and demand than the selected
checkpoint, so it was rejected without a larger gate. The selected 35/30/35 intact/sequence/recovery
mixture remains unchanged.

An audit of older, unpromoted N-2-focused artifacts found one checkpoint that improved the matched
screen handled count by four cases. Its exhaustive gate nevertheless handled only 1,801/6,720,
survived 3,720, served about 62.72% demand, and incurred 1,485 thermal-trip trials. It was rejected;
the artifact's transient curriculum implementation is also no longer reproducible and therefore
was never eligible to replace a controlled current-recipe run.

Increasing intact-state contrast from 35% to 45% instead reduced one-step N-1 recovery to 25%
while retaining 30% N-2 sequence rehearsal. It sustained roughly 0.54-0.56M late-run SPS but its
best smoke checkpoint handled only 25.48%, with worse survival and demand. It was rejected without
a larger gate, leaving the selected 35/30/35 mixture bracketed on both sides.

A PPO-artifact audit found that increasing value-function clipping from 0.01 to 0.02 improved
survival and demand on small screens. Its exhaustive checkpoint handled only 1,785/6,720, however,
while surviving 3,770, serving about 63.13% demand, and incurring 1,309 thermal-trip trials. The
secondary gains did not offset 19 fewer handled cases. A clean paired-seed 0.015 midpoint sustained
roughly 0.53-0.56M SPS but regressed on the matched screen to 26.55% handled, 53.27% survival, and
61.52% demand. Both were rejected, leaving the selected 0.01 value clip bracketed.
The same artifact screen found no competitive signal from policy clip 0.20, value coefficient
0.05, or replay ratio 1.4.

Follow-up screening of the remaining PPO midpoints found that prioritized-replay alpha 0.1 never
exceeded 26.19% handled on the smoke set. Raising the entropy-floor ratio from 0.5 to 0.75 only tied
smoke handling while slightly regressing survival and demand. The best policy-clip 0.20 checkpoint
tied the selected model at 454/1,680 handled on the matched screen but lost 17 survivals and demand,
with no thermal improvement. None advanced to an exhaustive gate.
A clean paired-seed policy-clip 0.50 midpoint sustained roughly 0.53-0.54M SPS but its best smoke
checkpoint handled only 26.19%. Later checkpoints improved survival and thermal trips while handling
fell to 25.00-25.24%, so the midpoint was rejected without a larger gate. The selected loose clip of
about 0.9168 remains the measured setting.

An observation-encoding experiment represented only the binary topology categories (line closed,
line available, terminal busbar, and coupler closed) as symmetric -1/+1 values instead of 0/1. It
preserved all 221 observations, all 91 unrestricted switching actions, and the same physics and
reward recipe. Its clean paired-seed run sustained roughly 0.52-0.54M SPS during mature PPO
updates. The 44.31M-step
checkpoint edged the selected model by two handled cases on the matched screen, so it received the
full gate. Across 6,720 held-out AC trials it handled 1,779 (26.47%), survived 3,697 (55.01%),
served about 61.52% demand, and incurred 1,337 thermal-trip trials (19.90%). This regressed from the
selected model's 1,804 handled, 3,764 survived, 62.52% demand, and 1,312 thermal-trip trials, so
symmetric topology encoding was rejected. Its experimental configuration surface was removed.

Lowering the PPO learning rate from 0.001 to 0.00075 converged more slowly but recovered to roughly
0.52-0.54M SPS during mature updates. Its final two checkpoints only tied the selected smoke handled
rate while losing survival and demand. The better-balanced final checkpoint then reached 26.85%
handled, 52.02% survival, 60.41% demand, and 21.85% thermal trips on the matched 1,680-trial screen,
versus 27.02%, 55.00%, 62.13%, and 20.48% for the selected model. It was rejected without an
exhaustive gate. Together with the previously rejected 0.0015 run, this brackets the selected 0.001
learning rate.

Increasing consecutive N-2 sequence rehearsal from 30% to 40% while retaining 35% intact-state
contrast reduced one-step N-1 rehearsal from 35% to 25%. It sustained roughly 0.53-0.56M SPS late
in training and improved the matched-screen handled count by three cases, so it received the full
gate. Across 6,720 held-out AC trials it handled 1,799 (26.77%), survived 3,668 (54.58%), served
about 61.98% demand, and incurred 1,271 thermal-trip trials (18.91%). Although it reduced thermal
trips by 41, it missed the selected model by five handled cases, 96 survivals, and about 0.54 demand
percentage points. The selected 35/30/35 intact/sequence/recovery mixture remains unchanged.

A second normalization experiment addressed the 28 shared node features: inactive nodes mapped to
-1, active DC nodes and nominal AC voltage to zero, and the physical 0.90/1.10 AC voltage limits to
-1/+1 without clipping violations. It preserved the 221/91 contract and sustained roughly
0.53-0.55M SPS late in its clean paired-seed run. Its best smoke checkpoint missed the selected
handled count by one case and advanced to the matched screen, where it reached 26.96% handled,
53.75% survival, 61.87% demand, and 26.31% thermal trips. The selected representation reached
27.02%, 55.00%, 62.13%, and 20.48% on the same cases. The normalized representation was rejected
and its experimental configuration surface was removed rather than retained as unused complexity.

Reward diagnostics on the selected checkpoint found that 27.93% of solved post-click steps were
thermally unsafe, while only 0.048% were voltage-unsafe without also being thermally unsafe. A
reward-only peak accumulated thermal-stress cost was therefore tested without changing protection,
outages, observations, or actions. Weight 0.05 retained high SPS but collapsed survival and demand
on the smoke screen without reducing thermal trips. A conservative 0.01 midpoint sustained roughly
0.54-0.56M SPS, improved the matched handled count by three, and reduced thermal trips, so it
received the full gate. Across 6,720 held-out AC trials it handled 1,793 (26.68%), survived 3,690
(54.91%), served about 61.65% demand, and incurred 1,239 thermal-trip trials (18.44%). The 73 fewer
thermal-trip trials did not offset 11 fewer handled cases, 74 fewer survivals, and lower demand than
the selected model. Both weights were rejected and the experimental reward surface was removed.

Lowering only the cosine entropy-floor ratio from 0.50 to 0.25 retained the selected 0.15 initial
coefficient while allowing stronger late specialization. The run sustained roughly 0.54-0.55M SPS,
and its 59.06M-step checkpoint tied the selected handled count on the matched screen while reducing
thermal trips, so it received the full gate. Across 6,720 held-out AC trials it handled 1,786
(26.58%), survived 3,705 (55.13%), served about 61.68% demand, and incurred 1,166 thermal-trip
trials (17.35%). Although it prevented 146 thermal-trip trials, it lost 18 handled cases, 59
survivals, and demand relative to the selected checkpoint. Together with the previously rejected
0.75 floor, this brackets the selected 0.50 entropy-floor ratio.

The earlier `gamma=0.995` / `GAE lambda=0.9` rejection changed both credit-assignment parameters.
An isolated `GAE lambda=0.97` run therefore held the selected gamma and every other setting fixed.
It sustained roughly 0.54-0.55M SPS, and its 36.94M-step checkpoint was competitive on both smoke
and matched screens, so it received the full gate. Across 6,720 held-out AC trials it handled 1,781
(26.50%), survived 3,745 (55.73%), served about 62.86% demand, and incurred 1,242 thermal-trip
trials (18.48%). The demand and thermal improvements did not offset 23 fewer handled cases and 19
fewer survivals than the selected checkpoint, so the selected `0.986245...` GAE lambda remains.
Increasing only GAE lambda to 0.995 also sustained roughly 0.54-0.55M SPS, but no checkpoint
exceeded 26.19% handled on the 420-trial smoke set versus the selected model's 26.43%. The most
survivable checkpoint still had lower handling and no compensating thermal improvement, so it was
rejected without a larger gate. Isolated 0.97 and 0.995 runs now bracket the selected GAE lambda.

Discounting was then isolated from GAE: a `gamma=0.999` run held the selected GAE lambda and every
other setting fixed. It sustained roughly 0.54-0.55M SPS and improved handled count on smoke and
matched screens, so it received the full gate. Across 6,720 held-out AC trials it handled 1,799
(26.77%), survived 3,691 (54.93%), served about 61.96% demand, and incurred 1,385 thermal-trip
trials (20.61%). It lost five handled cases, 73 survivals, demand, and thermal safety versus the
selected checkpoint, so the selected `0.9998925...` gamma remains.
The higher endpoint `gamma=1.0` retained roughly 0.52-0.55M mature SPS and produced a competitive
66.43M-step checkpoint, so it also received the full gate. It handled 1,789/6,720 trials (26.62%),
survived 3,716 (55.30%), served about 62.22% demand, and incurred 1,274 thermal-trip trials
(18.96%). The 38 fewer thermal-trip trials did not offset 15 fewer handled cases, 48 fewer
survivals, and lower demand. Isolated 0.999 and 1.0 runs now bracket the selected gamma.

Relaxing only PPO's maximum gradient norm from the selected `5.0` to `10.0` sustained roughly
0.54-0.56M mature SPS. The predeclared smoke-stage winner at 29.56M steps handled 456/1,680 cases
on the matched held-out screen, two more than the selected checkpoint, so it received the full
gate. Across 6,720 held-out AC trials it handled 1,785 (26.56%), survived 3,705 (55.13%), served
about 62.12% demand, and incurred 1,411 thermal-trip trials (21.00%). It lost 19 handled cases, 59
survivals, demand, and thermal safety relative to the selected checkpoint, so it was rejected.
Together with the earlier rejected `1.5` run, this brackets the selected `5.0` maximum gradient
norm while preserving the high-throughput regime.

Doubling only the signed congestion-progress reward from `1.0` to `2.0` added no new computation
and sustained roughly 0.53-0.56M mature SPS. Every checkpoint was exported and screened. The best
checkpoints handled 110/420 trials (26.19%), below the selected checkpoint's 111 (26.43%), without
a compensating survival or demand improvement. Later checkpoints reduced thermal trips but fell
further on handling, so the run was rejected without a larger gate and progress weight `1.0`
remains selected. Halving the same progress credit to `0.5` also sustained roughly 0.53-0.56M
mature SPS. Its final checkpoint improved smoke and matched handled counts and therefore received
the full gate. Across 6,720 held-out AC trials it handled 1,799 (26.77%), survived 3,683 (54.81%),
served about 61.97% demand, and incurred 1,388 thermal-trip trials (20.65%). It lost five handled
cases, 81 survivals, demand, and thermal safety relative to the selected checkpoint. The selected
`1.0` progress credit is therefore bracketed by clean `0.5` and `2.0` runs.

Halving only the persistent congestion-state cost from `5.0` to `2.5` sustained roughly
0.52-0.55M mature SPS. Every checkpoint was exported and screened, but none exceeded 110/420
handled trials (26.19%) versus the selected checkpoint's 111 (26.43%). The late policies also
regressed survival, demand, and thermal-trip rate, so the run was rejected without a larger gate.
Doubling the cost to `10.0` retained roughly 0.52-0.55M mature SPS. Its best checkpoint only tied
the selected smoke handled count at 111/420 while regressing survival, demand, and thermal-trip
rate, so it too was rejected without a larger gate. The selected `5.0` congestion-state cost is
therefore bracketed by clean `2.5` and `10.0` runs.

Randomizing the operating period of short curriculum resets tested broader load/weather context
coverage without changing deployment behavior, observations, actions, or grid physics. It retained
roughly 0.52-0.54M mature SPS. Its best checkpoint only tied the selected smoke handled count at
111/420, while the tied checkpoints regressed survival, demand, and thermal-trip rate. Later
checkpoints improved some secondary context metrics but handled fewer cases, so operating-period
randomization was rejected without a larger gate.

The earlier default-Adam rejection changed momentum settings together, so `beta2=0.999` was
isolated while retaining the selected `beta1=0.999`. It sustained roughly 0.53-0.57M mature SPS
and tied handled count while improving survival and demand on the matched screen, so it received
the full gate. Across 6,720 held-out AC trials it handled 1,796 (26.73%), survived 3,781 (56.26%),
served about 63.13% demand, and incurred 1,275 thermal-trip trials (18.97%). The 17 additional
survivals, 0.61 demand-point gain, and 37 fewer thermal-trip trials did not offset eight fewer
handled cases on the primary score, so it was not selected; the secondary signal warrants a
midpoint test. That clean `beta2=0.995` midpoint sustained roughly 0.52-0.54M mature SPS, but its
best checkpoints only tied the selected 111/420 smoke handled count while survival fell to
51.9-52.6% and thermal trips rose to 26.4-26.9%. It was rejected without a larger gate, leaving
the selected `0.990076...` second-moment coefficient unchanged.

Increasing only Adam epsilon from the selected `6.13e-5` to `1e-4` sustained roughly 0.53-0.54M
mature SPS. Its final checkpoint improved handled count and thermal trips on the matched screen, so
it received the full gate. Across 6,720 held-out AC trials it handled 1,791 (26.65%), survived
3,733 (55.55%), served about 62.00% demand, and incurred 1,178 thermal-trip trials (17.53%). The
134 fewer thermal-trip trials did not offset 13 fewer handled cases, 31 fewer survivals, and lower
demand, so it was rejected.
Lowering epsilon to `1e-5` retained roughly 0.51-0.54M mature SPS and improved the smoke handled
count by one while reducing thermal trips, but the final checkpoint regressed on the matched
screen to 451/1,680 handled (26.85%), 903 survivals (53.75%), and 61.61% demand, with only four
fewer thermal-trip trials. It was rejected without a full gate. The selected `6.13e-5` Adam
epsilon is therefore bracketed by clean `1e-5` and `1e-4` runs.

Increasing only PPO's value-loss coefficient from `0.1` to `0.2` sustained roughly 0.53-0.54M
mature SPS. Its 29.56M-step checkpoint tied smoke handling with a demand gain, so it received the
matched screen, where it handled 451/1,680 trials (26.85%), survived 912 (54.29%), served 62.37%
demand, and incurred 387 thermal-trip trials (23.04%). The small demand gain did not compensate
for three fewer handled cases, 12 fewer survivals, and 43 additional thermal-trip trials. Together
with the earlier rejected `0.05` run, this brackets the selected `0.1` value coefficient.

Tightening both V-trace importance clips from the selected `0.1` to `0.05` sustained roughly
0.53-0.54M mature SPS, but no checkpoint exceeded 110/420 smoke handled trials (26.19%). The
best-handling checkpoint also regressed survival, demand, and thermal-trip rate, so it was rejected
without a larger gate. Together with the earlier rejected `1.0` clips, this brackets the selected
V-trace clipping value.

A two-hidden-layer, 128-wide MLP tested added depth without increasing compute: it had 73.3K
parameters versus 146.3K for the selected one-layer 256-wide model and sustained roughly
0.54-0.56M mature SPS. Its best checkpoint only tied 111/420 smoke handled trials, while losing
six survivals, demand, and thermal safety. Later checkpoints handled fewer cases, so the smaller
deep architecture was rejected without a larger gate.
A 101.9K-parameter two-layer, 160-wide midpoint retained roughly 0.52-0.54M mature SPS but no
checkpoint exceeded 109/420 smoke handled trials (25.95%). Its later policies also showed
excessive switching and thermal failures. It was rejected without a larger gate, confirming that
the smaller-deeper MLP family does not improve the selected one-layer 256-wide speed/quality point.

Shortening curriculum trials from four to two steps increased reset overhead: mature SPS was
roughly 0.51-0.53M after an early trough near 0.46M. Its best checkpoint handled only 109/420
smoke trials (25.95%), with worse survival, demand, and thermal safety. It was rejected without a
larger gate. Together with the rejected six-step run, this brackets the selected four-step
curriculum duration for both quality and throughput.

Doubling the existing reward-only repeat-action penalty from `0.01` to `0.02` retained roughly
0.52-0.53M mature SPS, but no checkpoint exceeded 109/420 smoke handled trials (25.95%). The best
policies also regressed survival, demand, and thermal safety, indicating that the stronger penalty
suppressed useful repeated switching along with loops. It was rejected without a larger gate.
Halving the same penalty to `0.005` sustained roughly 0.52-0.54M mature SPS. Its best checkpoint
only tied the selected 111/420 smoke handled count while losing six survivals, demand, and thermal
safety, so it was also rejected without a larger gate. The selected `0.01` repeat-action penalty
is therefore bracketed on both sides.

Reducing the short-curriculum reset share only slightly, from `99.5%` to `99%`, doubled exact
full-task exposure while sustaining roughly 0.53-0.55M mature SPS. No checkpoint exceeded
110/420 smoke handled trials (26.19%), and thermal-trip rates were substantially worse. The fine
midpoint was rejected without a larger gate; extra full-task exposure did not improve held-out
N-2 recovery.
At the other endpoint, using short curriculum trials for `100%` of resets retained the high-SPS
regime and tied smoke handling with slightly fewer thermal trips, so it received the matched
screen. It handled 451/1,680 trials (26.85%), survived 911 (54.23%), served 62.10% demand, and
incurred 333 thermal-trip trials (19.82%). Eleven fewer trips did not offset three fewer handled
cases, 13 fewer survivals, and slightly lower demand. The selected `99.5%` reset share is therefore
tightly bracketed by rejected `99%` and `100%` runs.

Batching both curriculum N-2 outages onto the same pre-response transition tested the browser
contract in which a person clicks two lines before the policy first responds. This changed only
training curriculum timing; actions, observations, outage persistence, solver physics, and the
deployment path were unchanged. The fresh run sustained roughly 0.54-0.57M mature SPS. Its best
smoke checkpoint tied the selected 111/420 handled cases but lost survival and increased thermal
trips. At the 1,680-trial matched gate, the two best checkpoints handled 457 and 456 cases versus
454 for the selected checkpoint, but survived only 908 and 906 versus 924 and incurred 414 and
376 thermal-trip trials versus 344. The marginal handling gain did not compensate for the larger
safety regression, so the batched timing was rejected without exhaustive QA and the original
one-outage-per-agent-step curriculum timing was restored.

An SPS-first midpoint used 1,280 agents, horizon 60, minibatches of 19,200, and 52,429 per-slot
curriculum steps. This kept a four-minibatch PPO update and approximately preserved global rollout
and curriculum exposure between the selected 1,024-agent setup and the faster but weaker larger
agent-count runs. Mature throughput improved to roughly 0.57-0.59M SPS. Its 69.20M-step checkpoint
improved smoke and matched handling, so it received the exhaustive gate. Across 6,720 held-out AC
trials it handled 1,802 (26.82%), survived 3,671 (54.63%), served about 61.61% demand, and incurred
about 1,429 thermal-trip trials (21.26%). The speed gain did not offset two fewer handled cases, 93
fewer survivals, lower demand, and 117 more thermal-trip trials than the selected checkpoint, so
the 1,280-agent geometry was rejected.
A second 1,280-agent run from random initialization used independent seed 5731 to test whether the
faster geometry's near miss was seed-specific. It reproduced roughly 0.57M mature SPS, but its best
smoke checkpoint handled only 110/420 trials (26.19%), survived 220 (52.38%), served 61.60% of
demand, and incurred 103 thermal-trip trials (24.52%). Since it missed the selected checkpoint on
handling, survival, and thermal safety, it was rejected without a larger gate. The throughput gain
is reproducible, but preserving selected-model quality at 1,280 agents is not.

A cleaner 1,152-agent midpoint used horizon 64, the selected 18,432 minibatch, and 58,254
per-slot curriculum steps. Since `1,152 * 64 = 73,728`, this exactly preserved the selected global
rollout size and four-minibatch update while changing vector parallelism. Mature throughput was
roughly 0.55-0.58M SPS, only a small and inconsistent gain. No checkpoint exceeded the selected
111/420 smoke handled count; the tied checkpoint survived 220 cases versus 225, served less
demand, and incurred more thermal trips. It was rejected without a larger gate, leaving 1,024
agents as the measured speed/quality setting.

A build-only SPS probe compiled the power-grid environment object with `-O3 -march=native` while
leaving PufferLib core and all environment semantics untouched. The fresh selected-recipe probe
reached only roughly 0.47-0.48M SPS near 30M steps, below the normal build at the same phase.
All parity tests passed, but measured end-to-end throughput rejected the compiler flags; the normal
`-O2 -mavx2 -mfma` environment build was restored.

Expanding the per-worker DC factorization cache from one to 32 exact topology-keyed entries tested
whether interleaved vector environments were causing avoidable refactorization. Solver and
environment parity tests passed, but the larger thread-local working set slowed the fresh
selected-recipe probe to roughly 0.44-0.48M SPS by 17M steps, well below the normal trajectory.
The run was stopped at the SPS gate and the single-entry cache was restored.

A one-layer, 256-wide native MinGRU tested whether recurrent state could help with ordered clicks,
persistent outages, and multi-step switching responses without changing the 221 observations or
91 switching-only actions. It trained from random initialization to 80M steps and produced native
deployment-compatible checkpoints directly. Mature throughput was roughly 0.54-0.56M SPS. Its
best smoke checkpoint handled 108/420 trials (25.71%), survived 221 (52.62%), served 60.91% of
demand, and incurred 92 thermal-trip trials (21.90%). The selected stateless MLP handled 111,
survived 225, served 61.52% of demand, and had the same thermal-trip count on this gate. The
recurrent candidate was therefore rejected without a larger gate. Its late training entropy also
fell to roughly 0.09, motivating a separate recurrent entropy-retention test rather than promotion.
A strong recurrent entropy-retention endpoint raised the initial coefficient from `0.15` to `0.20`
and its annealed floor from 50% to 75%. Entropy remained roughly 2.2-2.7, causing excessive
switching and reducing mature throughput to roughly 0.48-0.50M SPS. Its best smoke checkpoint
handled 105/420 trials (25.00%), survived 213 (50.71%), and served 58.89% of demand. It was
rejected on both quality and speed without a larger gate.
A narrower recurrent midpoint retained the initial `0.15` coefficient and raised only its final
floor from 50% to 75%. Mature throughput recovered to roughly 0.52-0.54M SPS, but its best smoke
checkpoint again handled only 108/420 trials (25.71%). A later checkpoint matched the selected
model's 225 survivals and nearly matched demand while reducing thermal trips from 92 to 88, but it
handled only 107 trials. It was rejected without a larger gate. Together these runs show that
neither recurrent memory nor recurrent-specific entropy retention improves the selected stateless
MLP speed/quality point.

Lengthening only the consecutive N-2 curriculum's post-outage response window tested whether its
hard-coded four steps were limiting multi-switch recovery rehearsal. An eight-step endpoint
recovered to roughly 0.54-0.55M mature SPS but no checkpoint exceeded 110/420 smoke handled
trials, so it was rejected without a larger gate. The six-step midpoint sustained roughly
0.53-0.54M mature SPS and reached 112/420 on smoke. Its 29.56M-step checkpoint then handled
455/1,680 matched trials, one more than the selected checkpoint, and advanced to exhaustive QA.
Across 6,720 held-out AC trials it handled 1,774 (26.40%), survived 3,673 (54.66%), served about
62.30% of demand, and incurred 1,485 thermal-trip trials (22.10%). It regressed from the selected
model's 1,804 handled, 3,764 survivals, 62.52% demand, and 1,312 thermal-trip trials. The longer
sequence windows and their temporary compile-time configuration were removed, restoring the
original four-response-step curriculum.
Using the full 30% N-2 sequence share from the start instead of ramping it over the curriculum
created a very harsh initial distribution and reduced early throughput through frequent terminal
resets. Mature SPS recovered to roughly 0.53-0.55M. Its best checkpoint only tied 111/420 smoke
handled trials while losing seven survivals, demand, and thermal safety, so constant-from-start
sequence exposure was rejected without a larger gate.
Accelerating the original zero-to-30% N-2 sequence ramp by 2x preserved an easy start and reached
roughly 0.54-0.55M mature SPS. Its 51.68M-step checkpoint handled 113/420 smoke trials and
457/1,680 matched trials, clearing the selected model by two and three cases respectively. On the
6,720-trial exhaustive gate, however, it handled 1,796 (26.73%), survived 3,669 (54.60%), served
about 61.66% of demand, and incurred 1,373 thermal-trip trials (20.43%). It lost eight handled
cases, 95 survivals, demand, and thermal safety relative to the selected checkpoint, so the 2x
ramp was rejected.
The gentler 1.5x ramp midpoint recovered to roughly 0.55-0.56M late SPS but never exceeded the
selected 111/420 smoke handled count. Its tied checkpoint had lower survival and demand with one
additional thermal-trip trial; later checkpoints handled fewer cases. It was rejected without a
larger gate, and the original linear N-2 sequence ramp was restored.

The default cosine learning-rate schedule reaches zero just as the linear N-2 curriculum reaches
its full share, so nonzero final LR floors were tested as pure PPO changes. A 25% floor retained
roughly 0.53-0.54M mature SPS but destabilized late training into excessive switching; no
checkpoint exceeded 106/420 smoke handled trials. A 10% floor retained roughly 0.55M mature SPS
and reached 112/420 smoke and 455/1,680 matched handled trials. Its exhaustive checkpoint handled
1,788/6,720 (26.61%), survived 3,676 (54.70%), served about 61.39% of demand, and incurred 1,298
thermal-trip trials (19.32%). Fourteen fewer trips did not offset 16 fewer handled cases, 88 fewer
survivals, and lower demand. Finally, a 5% floor only tied 111/420 smoke handling while regressing
all secondary metrics. The selected zero-floor cosine schedule is therefore retained.

An empirical observation-conditioning audit sampled 200,000 unrestricted DC rollout states and
20,000 AC rollout states across randomized operating contexts and outages. Binary topology
features remained exactly in `[0, 1]`; injections stayed roughly within `[-1.17, 1.68]` per unit,
AC node voltages within `[0, 1.18]`, and generator reactive power within `[-0.40, 1.07]` per unit.
AC branch loading stayed roughly within `[-2.51, 4.11]`; deliberately adversarial random DC
switching produced rare loading tails near `+-13.7`, while randomized reset states peaked near
6.06. Constant or low-variance dimensions corresponded to physically inactive AC-only features,
fixed topology elements, or low-variance weather channels rather than an accidental scaling bug.
The fixed physical-base normalization was therefore retained; fitted statistics, clipping, and
another encoding change were not warranted by the measured ranges.

A fresh 120M-step run tested whether the selected linear curriculum simply needed more optimizer
updates after reaching its full N-2 share near 67M steps. It used the selected zero-floor cosine
schedule over the longer single run, with no checkpoint loading or fine-tuning, and sustained
roughly 0.52-0.56M mature SPS. Screening every checkpoint found no policy above 110/420 smoke
handled trials. Policies after roughly 95M steps regressed sharply in handling, survival, and
demand as switching increased. The longer budget was rejected without a larger gate; 80M remains
the measured speed/quality training horizon.

Targeted AC rehearsal was then bracketed more finely between the selected 1% and rejected 2%
shares. A 1.5% run recovered to roughly 0.52M mature SPS and reached 112/420 smoke handled trials
with improved demand. On the matched 1,680-trial screen it handled only 450 cases versus 454 for
the selected checkpoint, survived 922 versus 924, served 62.90% versus 62.13% demand, and incurred
351-352 thermal-trip trials versus 344. The 1.25% midpoint recovered to roughly 0.56M late SPS but
never exceeded 109/420 smoke handling. Both were rejected; together with the previously rejected
0.75% run, the selected 1% targeted AC share remains bracketed.

A post-outage-only safe-state maintenance bonus tested whether the global safe-reward signal could
be concentrated on holding recovered grids without over-rewarding intact no-op states. Doubling
the `0.05` reward only after an outage retained roughly 0.54M mature SPS and reached 112/420 smoke
handled trials. Its matched checkpoint tied 454/1,680 handling and nearly tied survival while
improving demand to 63.05%, but incurred 361 thermal-trip trials versus 344. The 1.5x midpoint only
tied 111/420 smoke handling once, regressed survival and thermal safety, and converged to excessive
switching at roughly 0.52M SPS. Both bonuses and their temporary compile-time surface were removed;
the uniform `0.05` safe-step reward remains selected.

Prioritized-replay importance correction was isolated from the already bracketed replay alpha.
Reducing `prio_beta0` from the selected `1.0` to `0.75` retained roughly 0.54M mature SPS, but its
best checkpoint handled only 110/420 smoke trials (26.19%), survived 221, served 60.89% of demand,
and incurred 93 thermal-trip trials. The selected checkpoint handled 111, survived 225, served
61.52% of demand, and incurred 92 trips on the identical cases. A clean `0.875` midpoint reached
roughly 0.52-0.55M mature SPS and tied 111 handled trials, but survived only 221 and incurred 126
thermal-trip trials; its 62.19% demand result did not compensate for the safety regressions. Both
runs were rejected without larger gates, leaving full importance correction selected.

Adam's first-moment coefficient was then isolated because the earlier default-Adam run changed
both momentum terms and the later clean follow-ups varied only `beta2`. Reducing `beta1` from the
sweep-selected `0.999` to a conservative `0.995` midpoint was computationally free and reached
roughly 0.55-0.56M mature SPS. Its best checkpoint only tied 111/420 smoke handled trials, however,
while surviving 217, serving 60.79% of demand, and incurring 109 thermal-trip trials. It was
rejected without a larger gate, leaving `beta1=0.999` selected.

Halving the small physical switch cost from `0.002` to `0.001` tested whether cheaper multi-switch
sequences could improve N-2 recovery while the separate `0.25` ineffective/secure-state penalty
continued to discourage churn. Mature throughput improved to roughly 0.55-0.56M SPS and the
36.94M-step checkpoint reached 112/420 smoke handled trials, so it received the matched gate. It
then tied the selected checkpoint at 454/1,680 handled, but survived only 893 trials versus 924,
served 61.30% of demand versus 62.13%, and incurred 390 thermal-trip trials versus 344. The lower
cost was rejected without exhaustive QA because it was dominated on every secondary safety metric.
Doubling the physical switch cost to `0.004` retained roughly 0.55-0.56M mature SPS and produced
three checkpoints with 112/420 smoke handled trials. All three received the matched gate; they
handled 447, 447, and 452/1,680 trials versus 454 for the selected checkpoint. The best survival
count was 911 versus 924, and none offered a compensating primary or safety gain. They were
rejected without exhaustive QA, leaving the selected `0.002` physical switch cost bracketed by
clean `0.001` and `0.004` runs.

A `0.75` midpoint for the one-step-filtered recovery fraction mixed 25% unrestricted overloaded
N-1 starts into the late curriculum, between the selected fully filtered setting and the earlier
rejected 50/50 endpoint. The harder resets reduced early throughput, but mature SPS recovered to
roughly 0.54-0.55M. No checkpoint exceeded the selected 111/420 smoke handled trials; tied
checkpoints regressed survival and thermal safety. It was rejected without a larger gate, leaving
fully one-step-filtered N-1 recovery rehearsal selected.

Forcing the rare unrestricted ordinary episodes to sample N-2 only tested contingency-count domain
randomization while the dominant curriculum continued to supply N-1 recovery examples. It retained
roughly 0.54-0.55M mature SPS, but no checkpoint exceeded 110/420 smoke handled trials and there
was no compensating safety gain. It was rejected without a larger gate; ordinary episodes retain
the unfiltered N-1/N-2 count mixture used by evaluation.

The built-in progressive recoverable-line schedule (`curriculum_outage_line=-2`) tested an
easy-to-hard expansion from a small interleaved set to every eligible line, without filtering the
final task. Early throughput fell to roughly 0.47M SPS but mature throughput recovered to about
0.54M. No checkpoint exceeded 110/420 smoke handled trials and the late policies regressed
survival and thermal safety. It was rejected without a larger gate; recovery starts continue to
sample every eligible line from the beginning.

Reducing PPO replay ratio from `1.0` to `0.75` was an SPS-first test of fewer optimizer updates per
environment sample. Mature throughput rose to roughly 0.60M SPS, materially above the selected
0.52-0.56M range. Its 29.56M-step checkpoint tied 111/420 smoke handled trials and received a
matched screen because speed is a first-class objective. It handled 451/1,680 trials versus 454,
survived 913 versus 924, served 61.93% of demand versus 62.13%, and incurred 384 thermal-trip
trials versus 344. The substantial speed gain did not preserve the selected quality point, so it
was not promoted; a midpoint is warranted.
The clean `0.875` replay midpoint reached roughly 0.56-0.59M late SPS, but no checkpoint exceeded
110/420 smoke handled trials and its late policies materially regressed survival and demand. It was
rejected without a larger gate. Together with the previously uncompetitive `1.4` artifact, these
runs retain replay ratio `1.0` as the measured quality setting while documenting `0.75` as a faster
but weaker operating point.

A one-layer 224-wide MLP tested the architecture midpoint between the selected 256-wide model and
the faster but weaker 192-wide endpoint. It reduced parameters from 146.3K to 120.8K and reached
roughly 0.56-0.57M late SPS, but no checkpoint exceeded 106/420 smoke handled trials (25.24%). The
modest speed/model-size gain did not justify the large quality loss, so width 256 remains selected.

An evidence-based SPS combination paired the 2,048-agent geometry with `safe_step_reward=0.025`.
Both components had been measured independently: the larger geometry improved speed, survival,
and demand but lost handled cases, while the lower reward recovered handled cases at 1,024 agents.
The fresh combination sustained roughly 0.63-0.64M SPS and its 44.38M-step checkpoint reached
112/420 smoke handled trials. On the matched screen it handled 452/1,680 trials versus 454,
survived 898 versus 924, served 61.76% of demand versus 62.13%, and incurred 409 thermal-trip
trials versus 344. The effects did not combine favorably; it was rejected without exhaustive QA.

Compiler-instrumented standalone profiling of the default double benchmark attributed about 61%
of runtime to `power_grid_solve_scaled`, 14% to observation construction, and 8% to topology
validation. A semantics-preserving cache-hit probe reused the cached node-row map and skipped the
duplicate factor-cache search and key check. All 16 parity/training tests passed, but three
one-million-episode timings were unchanged: 8.68-8.76 seconds before versus 8.73-8.74 seconds
after. The edit was removed rather than retaining unmeasured complexity.

Training-mode profiling then correctly included the refined-float DC path. Its residual correction
made a standalone exact-double solve appear faster, but that did not transfer to vector training:
the full exact-double candidate sustained only roughly 0.47-0.50M SPS versus 0.52-0.56M selected.
Its 73.80M-step checkpoint improved the matched handled count to 461/1,680 and reduced thermal
trips, so it received exhaustive QA. Across 6,720 held-out AC trials it handled 1,800 (26.79%),
survived 3,680 (54.76%), served about 61.77% demand, and incurred 1,280 thermal-trip trials
(19.05%). Thirty-two fewer trips did not offset four fewer handled cases, 84 fewer survivals,
lower demand, and lower training throughput. Exact-double training was rejected and the simpler
refined-float path was restored.

## Browser demo

```sh
source /path/to/emsdk/emsdk_env.sh
./build.sh power_grid --web
python -m http.server 8000 --directory build/web/power_grid
```

The current browser resource is the historical 256x3 MinGRU compatibility baseline and uses the AC
solver. It is not the selected one-layer MLP candidate. Automatic outages are disabled; the person
may click at most two lines. Infeasible clicks remain applied. The existing public deployment may
lag this source revision and must not be treated as evidence for the current 91-action contract or
the selected candidate until the correct policy is integrated, rebuilt, and redeployed.

## Limitations

IEEE-14 is a compact research benchmark. Ratings, renewable placement, weather exposure,
two-busbar layouts, dispatch, and outage distributions are assumptions. Real deployment requires
authenticated network cases and ratings, route/conductor geometry, dispatch and remedial-action
rules, protection/interlocks, transient and frequency studies, operator review, and monitoring.
No current checkpoint meets the requested 95% exhaustive handled gate.
