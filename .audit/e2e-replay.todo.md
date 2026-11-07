# e2e replay runner — poteto Feature playbook

- [x] 1. `how` over the affected subsystem. Done in-thread: read pipeline.py (615 lines,
      stage split + reusable llm_stage/dispatch_stage/build_alert_context), enrich.py config
      (KAFKA_INPUT_TOPIC/OUTPUT_TOPIC, REQUIRED_KEYS), and probed both interpreters.
      skip (subagent fan-out): the finding that decides the design is one command's output,
      not a survey — kafka is .venv-only, pandas/pyarrow Anaconda-only.
- [x] 2. `architect` parallel design exploration. skip: the interpreter split leaves exactly
      one viable topology (see Decisions below). Two candidates were considered and one is
      disqualified on honesty grounds, not on taste, so there is nothing to race.
- [x] 3. Throughput checkpoint (below).
- [ ] 4. Delegate code-writing with a named data shape + organizing structure.
- [ ] 5. Verify on the matching surface: a real run against live Kafka + enrich + Ollama,
      ending in a real LINE broadcast. Not a unit test standing in for it.
- [ ] 6. Rebase into small ordered commits.
- [ ] 7. `interrogate` if the design is contested. skip unless step 5 surfaces a dispute.
- [ ] 8. Opening a PR. skip: this repo commits to a branch, no forge PR in use.

## Throughput checkpoint

- **Blocking first steps.** DONE: the interpreter probe. It is the gate — it rules out a
  single-process consumer-scorer, which was the obvious first design.
- **Independent workstreams.** n/a: one module, one owner. The stages are sequential by
  construction (each consumes the previous stage's artifact), so there is nothing to
  parallelize. Fanning out would create writers on one file.
- **Shared mutable state.** The checkout itself. No worktree: `data/` and `models/` are
  gitignored and absent from one, and CLAUDE.md records a junction-plus-rm-rf that destroyed
  `data/`. Resolved by exclusive-writer instead — the delegate is the only writer until it
  returns, and the parent writes no files meanwhile.
- **Smallest safe decomposition.** One owner. Named because the work is a single integration
  seam whose correctness is only observable by running the whole chain.

## Decisions

1. **The event payload carries the 94 scaled features.** The alternative — publish only
   identifiers and have the scorer look the row up in the parquet by index — would make the
   Kafka leg decorative: the two halves still would not have met, and the claim would be
   false in exactly the way this work exists to fix. Cost of the honest choice: a replayed
   event is a pre-extracted CICFlowMeter flow, not a raw firewall log, so the claim this
   earns is "end to end from a replayed flow event", never "from raw traffic". A real
   deployment needs a feature-extraction stage that does not exist here.
2. **Stages are a table, not a chain of ifs** (principle-model-the-domain). Each row names
   its interpreter, argv, the artifact it must produce and its precondition, so the
   coordinator walks the table and each stage's contract is visible in one place.
3. **Reuse pipeline.py, do not fork it.** The new scorer must write the same payload shape
   as `STAGE_JSON`, so `llm_stage` and `dispatch_stage` are imported and called unmodified.

## Findings from rendering the measured alerts into the new LINE format (2026-11-08)

These came out of formatting work, not from an experiment, and two of them belong in
chapter 4 section 4.11. They are recorded here rather than in CLAUDE.md because a delegate
holds that file for the e2e work.

1. **Fixed.** 50 of the 100 alerts in `reports/llm_faithfulness_raw.jsonl` carry neither
   `technique_id` nor `mitre_tactic`, and `ctx_tactic` is null in all 50 as well, so no
   deterministic fallback exists. `llm_explain._technique_context` votes over the RAG hits
   and returns nothing when they disagree. The message now says so explicitly. See commit
   2b0c0e3.

2. **A concrete instance of the measurement blind spot CLAUDE.md names, not fixed.** Four
   alerts, all DoS Slowhttptest, have prose reading "หลักฐานจาก RAG ไม่สอดคล้องกันพอที่จะระบุ
   เทคนิค MITRE ได้" while their own `technique_id` field asserts T1499.002 with tactic
   impact. The prose disclaims what the structured field states, in one message. Every
   faithfulness check passes on these rows because the checks verify entities (IPs, ports)
   and these entities are all correct. CLAUDE.md already states the blind spot as "prose
   whose entities are all correct and whose meaning is wrong" and reports
   `alerts_with_bad_claim` at 0.000. This is 4 of 100 rows where that metric is blind by
   construction, which is worth reporting as a named limitation rather than leaving the
   0.000 to speak for itself. The old four-line LINE message hid it by never printing the
   technique.

3. **The demo will look repetitive and that is the dataset, not the system.** Across the 100
   alerts there are only 27 distinct (source IP, destination IP, destination port) triples,
   and 53 of the 100 are the identical triple 172.16.0.1 -> 192.168.10.50:80. CICIDS-2017
   runs every DoS and DDoS variant from one attacker host against one victim. For a defense
   demo, pick rows across the port range (21, 22, 80, 444, 8080) or the alerts will all read
   as the same event.

4. **Measured, n=3.** Three broadcasts moved the channel's quota consumption from 8 to 14,
   so 2 units per broadcast. This reproduces the earlier n=1 measurement and confirms LINE
   bills per recipient. Usable capacity on this channel is 150 broadcasts per month, not 300.
