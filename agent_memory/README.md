# Agent Memory Module

`agent_memory/` is a durable, repo-local memory layer for future agents working on this benchmark.

It stores:

- a structured manifest of memory cards,
- detailed markdown cards that merge project facts, recent modifications, and audit findings,
- a lightweight Python loader for listing, reading, and searching cards.

The intended workflow is:

1. read `manifest.yaml`,
2. load the relevant card(s) through `agent_memory.registry`,
3. treat cards with `trust: trusted` as the default source of truth unless newer code contradicts them,
4. treat cards with `trust: usable_with_verification` as reusable but not final,
5. treat `stale_or_conflicting` findings as repair backlog.

This module is not a runtime dependency of training or evaluation. It is an operational memory package for maintenance, audits, and handoff.

## Mandatory synchronization rule

Any repo change that modifies code, config, docs, outputs contract, CLI behavior, or experiment workflow must also update `agent_memory/` in the same task.

The minimum requirement is:

1. update at least one relevant memory card,
2. append the change to `cards/03_recent_modifications.md`,
3. update `cards/05_reuse_rules.md` or `cards/07_maintenance_protocol.md` if the change affects engineering process or repo contract,
4. keep `manifest.yaml` aligned if a new card is added.

A task is not complete until the operational memory reflects the new source of truth.
