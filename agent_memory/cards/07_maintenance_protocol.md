# Maintenance Protocol

## Purpose

This card defines the operational rule for keeping repo-local agent memory aligned with the actual codebase.

## Hard rule

Any change that modifies repository behavior or the engineering contract must update `agent_memory/` in the same task before the work is considered complete.

This includes:

- code changes
- config changes
- CLI changes
- output directory changes
- training and evaluation workflow changes
- documentation changes that redefine the current source of truth

## Minimum required updates

For every qualifying change:

1. update at least one relevant memory card
2. append the change to `cards/03_recent_modifications.md`
3. update `cards/05_reuse_rules.md` if reuse guidance changed
4. update this card if the maintenance process itself changed
5. update `manifest.yaml` if cards were added, removed, or renamed

## Review checklist

Before closing a task, confirm:

- the code matches the claimed behavior
- docs and CLI examples reflect the new behavior
- tests cover the new contract where practical
- agent memory reflects the new source of truth

## Non-compliance examples

The following are considered incomplete work:

- adding a new training flag without updating memory
- moving outputs to a new directory without updating reuse rules
- changing checkpoint behavior without updating the training pipeline card
- closing an audit or implementation task while memory still describes the old contract

## Trusted conclusion

`agent_memory/` is part of the repository contract, not optional commentary. Future agents should treat memory synchronization as required maintenance, not as an afterthought.
