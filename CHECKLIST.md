# Large-N Calibration Round 2 Checklist

- [x] Create experiment charter and checklist.
- [x] Audit current large-N normalization and difficulty scaling code.
- [x] Identify the main failure mode behind unstable normalized scores.
- [x] Implement reference normalization stabilization.
- [x] Tighten any remaining large-N coverage/goal difficulty rules if needed.
- [x] Update docs for the revised calibration contract.
- [x] Retrain BC+PPO on density-preserving N={20,40}.
- [x] Re-run cross-N evaluation for the new checkpoint.
- [x] Compare new results against the previous BC+PPO large-N run.
- [x] Run tests.
- [ ] Write updated findings and next-step recommendation.
