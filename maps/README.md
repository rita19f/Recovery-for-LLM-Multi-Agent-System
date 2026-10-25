# Maps

- `development/`: visible during implementation and prompt tuning.
- `evaluation/`: frozen held-out maps used only after the protocol is fixed.

Every map contains an immutable `id`, schema version, shared topology, agent
starts/goals, complementary hazards, phase budgets, reference routes, and
precommitted structured fault schedules.

