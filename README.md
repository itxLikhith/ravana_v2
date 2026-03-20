# RAVANA v2 — GRACE Architecture

**Governance · Reflection · Adaptation · Constraint · Exploration**

A proto-homeostatic cognitive system with fully bounded dynamics.

---

## 🧭 Phase A Complete: Stable Physics

RAVANA v2 now operates as a **closed-loop regulated system** with four layers of control:

| Layer | Function | Mechanism |
|-------|----------|-----------|
| **Predictive** | Foresight | Look-ahead dampening based on horizon projection |
| **Boundary** | Soft resistance | Sigmoid pressure curve (air, not brick wall) |
| **Center** | Homeostasis | Anti-overshoot pull toward target dissonance |
| **Hard Stop** | Absolute limits | Constraints that cannot be breached |
| **Constitution** | Identity enforcement | Final clamp that overrides all downstream |

### The Identity Clamp — Keystone Innovation

The system now has **constitutional enforcement**: no behavioral layer can override the identity bounds. This closes the loophole where perfect regulation could be bypassed downstream.

> "Predictive dampening = foresight 👁️ | Constraints = law 🚧 | Identity clamp = constitution 📜"

### Current Metrics (Healthy Baseline)

- **Dissonance range**: 0.18–0.84 (healthy exploration, not hugging extremes)
- **Identity range**: 0.11–0.94 (plasticity without collapse)
- **Constraint hits**: 8/100 (curious but disciplined)
- **Mode switches**: 31 (responsive, not stuck in loops)

### Clamp Diagnostics

The governor now tracks:
- `clamp_activations` — how often the clamp intervenes
- `clamp_corrections_total` — cumulative correction magnitude
- `upstream_suggestions` — total controller recommendations
- `clamp_correction_history` — per-episode correction log

**Ideal state**: Clamp acts as safety net, not active controller.

---

## 🚀 Phase B: Emergent Intelligence

Next: Signal interpretation, strategy formation, and meta-learning — built on top of stable physics.

**Principle**: Don't add intelligence. Let intelligence emerge on top of stable physics.

---

## Architecture

```
core/
  governor.py      — Central regulation (first-class citizen)
  identity.py        — Identity dynamics with momentum
  resolution.py      — Conflict resolution engine
  state.py           — State manager (wires components)

probes/
  constraint_stress.py      — Monitor constraint system
  exploration_pressure.py     — Track exploration drive
  learning_signal.py        — Extract learning indicators

training/
  pipeline.py        — Training orchestration
```

---

## Quick Start

```bash
python run_training.py
```

---

## License

MIT — Built for the RAVANA-AGI-Research initiative.
