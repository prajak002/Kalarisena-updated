#!/usr/bin/env python3
"""Regression tests for ModeSwitch.

Run: .venv/bin/python tests/test_mode_switch.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.switch.mode_switch import Mode, ModeSwitch, SwitchConfig

_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        _FAILURES.append(name)


def feat(cp=0.1, mom=0.0, h=0.75, fallen=False, upright=True, step=0) -> dict:
    return {"cp_margin": cp, "momentum_norm": mom, "base_height": h,
            "is_fallen": fallen, "is_upright": upright, "step_index": step}


def test_stays_nominal_when_stable() -> None:
    print("test_stays_nominal_when_stable")
    s = ModeSwitch(SwitchConfig())
    modes = [s.step(feat(step=i)) for i in range(50)]
    check("no spurious transitions", all(m is Mode.NOMINAL for m in modes))
    check("empty transition log", s.transition_log == [])


def test_cp_margin_triggers_fall() -> None:
    print("test_cp_margin_triggers_fall")
    cfg = SwitchConfig()
    s = ModeSwitch(cfg)
    s.step(feat(cp=0.1, step=0))
    m = s.step(feat(cp=cfg.delta1 - 0.001, step=1))
    check("cp_margin below delta1 -> FALL", m is Mode.FALL)
    check("trigger recorded as cp_margin",
          s.transition_log[-1]["trigger"] == "cp_margin", str(s.transition_log[-1]))


def test_momentum_triggers_fall() -> None:
    print("test_momentum_triggers_fall")
    cfg = SwitchConfig()
    s = ModeSwitch(cfg)
    m = s.step(feat(cp=0.5, mom=cfg.delta2 + 1.0, step=0))
    check("momentum above delta2 -> FALL", m is Mode.FALL)
    check("trigger recorded as momentum_norm",
          s.transition_log[-1]["trigger"] == "momentum_norm")


def test_delta1_boundary_is_inclusive() -> None:
    """cp_margin exactly at delta1 must trigger (the law uses <=)."""
    print("test_delta1_boundary_is_inclusive")
    cfg = SwitchConfig()
    s = ModeSwitch(cfg)
    check("cp_margin == delta1 -> FALL", s.step(feat(cp=cfg.delta1, step=0)) is Mode.FALL)


def test_full_recovery_cycle_and_dwell() -> None:
    print("test_full_recovery_cycle_and_dwell")
    cfg = SwitchConfig(min_dwell_steps=5)
    s = ModeSwitch(cfg)
    s.step(feat(cp=0.01, step=0))                     # -> FALL
    s.step(feat(cp=0.01, h=0.1, step=1))              # -> RECOVERY (base_height)
    check("reached RECOVERY", s.current_mode is Mode.RECOVERY)

    # Must NOT return to nominal before the dwell requirement is met.
    for i in range(cfg.min_dwell_steps - 1):
        s.step(feat(cp=0.2, h=0.75, upright=True, step=10 + i))
    check("still RECOVERY before dwell elapses", s.current_mode is Mode.RECOVERY)
    s.step(feat(cp=0.2, h=0.75, upright=True, step=100))
    check("returns to NOMINAL after dwell", s.current_mode is Mode.NOMINAL)


def test_dwell_counter_resets_when_not_upright() -> None:
    """An interrupted recovery must restart the dwell count, not accumulate."""
    print("test_dwell_counter_resets_when_not_upright")
    cfg = SwitchConfig(min_dwell_steps=4)
    s = ModeSwitch(cfg)
    s.step(feat(cp=0.01, step=0))
    s.step(feat(cp=0.01, h=0.1, step=1))
    for i in range(3):
        s.step(feat(cp=0.2, h=0.75, upright=True, step=10 + i))
    s.step(feat(cp=0.2, h=0.4, upright=False, step=20))    # interruption
    for i in range(3):
        s.step(feat(cp=0.2, h=0.75, upright=True, step=30 + i))
    check("did not return to NOMINAL on stale dwell credit",
          s.current_mode is Mode.RECOVERY)


def test_reset_clears_state() -> None:
    print("test_reset_clears_state")
    s = ModeSwitch(SwitchConfig())
    s.step(feat(cp=0.0, step=0))
    s.reset()
    check("mode reset to NOMINAL", s.current_mode is Mode.NOMINAL)
    check("transition log cleared", s.transition_log == [])


def test_transition_log_is_a_copy() -> None:
    print("test_transition_log_is_a_copy")
    s = ModeSwitch(SwitchConfig())
    s.step(feat(cp=0.0, step=0))
    log = s.transition_log
    log.append({"bogus": True})
    check("external mutation does not corrupt internal log", len(s.transition_log) == 1)


def main() -> int:
    test_stays_nominal_when_stable()
    test_cp_margin_triggers_fall()
    test_momentum_triggers_fall()
    test_delta1_boundary_is_inclusive()
    test_full_recovery_cycle_and_dwell()
    test_dwell_counter_resets_when_not_upright()
    test_reset_clears_state()
    test_transition_log_is_a_copy()
    print(f"\n{'ALL TESTS PASSED' if not _FAILURES else 'FAILURES: ' + ', '.join(_FAILURES)}")
    return 0 if not _FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
