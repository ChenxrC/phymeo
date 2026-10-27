"""Independent SI equations with the frozen P6 FP32 error budgets."""

from .articulated import _audit_slider_substep


def audit_device_slider_substep(fixture, before, after, events, predicted_q, predicted_v, *, h):
    return _audit_slider_substep(
        fixture,
        before,
        after,
        events,
        predicted_q,
        predicted_v,
        h=h,
        limits=(5e-6, 5e-5, 5e-8, 5e-5, 2e-5),
        label="P6",
    )
