"""Frozen discrete deformable–deformable comparisons on native, separately installed engines."""


def run_case(case, *, engine, device):
    if engine == "newton":
        from .newton_runner import observe, validate
    elif engine == "phymeo":
        from .phymeo_runner import observe, validate
    else:
        raise ValueError(engine)
    data = observe(case, device)
    try:
        validate(data, case)
    except AssertionError as exc:
        exc.observed = data
        raise
    return data
