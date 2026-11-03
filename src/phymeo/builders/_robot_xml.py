"""Validation for the bounded XML vocabularies supported by robot importers."""

import numpy as np


def _name(element, attribute, where):
    value = element.get(attribute, "").strip()
    if not value:
        raise ValueError(f"{where} requires a nonempty {attribute}")
    return value


def _attributes(element, allowed, where):
    extra = set(element.attrib) - allowed
    if extra:
        raise ValueError(f"{where} contains unsupported attribute(s) {sorted(extra)}")


def _children(element, allowed, where):
    for child in element:
        if child.tag not in allowed:
            raise ValueError(f"{where} contains unsupported <{child.tag}>")


def _child(element, tag, where, *, required=True):
    items = element.findall(tag)
    if len(items) > 1 or (required and not items):
        raise ValueError(f"{where} requires exactly one <{tag}>")
    return items[0] if items else None


def _numbers(element, key, count, where, default=None):
    if key not in element.attrib:
        if default is not None:
            return np.array(default, dtype=float)
        raise ValueError(f"{where} requires {key}")
    try:
        values = np.array([float(s) for s in element.attrib[key].split()])
    except ValueError as exc:
        raise ValueError(f"{where} has invalid {key}") from exc
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError(f"{where} requires {count} finite value(s) for {key}")
    return values


def _number(element, key, where):
    return float(_numbers(element, key, 1, where)[0])
