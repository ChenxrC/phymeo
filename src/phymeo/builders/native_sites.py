"""Cold compilation-bound resolution of public body sites."""

from dataclasses import dataclass

from ..sensors.body_site import BodySite


@dataclass(frozen=True)
class NativeBodySites:
    compilation: object
    sites: tuple
    bodies: tuple
    contract_version: int = 1


def bind_body_sites(template, sites):
    sites = tuple(sites)
    if not sites or any(not isinstance(site, BodySite) for site in sites):
        raise ValueError("body site binding requires nonempty BodySite values")
    if len({site.name for site in sites}) != len(sites):
        raise ValueError("body site names must be unique")
    names = dict(template.body_bindings)
    if any(site.body not in names for site in sites):
        raise ValueError("body site references unknown compiled public body")
    return NativeBodySites(template.compilation, sites, tuple(names[site.body] for site in sites))
