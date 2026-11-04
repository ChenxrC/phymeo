"""Run from an external directory against a freshly installed, explicitly selected Panda wheel."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    site = args.site.resolve(strict=True)
    sys.path.insert(0, str(site))
    import numpy as np

    from phymeo.builders import panda
    from phymeo.experiments import panda_cube

    dist = distribution("phymeo")
    assert site in panda.PANDA_URDF.resolve().parents
    assert site in Path(panda.__file__).resolve().parents
    assert site in Path(panda_cube.__file__).resolve().parents
    direct = json.loads(dist.read_text("direct_url.json"))
    digest = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    assert direct["archive_info"]["hashes"]["sha256"] == digest
    names = ("panda.urdf", "NOTICE", "LICENSE", "SOURCES.md")
    hashes = {}
    with ZipFile(args.wheel) as archive:
        for name in names:
            rel = "phymeo/builders/panda_assets/" + name
            installed = (site / rel).read_bytes()
            assert installed == archive.read(rel)
            hashes[name] = hashlib.sha256(installed).hexdigest()
    asset = panda.load_panda()
    q = asset.configuration(
        [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4], 0.076
    )
    left, _ = asset.authored_link_pose("panda_leftpad", q)
    right, _ = asset.authored_link_pose("panda_rightpad", q)
    opening = np.linalg.norm(left - right)
    assert len(asset.collisions) == 19 and abs(opening - 0.07596) < 1e-12
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(
            dict(
                wheel_sha256=digest,
                files=hashes,
                body_proxies=len(asset.collisions),
                pad_gap_m=opening,
                urdf=str(panda.PANDA_URDF),
            ),
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
