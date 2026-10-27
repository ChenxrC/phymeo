"""Check local documentation links and offline SVG/HTML structure (stdlib only)."""

import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for name in ("href", "src"):
            value = values.get(name)
            if value:
                self.targets.append(value)
                if tag in {"script", "link", "img", "iframe"} and urlsplit(value).scheme:
                    self.external_resources.append(value)


def main():
    errors = []
    checked = 0
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), *sorted((ROOT / "docs").rglob("*.html")), *sorted((ROOT / "tyx_flaw").rglob("*.md"))]
    for path in files:
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".html":
            parsed = Links()
            parsed.feed(source)
            targets = parsed.targets
            if parsed.external_resources:
                errors.append(f"{path.relative_to(ROOT)}: external rendering resources: {parsed.external_resources}")
        else:
            source_without_code = re.sub(r"```.*?```", "", source, flags=re.S)
            targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source_without_code)
        for target in targets:
            target = target.strip().strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            checked += 1
            local = (path.parent / unquote(parts.path)).resolve()
            if not local.is_relative_to(ROOT):
                errors.append(f"{path.relative_to(ROOT)}: link outside repository: {target}")
            elif not local.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing local target: {target}")
    for path in sorted((ROOT / "docs").rglob("*.svg")):
        try:
            source = path.read_text(encoding="utf-8")
            element = ET.fromstring(source)
            ids = [node.attrib["id"] for node in element.iter() if "id" in node.attrib]
            if len(ids) != len(set(ids)):
                errors.append(f"{path.relative_to(ROOT)}: duplicate SVG IDs")
            for reference in re.findall(r"url\(#([^)]+)\)", source):
                if reference not in ids:
                    errors.append(f"{path.relative_to(ROOT)}: unresolved SVG reference {reference}")
            if "viewBox" not in element.attrib:
                errors.append(f"{path.relative_to(ROOT)}: SVG missing viewBox")
        except ET.ParseError as exc:
            errors.append(f"{path.relative_to(ROOT)}: malformed SVG: {exc}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation checks passed: {len(files)} documents, {checked} local links; SVG references valid.")
    print("Scope: local file links and SVG structure, not external URLs, Markdown anchors or visual layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
