"""Reject static imports outside the permitted tyx public adapter boundary.

This is an architecture check, not a Python sandbox. Dynamic reflection is not
exhaustively analyzed; code review remains necessary for new adapter code.
"""

import ast
from pathlib import Path


def violations(source: str, *, in_adapter: bool) -> list[str]:
    tree = ast.parse(source)
    errors, aliases = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "tyx" or item.name.startswith("tyx."):
                    if not in_adapter:
                        errors.append(f"line {node.lineno}: tyx import outside backends")
                    if item.name != "tyx":
                        errors.append(f"line {node.lineno}: import from the public tyx root only")
                    aliases.add(item.asname or item.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tyx" or node.module.startswith("tyx."):
                if not in_adapter:
                    errors.append(f"line {node.lineno}: tyx import outside backends")
                if node.module != "tyx" or any(
                    n.name.startswith("_") or n.name == "*" for n in node.names
                ):
                    errors.append(f"line {node.lineno}: import named public tyx root exports only")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node
            chain = []
            while isinstance(root, ast.Attribute):
                chain.append(root.attr)
                root = root.value
            if (
                isinstance(root, ast.Name)
                and root.id in aliases
                and any(n.startswith("_") for n in chain)
            ):
                errors.append(f"line {node.lineno}: private tyx attribute access")
    return errors


def main():
    root = Path(__file__).resolve().parents[1] / "src" / "phymeo"
    errors = []
    files = list(root.rglob("*.py"))
    for path in files:
        for error in violations(
            path.read_text(encoding="utf-8"), in_adapter=path.is_relative_to(root / "backends")
        ):
            errors.append(f"{path.relative_to(root)}: {error}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Public tyx import boundary passed: {len(files)} Python files checked.")


if __name__ == "__main__":
    main()
