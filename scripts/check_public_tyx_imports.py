"""Check the declared public TYX adapter imports (not a Python sandbox)."""

import ast
from pathlib import Path

# These are existing public interop consumers, not permission for arbitrary
# training modules or TYX internal submodules.
TORCH_ADAPTER_IMPORTS = {
    "backends/mujoco_task_workspace.py": frozenset({"to_torch"}),
    "training/mujoco_tyx_env.py": frozenset({"from_torch", "stream_from_torch"}),
}
ROOT_ADAPTER_IMPORTS = {
    "training/policy_response_diagnostics.py": frozenset(
        {"Runtime", "CudaStream", "use_runtime", "CudaArray"}
    ),
}


def violations(source: str, *, in_adapter: bool, relative_path: str = "") -> list[str]:
    tree = ast.parse(source)
    errors, aliases, root_aliases, restricted_roots = [], set(), set(), set()
    path = relative_path.replace("\\", "/")
    torch_names = TORCH_ADAPTER_IMPORTS.get(path, frozenset())
    root_names = ROOT_ADAPTER_IMPORTS.get(path, frozenset())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "tyx" or item.name.startswith("tyx."):
                    if not in_adapter and not (item.name == "tyx" and root_names):
                        errors.append(f"line {node.lineno}: tyx import outside declared adapters")
                    if item.name != "tyx":
                        errors.append(f"line {node.lineno}: import from the public tyx root only")
                    alias = item.asname or item.name.split(".")[0]
                    aliases.add(alias)
                    if item.name == "tyx":
                        root_aliases.add(alias)
                    if not in_adapter and root_names and item.name == "tyx":
                        restricted_roots.add(alias)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tyx" or node.module.startswith("tyx."):
                named = {item.name for item in node.names}
                public_names = all(not name.startswith("_") and name != "*" for name in named)
                root_ok = (
                    in_adapter and node.module == "tyx" and public_names and "torch" not in named
                )
                torch_ok = (
                    node.module == "tyx.torch"
                    and public_names
                    and bool(torch_names)
                    and named <= torch_names
                )
                if not (root_ok or torch_ok):
                    errors.append(
                        f"line {node.lineno}: import named public tyx adapter exports only"
                    )
                # ImportFrom aliases must not bypass the private-attribute check.
                aliases.update(item.asname or item.name for item in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node
            chain = []
            while isinstance(root, ast.Attribute):
                chain.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name) and root.id in aliases:
                if any(name.startswith("_") for name in chain):
                    errors.append(f"line {node.lineno}: private tyx attribute access")
                if root.id in root_aliases and chain[-1] == "torch":
                    errors.append(f"line {node.lineno}: use declared named tyx.torch imports")
                if root.id in restricted_roots and chain[-1] not in root_names:
                    errors.append(f"line {node.lineno}: undeclared training tyx root export")
    return errors


def main():
    root = Path(__file__).resolve().parents[1] / "src" / "phymeo"
    errors = []
    files = list(root.rglob("*.py"))
    for path in files:
        for error in violations(
            path.read_text(encoding="utf-8"),
            in_adapter=path.is_relative_to(root / "backends"),
            relative_path=path.relative_to(root).as_posix(),
        ):
            errors.append(f"{path.relative_to(root)}: {error}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Public tyx import boundary passed: {len(files)} Python files checked.")


if __name__ == "__main__":
    main()
