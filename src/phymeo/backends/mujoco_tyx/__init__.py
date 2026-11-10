"""Independent TYX port of explicitly admitted MuJoCo algorithms, not Warp wrappers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MuJoCoTyxIssue:
    code: str
    detail: str


class MuJoCoTyxUnsupported(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{row.code}: {row.detail}" for row in self.issues))
