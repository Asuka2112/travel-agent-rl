from __future__ import annotations


def bidirectional_pairs(left: str, right: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if left == right:
        raise ValueError("pairwise candidates must be distinct")
    return ((left, right), (right, left))

