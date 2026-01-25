"""
Validation utilities for suspension geometry.

Provides comprehensive validation with helpful error messages including Levenshtein-
based suggestions for typos.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationError:
    """
    A validation error with optional suggestion for correction.

    Attributes:
        message: The error message describing what's wrong.
        suggestion: Optional suggestion for how to fix the error.
    """

    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message} ({self.suggestion})"
        return self.message


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions,
        substitutions) required to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise.
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_closest_matches(
    key: str,
    valid_keys: set[str],
    max_results: int = 3,
    max_distance: int = 3,
) -> list[str]:
    """
    Find the closest matching keys using Levenshtein distance.

    Args:
        key: The key to find matches for.
        valid_keys: Set of valid keys to match against.
        max_results: Maximum number of suggestions to return.
        max_distance: Maximum edit distance to consider a match.

    Returns:
        List of closest matching keys, sorted by distance then alphabetically.
    """
    distances: list[tuple[int, str]] = []

    for valid_key in valid_keys:
        distance = levenshtein_distance(key.upper(), valid_key.upper())
        if distance <= max_distance:
            distances.append((distance, valid_key))

    # Sort by distance, then alphabetically.
    distances.sort(key=lambda x: (x[0], x[1]))

    return [key for _, key in distances[:max_results]]


def format_validation_errors(errors: list[ValidationError]) -> str:
    """
    Format a list of validation errors into a human-readable string.

    Args:
        errors: List of ValidationError objects.

    Returns:
        Formatted error message string.
    """
    if not errors:
        return ""

    lines = ["Validation failed:"]
    for error in errors:
        lines.append(f"  - {error}")

    return "\n".join(lines)


__all__ = [
    "ValidationError",
    "find_closest_matches",
    "format_validation_errors",
    "levenshtein_distance",
]
