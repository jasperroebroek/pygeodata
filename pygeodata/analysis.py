"""The project's tracked classes, derived from source or by importing it: one result type, one wire format.

:func:`analyse_in_subprocess` owns the protocol with the ``pygeodata analyse --import-all --json``
command, so a caller can analyse a project without importing its code into its own process.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import msgspec

from pygeodata.dependencies import ClassFacts, dependency_tree_hash

if TYPE_CHECKING:
    from pathlib import Path


class AnalysisError(Exception):
    """The analysing subprocess failed, timed out or returned something undecodable."""


@dataclass(frozen=True)
class ClassRecord:
    # lineno is where the class's source starts, at its first decorator; paths are strings, as in the catalog
    facts: ClassFacts
    path: str
    lineno: int
    tree_hash: str


@dataclass
class ProjectAnalysis:
    mode: Literal['static', 'import']
    root: str
    classes: dict[str, ClassRecord]
    # files that could not be parsed, duplicate class names; in import mode also failed imports and disagreements
    problems: list[str] = field(default_factory=list)

    @property
    def facts(self) -> dict[str, ClassFacts]:
        return {name: cls.facts for name, cls in self.classes.items()}

    @property
    def tree_hashes(self) -> dict[str, str]:
        return {name: cls.tree_hash for name, cls in self.classes.items()}


def tree_hashes_of(facts: dict[str, ClassFacts]) -> dict[str, str]:
    return {name: dependency_tree_hash(facts.__getitem__, name) for name in facts}


def encode_analysis(analysis: ProjectAnalysis) -> bytes:
    return msgspec.json.encode(analysis)


def decode_analysis(data: bytes | str) -> ProjectAnalysis:
    return msgspec.json.decode(data, type=ProjectAnalysis)


def analyse_in_subprocess(root: Path, *, timeout: float) -> ProjectAnalysis:
    """The project at ``root`` analysed by importing it in a fresh interpreter."""
    command = [sys.executable, '-m', 'pygeodata', 'analyse', '--import-all', '--json']
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout, check=False)  # noqa: S603 — fixed interpreter and arguments
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError(f'analysing {root} took longer than {timeout} s') from exc
    stderr = result.stderr.decode(errors='replace').strip()
    if result.returncode != 0:
        raise AnalysisError(f'analysing {root} exited with status {result.returncode}:\n{stderr}')
    try:
        return decode_analysis(result.stdout)
    except msgspec.DecodeError as exc:
        raise AnalysisError(f'analysing {root} returned undecodable output ({exc}):\n{stderr}') from exc
