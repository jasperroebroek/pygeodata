from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygeodata.catalog.types import CatalogEntry, ParamRow
    from pygeodata.records import SpecRecord


class FilterTarget(StrEnum):
    ALL = 'all'
    CLASS = 'class'
    CRS = 'crs'
    KEY_GROUP = 'key_group'
    KEY = 'key'
    VALUE = 'value'
    PATH = 'path'


ROW_TARGETS = frozenset(
    {FilterTarget.ALL, FilterTarget.KEY_GROUP, FilterTarget.KEY, FilterTarget.VALUE, FilterTarget.PATH},
)


class FilterOperator(StrEnum):
    CONTAINS = 'contains'
    EQUALS = 'equals'
    STARTS = 'starts'
    NOT_CONTAINS = 'not_contains'


class LogicMode(StrEnum):
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'


@dataclass(frozen=True, slots=True)
class Filter:
    target: FilterTarget = FilterTarget.ALL
    operator: FilterOperator = FilterOperator.CONTAINS
    value: str = ''

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Filter:
        return cls(
            target=FilterTarget(data.get('target', FilterTarget.ALL.value)),
            operator=FilterOperator(data.get('operator', FilterOperator.CONTAINS.value)),
            value=data.get('value', ''),
        )


def parse_filters(items: list[dict[str, str]] | None) -> list[Filter]:
    if not items:
        return []
    return [Filter.from_dict(item) for item in items]


def combine(matches: list[bool], mode: LogicMode) -> bool:
    if mode is LogicMode.OR:
        return any(matches)
    if mode is LogicMode.NOT:
        return not any(matches)
    return all(matches)


def _compare(text: str | None, operator: FilterOperator, value: str) -> bool:
    left = (text or '').lower()
    right = (value or '').lower()

    if operator is FilterOperator.EQUALS:
        return left == right
    if operator is FilterOperator.STARTS:
        return left.startswith(right)
    if operator is FilterOperator.NOT_CONTAINS:
        return right not in left
    return right in left  # CONTAINS


def row_matches_filter(row: ParamRow, flt: Filter) -> bool:
    if flt.target is FilterTarget.KEY_GROUP:
        return _compare(row.key_group, flt.operator, flt.value)
    if flt.target is FilterTarget.KEY:
        return _compare(row.final_key, flt.operator, flt.value)
    if flt.target is FilterTarget.VALUE:
        return _compare(row.value_text, flt.operator, flt.value)
    if flt.target is FilterTarget.PATH:
        return _compare(row.path, flt.operator, flt.value)
    if flt.target is FilterTarget.ALL:
        return _compare(row.search_blob, flt.operator, flt.value)
    return False


def entry_matches_filter(entry: CatalogEntry, spec: SpecRecord | None, flt: Filter) -> bool:
    if flt.target is FilterTarget.CLASS:
        return _compare(entry.class_name, flt.operator, flt.value)
    if flt.target is FilterTarget.CRS:
        return _compare(spec.crs if spec is not None else None, flt.operator, flt.value)
    # ALL and row-level targets: check class name then fall through to rows
    if _compare(entry.class_name, flt.operator, flt.value):
        return True
    return any(row_matches_filter(row, flt) for row in entry.rows)


def entry_matches_filters(
    entry: CatalogEntry,
    spec: SpecRecord | None,
    filters: list[Filter],
    mode: LogicMode,
) -> bool:
    if not filters:
        return True
    return combine([entry_matches_filter(entry, spec, flt) for flt in filters], mode)


def matching_rows(entry: CatalogEntry, filters: list[Filter], mode: LogicMode) -> list[ParamRow]:
    row_filters = [flt for flt in filters if flt.target in ROW_TARGETS]
    if not row_filters:
        return entry.rows
    return [row for row in entry.rows if combine([row_matches_filter(row, flt) for flt in row_filters], mode)]
