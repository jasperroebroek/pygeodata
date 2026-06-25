from dataclasses import dataclass
from enum import StrEnum

from pygeodata.catalog.types import EntryInfo, ParamRow


class FilterTarget(StrEnum):
    ALL = 'all'
    CLASS = 'class'
    CRS = 'crs'
    KEY_GROUP = 'key_group'
    KEY = 'key'
    VALUE = 'value'
    PATH = 'path'
    HAS_WARNINGS = 'has_warnings'
    HAS_ERROR = 'has_error'


class FilterOperator(StrEnum):
    CONTAINS = 'contains'
    EQUALS = 'equals'
    STARTS = 'starts'
    NOT_CONTAINS = 'not_contains'


@dataclass(frozen=True, slots=True)
class Filter:
    target: FilterTarget = FilterTarget.ALL
    operator: FilterOperator = FilterOperator.CONTAINS
    value: str = ''

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> 'Filter':
        return cls(
            target=FilterTarget(data.get('target', FilterTarget.ALL.value)),
            operator=FilterOperator(data.get('operator', FilterOperator.CONTAINS.value)),
            value=data.get('value', ''),
        )


def parse_filters(items: list[dict[str, str]] | None) -> list[Filter]:
    if not items:
        return []
    return [Filter.from_dict(item) for item in items]


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


def entry_header_matches(class_name: str, entry: EntryInfo, flt: Filter) -> bool:
    if flt.target is FilterTarget.CLASS:
        return _compare(class_name, flt.operator, flt.value)
    if flt.target is FilterTarget.CRS:
        return _compare(entry.spec.crs, flt.operator, flt.value)
    if flt.target is FilterTarget.HAS_WARNINGS:
        return bool(entry.warnings)
    if flt.target is FilterTarget.HAS_ERROR:
        return bool(entry.error)
    # ALL and row-level targets: check class name then fall through to rows
    if _compare(class_name, flt.operator, flt.value):
        return True
    return any(row_matches_filter(row, flt) for row in entry.rows)


def entry_matches_filters(
    class_name: str,
    entry: EntryInfo,
    filters: list[Filter],
    logic_mode: str,
) -> bool:
    if not filters:
        return True

    matches = [entry_header_matches(class_name, entry, flt) for flt in filters]

    if logic_mode == 'OR':
        return any(matches)
    if logic_mode == 'NOT':
        return not any(matches)
    return all(matches)


def prescan_text(
    class_name: str,
    params_blob: str,
    parsed_exprs: list[tuple[list['Filter'], str]],
) -> bool:
    """Cheap pre-filter against raw text before paying for full enrichment.

    Returns False only when the entry can be definitively ruled out.
    Returns True when uncertain (false positives allowed — enrichment confirms).
    ``params_blob`` should already be lowercased by the caller.
    """
    for filters, logic_mode in parsed_exprs:
        term_results = []
        for flt in filters:
            val = flt.value.lower()
            if flt.target is FilterTarget.CLASS:
                text = class_name.lower()
            elif flt.target is FilterTarget.CRS:
                term_results.append(True)
                continue
            else:
                text = params_blob

            if flt.operator is FilterOperator.CONTAINS:
                hit = val in text
            elif flt.operator is FilterOperator.NOT_CONTAINS:
                hit = val not in text
            elif flt.operator is FilterOperator.EQUALS:
                hit = val in text  # conservative: may over-include
            elif flt.operator is FilterOperator.STARTS:
                hit = val in text  # conservative
            else:
                hit = True
            term_results.append(hit)

        if logic_mode == 'AND' and not all(term_results):
            return False
        if logic_mode == 'NOT' and any(term_results):
            return False
        if logic_mode == 'OR' and not any(term_results):
            return False

    return True


def matching_rows(
    entry: EntryInfo,
    filters: list[Filter],
    logic_mode: str,
) -> list[ParamRow]:
    row_targets = {
        FilterTarget.KEY_GROUP,
        FilterTarget.KEY,
        FilterTarget.VALUE,
        FilterTarget.PATH,
        FilterTarget.ALL,
    }
    row_filters = [flt for flt in filters if flt.target in row_targets]

    if not row_filters:
        return entry.rows

    rows: list[ParamRow] = []
    for row in entry.rows:
        matches = [row_matches_filter(row, flt) for flt in row_filters]

        if logic_mode == 'AND':
            ok = all(matches)
        elif logic_mode == 'NOT':
            ok = not any(matches)
        else:
            ok = any(matches)

        if ok:
            rows.append(row)

    return rows
