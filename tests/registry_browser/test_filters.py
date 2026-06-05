import pytest

from pygeodata.registry_browser.filters import (
    Filter,
    FilterOperator,
    FilterTarget,
    entry_header_matches,
    entry_matches_filters,
    matching_rows,
    parse_filters,
    row_matches_filter,
)
from pygeodata.registry_browser.models import EntryInfo, ParamRow, SpecInfo


def make_row(
    key_group: str = 'params',
    final_key: str = 'year',
    value_text: str = '2020',
    path: str = 'params.year',
    search_blob: str = 'params year 2020',
) -> ParamRow:
    return ParamRow(
        path=path,
        key_group=key_group,
        final_key=final_key,
        value_text=value_text,
        value_type='int',
        search_blob=search_blob,
        depth=0,
    )


def make_entry(
    class_name: str = 'MyLoader',
    crs: str | None = 'EPSG:4326',
    rows: list[ParamRow] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> tuple[str, EntryInfo]:
    entry = EntryInfo(
        record_id='abc',
        class_name=class_name,
        object_type='Data',
        params_path='/data/params.json',
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash=None,
        instance_hash=None,
        params={},
        spec=SpecInfo(crs=crs),
        rows=rows or [],
        warnings=warnings or [],
        error=error,
    )
    return class_name, entry


# --- Filter.from_dict ---

def test_filter_from_dict_defaults() -> None:
    f = Filter.from_dict({})
    assert f.target == FilterTarget.ALL
    assert f.operator == FilterOperator.CONTAINS
    assert f.value == ''


def test_filter_from_dict_explicit_values() -> None:
    f = Filter.from_dict({'target': 'class', 'operator': 'equals', 'value': 'MyLoader'})
    assert f.target == FilterTarget.CLASS
    assert f.operator == FilterOperator.EQUALS
    assert f.value == 'MyLoader'


# --- parse_filters ---

def test_parse_filters_empty() -> None:
    assert parse_filters(None) == []
    assert parse_filters([]) == []


def test_parse_filters_multiple() -> None:
    items = [
        {'target': 'class', 'operator': 'equals', 'value': 'A'},
        {'target': 'crs', 'operator': 'contains', 'value': '4326'},
    ]
    result = parse_filters(items)
    assert len(result) == 2
    assert result[0].target == FilterTarget.CLASS
    assert result[1].target == FilterTarget.CRS


# --- _compare via row_matches_filter ---

@pytest.mark.parametrize('operator,value,expected', [
    (FilterOperator.CONTAINS, 'year', True),
    (FilterOperator.CONTAINS, 'month', False),
    (FilterOperator.EQUALS, 'year', True),
    (FilterOperator.EQUALS, 'YEAR', True),  # case-insensitive
    (FilterOperator.EQUALS, 'ye', False),
    (FilterOperator.STARTS, 'ye', True),
    (FilterOperator.STARTS, 'ar', False),
    (FilterOperator.NOT_CONTAINS, 'month', True),
    (FilterOperator.NOT_CONTAINS, 'year', False),
])
def test_compare_operators(operator, value, expected) -> None:
    row = make_row(final_key='year')
    flt = Filter(target=FilterTarget.KEY, operator=operator, value=value)
    assert row_matches_filter(row, flt) == expected


# --- row_matches_filter ---

def test_row_matches_key_group() -> None:
    row = make_row(key_group='inputs')
    assert row_matches_filter(row, Filter(target=FilterTarget.KEY_GROUP, value='inputs'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.KEY_GROUP, value='outputs'))


def test_row_matches_key() -> None:
    row = make_row(final_key='scenario')
    assert row_matches_filter(row, Filter(target=FilterTarget.KEY, value='scenario'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.KEY, value='region'))


def test_row_matches_value() -> None:
    row = make_row(value_text='ssp126')
    assert row_matches_filter(row, Filter(target=FilterTarget.VALUE, value='ssp'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.VALUE, value='rcp'))


def test_row_matches_path() -> None:
    row = make_row(path='inputs.scenario')
    assert row_matches_filter(row, Filter(target=FilterTarget.PATH, value='inputs'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.PATH, value='outputs'))


def test_row_matches_all_uses_search_blob() -> None:
    row = make_row(search_blob='inputs scenario ssp126')
    assert row_matches_filter(row, Filter(target=FilterTarget.ALL, value='ssp'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.ALL, value='rcp'))


def test_row_matches_non_row_target_returns_false() -> None:
    row = make_row()
    assert not row_matches_filter(row, Filter(target=FilterTarget.CLASS, value='anything'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.CRS, value='anything'))
    assert not row_matches_filter(row, Filter(target=FilterTarget.HAS_WARNINGS, value=''))
    assert not row_matches_filter(row, Filter(target=FilterTarget.HAS_ERROR, value=''))


# --- entry_header_matches ---

def test_entry_header_class_filter() -> None:
    cls, entry = make_entry(class_name='ClimateLoader')
    assert entry_header_matches(cls, entry, Filter(target=FilterTarget.CLASS, value='climate'))
    assert not entry_header_matches(cls, entry, Filter(target=FilterTarget.CLASS, value='ocean'))


def test_entry_header_crs_filter() -> None:
    cls, entry = make_entry(crs='EPSG:4326')
    assert entry_header_matches(cls, entry, Filter(target=FilterTarget.CRS, value='4326'))
    assert not entry_header_matches(cls, entry, Filter(target=FilterTarget.CRS, value='3857'))


def test_entry_header_has_warnings_true() -> None:
    cls, entry = make_entry(warnings=['dep hash stale'])
    assert entry_header_matches(cls, entry, Filter(target=FilterTarget.HAS_WARNINGS, value=''))


def test_entry_header_has_warnings_false() -> None:
    cls, entry = make_entry(warnings=[])
    assert not entry_header_matches(cls, entry, Filter(target=FilterTarget.HAS_WARNINGS, value=''))


def test_entry_header_has_error_true() -> None:
    cls, entry = make_entry(error='File missing')
    assert entry_header_matches(cls, entry, Filter(target=FilterTarget.HAS_ERROR, value=''))


def test_entry_header_has_error_false() -> None:
    cls, entry = make_entry(error=None)
    assert not entry_header_matches(cls, entry, Filter(target=FilterTarget.HAS_ERROR, value=''))


def test_entry_header_all_matches_class_name() -> None:
    cls, entry = make_entry(class_name='SoilLoader')
    assert entry_header_matches(cls, entry, Filter(target=FilterTarget.ALL, value='soil'))


def test_entry_header_all_falls_through_to_rows() -> None:
    row = make_row(search_blob='inputs scenario ssp126')
    cls, entry = make_entry(class_name='ClimateLoader', rows=[row])
    flt = Filter(target=FilterTarget.ALL, value='ssp126')
    assert entry_header_matches(cls, entry, flt)


def test_entry_header_all_no_match() -> None:
    cls, entry = make_entry(class_name='ClimateLoader', rows=[make_row(search_blob='year 2020')])
    flt = Filter(target=FilterTarget.ALL, value='ocean')
    assert not entry_header_matches(cls, entry, flt)


# --- entry_matches_filters ---

def test_entry_matches_no_filters() -> None:
    cls, entry = make_entry()
    assert entry_matches_filters(cls, entry, [], 'AND')


def test_entry_matches_and_all_pass() -> None:
    cls, entry = make_entry(class_name='ClimateLoader', crs='EPSG:4326')
    filters = [
        Filter(target=FilterTarget.CLASS, value='climate'),
        Filter(target=FilterTarget.CRS, value='4326'),
    ]
    assert entry_matches_filters(cls, entry, filters, 'AND')


def test_entry_matches_and_one_fails() -> None:
    cls, entry = make_entry(class_name='ClimateLoader', crs='EPSG:4326')
    filters = [
        Filter(target=FilterTarget.CLASS, value='climate'),
        Filter(target=FilterTarget.CRS, value='3857'),
    ]
    assert not entry_matches_filters(cls, entry, filters, 'AND')


def test_entry_matches_or_one_passes() -> None:
    cls, entry = make_entry(class_name='ClimateLoader', crs='EPSG:4326')
    filters = [
        Filter(target=FilterTarget.CLASS, value='ocean'),
        Filter(target=FilterTarget.CRS, value='4326'),
    ]
    assert entry_matches_filters(cls, entry, filters, 'OR')


def test_entry_matches_or_none_pass() -> None:
    cls, entry = make_entry(class_name='ClimateLoader', crs='EPSG:4326')
    filters = [
        Filter(target=FilterTarget.CLASS, value='ocean'),
        Filter(target=FilterTarget.CRS, value='3857'),
    ]
    assert not entry_matches_filters(cls, entry, filters, 'OR')


def test_entry_matches_not_none_match() -> None:
    cls, entry = make_entry(class_name='ClimateLoader')
    filters = [Filter(target=FilterTarget.CLASS, value='ocean')]
    assert entry_matches_filters(cls, entry, filters, 'NOT')


def test_entry_matches_not_any_match() -> None:
    cls, entry = make_entry(class_name='ClimateLoader')
    filters = [Filter(target=FilterTarget.CLASS, value='climate')]
    assert not entry_matches_filters(cls, entry, filters, 'NOT')


# --- matching_rows ---

def test_matching_rows_no_row_filters_returns_all() -> None:
    rows = [make_row(final_key='year'), make_row(final_key='region')]
    cls, entry = make_entry(rows=rows)
    # CRS filter is not a row-level target
    result = matching_rows(entry, [Filter(target=FilterTarget.CRS, value='4326')], 'AND')
    assert result == rows


def test_matching_rows_and_filters_all_must_match() -> None:
    rows = [
        make_row(final_key='year', value_text='2020', search_blob='year 2020'),
        make_row(final_key='region', value_text='europe', search_blob='region europe'),
    ]
    cls, entry = make_entry(rows=rows)
    filters = [
        Filter(target=FilterTarget.KEY, value='year'),
        Filter(target=FilterTarget.VALUE, value='2020'),
    ]
    result = matching_rows(entry, filters, 'AND')
    assert len(result) == 1
    assert result[0].final_key == 'year'


def test_matching_rows_or_any_matches() -> None:
    rows = [
        make_row(final_key='year', search_blob='year 2020'),
        make_row(final_key='region', search_blob='region europe'),
    ]
    cls, entry = make_entry(rows=rows)
    filters = [
        Filter(target=FilterTarget.KEY, value='year'),
        Filter(target=FilterTarget.KEY, value='region'),
    ]
    result = matching_rows(entry, filters, 'OR')
    assert len(result) == 2


def test_matching_rows_not_excludes_matches() -> None:
    rows = [
        make_row(final_key='year', search_blob='year 2020'),
        make_row(final_key='region', search_blob='region europe'),
    ]
    cls, entry = make_entry(rows=rows)
    filters = [Filter(target=FilterTarget.KEY, value='year')]
    result = matching_rows(entry, filters, 'NOT')
    assert len(result) == 1
    assert result[0].final_key == 'region'
