"""Tests for pygeodata.registry_browser.payloads."""
import pytest

from pygeodata.config import JSONKeys
from pygeodata.registry_browser.models import (
    ClassInfo,
    EntryInfo,
    FileRef,
    GroupInfo,
    LinkedEntry,
    ParamRow,
    SpecInfo,
)
from pygeodata.registry_browser.payloads import (
    _build_class_cards,
    _build_detail_payload,
    _build_table_rows,
    _build_visible_groups,
    _entry_is_visible,
    _entry_matches_spec_filters,
    _sidebar_counts,
    build_browser_payload,
)
from pygeodata.registry_browser.state import AppState
from pygeodata.types import SpecKeys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spec(crs='EPSG:4326', resolution='0.1°', shape='180x360', bounds_latlon=(-90, -180, 90, 180)):
    return SpecInfo(crs=crs, resolution=resolution, shape=shape, bounds=str(bounds_latlon), bounds_latlon=bounds_latlon)


def make_row(key='year', value='2020', group='', path='year'):
    return ParamRow(
        path=path, key_group=group, final_key=key, value_text=value,
        value_type='int', search_blob=f'{group} {key} {value}'.lower(), depth=0,
    )


def make_entry(
    record_id='rec1',
    class_name='MyLoader',
    object_type='data',
    rows=None,
    warnings=None,
    error=None,
    spec=None,
    state_hash='abc',
    primary_file=None,
    co_outputs=None,
    linked_entries=None,
    dep_hash_stale=False,
):
    return EntryInfo(
        record_id=record_id,
        class_name=class_name,
        object_type=object_type,
        params_path='/cache/MyLoader/.file.params.json',
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash=state_hash,
        instance_hash=None,
        params={},
        spec=spec or make_spec(),
        rows=rows or [],
        warnings=warnings or [],
        error=error,
        primary_file=primary_file,
        co_outputs=co_outputs or [],
        linked_entries=linked_entries or [],
        dep_hash_stale=dep_hash_stale,
    )


def make_class_info(class_name='MyLoader', object_type='data', loaded=True, source_stale=False, deps_stale=False):
    return ClassInfo(
        class_name=class_name,
        object_type=object_type,
        loaded=loaded,
        call_dependency_names=[],
        inheritance_dependency_names=[],
        source_stale=source_stale,
        deps_stale=deps_stale,
    )


def make_state(entries=None, classes=None, groups=None):
    entries = entries or {}
    classes = classes or {}
    groups = groups or {}
    return AppState(
        entries=entries,
        classes=classes,
        groups=groups,
        diagnostics={},
        spec_options={},
    )


def simple_state():
    entry = make_entry(record_id='rec1', class_name='MyLoader')
    group = GroupInfo(class_name='MyLoader', object_type='data', record_ids=['rec1'])
    cls = make_class_info('MyLoader')
    return make_state(
        entries={'rec1': entry},
        classes={'MyLoader': cls},
        groups={'MyLoader': group},
    )


# ---------------------------------------------------------------------------
# _entry_matches_spec_filters
# ---------------------------------------------------------------------------


def test_spec_filter_empty_passes_all():
    entry = make_entry(spec=make_spec(crs='EPSG:4326'))
    assert _entry_matches_spec_filters(entry, {})


def test_spec_filter_crs_exact_match():
    entry = make_entry(spec=make_spec(crs='EPSG:4326'))
    assert _entry_matches_spec_filters(entry, {SpecKeys.CRS: 'EPSG:4326'})
    assert not _entry_matches_spec_filters(entry, {SpecKeys.CRS: 'EPSG:3857'})


def test_spec_filter_crs_list_match():
    entry = make_entry(spec=make_spec(crs='EPSG:4326'))
    assert _entry_matches_spec_filters(entry, {SpecKeys.CRS: ['EPSG:4326', 'EPSG:3857']})
    assert not _entry_matches_spec_filters(entry, {SpecKeys.CRS: ['EPSG:3857']})


def test_spec_filter_crs_empty_list_passes():
    entry = make_entry(spec=make_spec(crs='EPSG:4326'))
    assert _entry_matches_spec_filters(entry, {SpecKeys.CRS: []})


def test_spec_filter_resolution():
    entry = make_entry(spec=make_spec(resolution='0.1°'))
    assert _entry_matches_spec_filters(entry, {SpecKeys.RESOLUTION: '0.1°'})
    assert not _entry_matches_spec_filters(entry, {SpecKeys.RESOLUTION: '0.5°'})


def test_spec_filter_shape():
    entry = make_entry(spec=make_spec(shape='1800x3600'))
    assert _entry_matches_spec_filters(entry, {SpecKeys.SHAPE: '1800x3600'})
    assert not _entry_matches_spec_filters(entry, {SpecKeys.SHAPE: '900x1800'})


def test_spec_filter_bounds_latlon_string_match():
    entry = make_entry(spec=make_spec(bounds_latlon=(-90, -180, 90, 180)))
    expected = str(list((-90, -180, 90, 180)))
    assert _entry_matches_spec_filters(entry, {SpecKeys.BOUNDS: expected})


def test_spec_filter_bounds_latlon_no_bounds_latlon():
    entry = make_entry(spec=SpecInfo(bounds_latlon=None))
    assert not _entry_matches_spec_filters(entry, {SpecKeys.BOUNDS: 'anything'})


def test_spec_filter_bounds_latlon_list():
    entry = make_entry(spec=make_spec(bounds_latlon=(-90, -180, 90, 180)))
    key = str(list((-90, -180, 90, 180)))
    assert _entry_matches_spec_filters(entry, {SpecKeys.BOUNDS: [key]})
    assert not _entry_matches_spec_filters(entry, {SpecKeys.BOUNDS: ['other']})


# ---------------------------------------------------------------------------
# _entry_is_visible
# ---------------------------------------------------------------------------


def _visible(entry, **kwargs):
    defaults = dict(
        class_name=entry.class_name,
        entry=entry,
        selected_classes=[],
        kind_filter='all',
        spec_filters={},
        filters=[],
        logic_mode='AND',
    )
    defaults.update(kwargs)
    return _entry_is_visible(**defaults)


def test_entry_visible_no_filters():
    assert _visible(make_entry())


def test_entry_not_visible_wrong_class():
    entry = make_entry(class_name='MyLoader')
    assert not _visible(entry, selected_classes=['OtherLoader'])


def test_entry_visible_correct_selected_class():
    entry = make_entry(class_name='MyLoader')
    assert _visible(entry, selected_classes=['MyLoader'])


def test_entry_not_visible_wrong_kind():
    entry = make_entry(object_type='data')
    assert not _visible(entry, kind_filter='figure')


def test_entry_visible_selected_class_overrides_kind_filter():
    # selected class bypasses kind filter
    entry = make_entry(class_name='MyLoader', object_type='data')
    assert _visible(entry, selected_classes=['MyLoader'], kind_filter='figure')


def test_entry_not_visible_spec_mismatch():
    entry = make_entry(spec=make_spec(crs='EPSG:4326'))
    assert not _visible(entry, spec_filters={SpecKeys.CRS: 'EPSG:3857'})


# ---------------------------------------------------------------------------
# _sidebar_counts
# ---------------------------------------------------------------------------


def test_sidebar_counts_all():
    state = simple_state()
    counts = _sidebar_counts(state, kind_filter='all', spec_filters={}, filters=[], logic_mode='AND')
    assert counts == {'MyLoader': 1}


def test_sidebar_counts_kind_filter_excludes():
    state = simple_state()
    counts = _sidebar_counts(state, kind_filter='figure', spec_filters={}, filters=[], logic_mode='AND')
    assert counts == {}


def test_sidebar_counts_zero_not_included():
    entry = make_entry(spec=make_spec(crs='EPSG:4326'))
    group = GroupInfo(class_name='MyLoader', object_type='data', record_ids=['rec1'])
    cls = make_class_info('MyLoader')
    state = make_state(entries={'rec1': entry}, classes={'MyLoader': cls}, groups={'MyLoader': group})
    counts = _sidebar_counts(state, kind_filter='all', spec_filters={SpecKeys.CRS: 'EPSG:3857'}, filters=[], logic_mode='AND')
    assert 'MyLoader' not in counts


def test_sidebar_counts_multiple_entries():
    e1 = make_entry(record_id='r1', class_name='MyLoader')
    e2 = make_entry(record_id='r2', class_name='MyLoader')
    group = GroupInfo(class_name='MyLoader', object_type='data', record_ids=['r1', 'r2'])
    cls = make_class_info('MyLoader')
    state = make_state(entries={'r1': e1, 'r2': e2}, classes={'MyLoader': cls}, groups={'MyLoader': group})
    counts = _sidebar_counts(state, kind_filter='all', spec_filters={}, filters=[], logic_mode='AND')
    assert counts['MyLoader'] == 2


# ---------------------------------------------------------------------------
# _build_visible_groups
# ---------------------------------------------------------------------------


def test_build_visible_groups_returns_entry():
    state = simple_state()
    groups = _build_visible_groups(state, selected_classes=[], kind_filter='all',
                                   spec_filters={}, filters=[], logic_mode='AND')
    assert len(groups) == 1
    class_name, entries = groups[0]
    assert class_name == 'MyLoader'
    assert len(entries) == 1


def test_build_visible_groups_kind_filter():
    state = simple_state()
    groups = _build_visible_groups(state, selected_classes=[], kind_filter='figure',
                                   spec_filters={}, filters=[], logic_mode='AND')
    assert groups == []


def test_build_visible_groups_sorted_by_class_name():
    e1 = make_entry(record_id='r1', class_name='ZLoader')
    e2 = make_entry(record_id='r2', class_name='ALoader')
    state = make_state(
        entries={'r1': e1, 'r2': e2},
        classes={'ZLoader': make_class_info('ZLoader'), 'ALoader': make_class_info('ALoader')},
        groups={
            'ZLoader': GroupInfo(class_name='ZLoader', object_type='data', record_ids=['r1']),
            'ALoader': GroupInfo(class_name='ALoader', object_type='data', record_ids=['r2']),
        },
    )
    groups = _build_visible_groups(state, selected_classes=[], kind_filter='all',
                                   spec_filters={}, filters=[], logic_mode='AND')
    assert groups[0][0] == 'ALoader'
    assert groups[1][0] == 'ZLoader'


def test_build_visible_groups_class_with_no_group():
    # Class in state.classes but not in groups — should produce no entries
    cls = make_class_info('GhostLoader')
    state = make_state(classes={'GhostLoader': cls})
    groups = _build_visible_groups(state, selected_classes=[], kind_filter='all',
                                   spec_filters={}, filters=[], logic_mode='AND')
    assert groups == []


# ---------------------------------------------------------------------------
# _build_class_cards
# ---------------------------------------------------------------------------


def test_build_class_cards_fields():
    state = simple_state()
    cards = _build_class_cards(state, sidebar_counts={'MyLoader': 1}, selected_classes=['MyLoader'])
    assert len(cards) == 1
    card = cards[0]
    assert card['class_name'] == 'MyLoader'
    assert card['selected'] is True
    assert card['total_record_count'] == 1
    assert card['visible_record_count'] == 1
    assert card['loaded'] is True


def test_build_class_cards_not_selected():
    state = simple_state()
    cards = _build_class_cards(state, sidebar_counts={}, selected_classes=[])
    assert cards[0]['selected'] is False
    assert cards[0]['visible_record_count'] == 0


def test_build_class_cards_sorted():
    state = make_state(
        classes={
            'ZLoader': make_class_info('ZLoader'),
            'ALoader': make_class_info('ALoader'),
        },
    )
    cards = _build_class_cards(state, sidebar_counts={}, selected_classes=[])
    assert cards[0]['class_name'] == 'ALoader'
    assert cards[1]['class_name'] == 'ZLoader'


def test_build_class_cards_stale_flags():
    cls = make_class_info('MyLoader', source_stale=True, deps_stale=True)
    state = make_state(classes={'MyLoader': cls})
    cards = _build_class_cards(state, sidebar_counts={}, selected_classes=[])
    assert cards[0]['source_stale'] is True
    assert cards[0]['deps_stale'] is True


# ---------------------------------------------------------------------------
# _build_table_rows
# ---------------------------------------------------------------------------


def test_build_table_rows_header_only():
    entry = make_entry(record_id='rec1', class_name='MyLoader', rows=[make_row()])
    groups = [('MyLoader', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={'MyLoader': make_class_info('MyLoader')},
        selected_entry=None, filters=[], logic_mode='AND', row_display='none',
    )
    assert len(rows) == 1
    assert rows[0]['row_type'] == 'header'
    assert rows[0]['record_id'] == 'rec1'


def test_build_table_rows_row_display_all():
    entry = make_entry(rows=[make_row('year', '2020'), make_row('region', 'eu')])
    groups = [('MyLoader', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={'MyLoader': make_class_info('MyLoader')},
        selected_entry=None, filters=[], logic_mode='AND', row_display='all',
    )
    types = [r['row_type'] for r in rows]
    assert types.count('header') == 1
    assert types.count('detail') == 2


def test_build_table_rows_row_display_selected():
    from pygeodata.registry_browser.filters import Filter, FilterTarget
    entry = make_entry(rows=[make_row('year', '2020'), make_row('region', 'eu')])
    groups = [('MyLoader', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={'MyLoader': make_class_info('MyLoader')},
        selected_entry=None,
        filters=[Filter(target=FilterTarget.KEY, value='year')],
        logic_mode='AND', row_display='selected',
    )
    detail_rows = [r for r in rows if r['row_type'] == 'detail']
    assert len(detail_rows) == 1
    assert detail_rows[0]['parameter'] == 'year'


def test_build_table_rows_focused_entry():
    entry = make_entry(record_id='rec1')
    groups = [('MyLoader', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={'MyLoader': make_class_info('MyLoader')},
        selected_entry='rec1', filters=[], logic_mode='AND', row_display='none',
    )
    assert rows[0]['focused'] is True


def test_build_table_rows_unfocused():
    entry = make_entry(record_id='rec1')
    groups = [('MyLoader', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={'MyLoader': make_class_info('MyLoader')},
        selected_entry='other', filters=[], logic_mode='AND', row_display='none',
    )
    assert rows[0]['focused'] is False


def test_build_table_rows_dep_hash_stale():
    entry = make_entry(record_id='rec1', dep_hash_stale=True)
    groups = [('MyLoader', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={'MyLoader': make_class_info('MyLoader')},
        selected_entry=None, filters=[], logic_mode='AND', row_display='none',
    )
    assert rows[0]['dep_hash_stale'] is True


def test_build_table_rows_missing_class_info():
    entry = make_entry(record_id='rec1', class_name='Ghost')
    groups = [('Ghost', [entry])]
    rows = _build_table_rows(
        visible_groups=groups, classes={},
        selected_entry=None, filters=[], logic_mode='AND', row_display='none',
    )
    assert rows[0]['source_stale'] is False


# ---------------------------------------------------------------------------
# _build_detail_payload
# ---------------------------------------------------------------------------


def test_build_detail_payload_selected_entry():
    entry = make_entry(record_id='rec1', class_name='MyLoader')
    cls = make_class_info('MyLoader')
    state = make_state(entries={'rec1': entry}, classes={'MyLoader': cls})
    detail = _build_detail_payload(state=state, selected_entry_info=entry, selected_classes=[])
    assert detail is not None
    assert detail['class_name'] == 'MyLoader'
    assert detail['selected_entry']['record_id'] == 'rec1'


def test_build_detail_payload_entry_class_not_in_state():
    entry = make_entry(class_name='GhostLoader')
    state = make_state()
    detail = _build_detail_payload(state=state, selected_entry_info=entry, selected_classes=[])
    assert detail is None


def test_build_detail_payload_single_selected_class():
    cls = make_class_info('MyLoader')
    state = make_state(classes={'MyLoader': cls})
    detail = _build_detail_payload(state=state, selected_entry_info=None, selected_classes=['MyLoader'])
    assert detail is not None
    assert detail['class_name'] == 'MyLoader'
    assert detail['selected_entry'] is None


def test_build_detail_payload_multiple_selected_classes():
    state = make_state(classes={'A': make_class_info('A'), 'B': make_class_info('B')})
    detail = _build_detail_payload(state=state, selected_entry_info=None, selected_classes=['A', 'B'])
    assert detail is None


def test_build_detail_payload_no_selection():
    state = make_state()
    detail = _build_detail_payload(state=state, selected_entry_info=None, selected_classes=[])
    assert detail is None


def test_build_detail_payload_figure_preview():
    file_ref = FileRef(label='fig.png', path='/data/fig.png', kind='image')
    entry = make_entry(record_id='rec1', class_name='MyLoader', primary_file=file_ref)
    cls = make_class_info('MyLoader')
    state = make_state(entries={'rec1': entry}, classes={'MyLoader': cls})
    detail = _build_detail_payload(state=state, selected_entry_info=entry, selected_classes=[])
    assert detail['selected_entry']['figure_previews'] == ['/data/fig.png']


def test_build_detail_payload_no_preview_for_non_image():
    file_ref = FileRef(label='data.tif', path='/data/data.tif', kind='raster')
    entry = make_entry(record_id='rec1', class_name='MyLoader', primary_file=file_ref)
    cls = make_class_info('MyLoader')
    state = make_state(entries={'rec1': entry}, classes={'MyLoader': cls})
    detail = _build_detail_payload(state=state, selected_entry_info=entry, selected_classes=[])
    assert detail['selected_entry']['figure_previews'] == []


def test_build_detail_payload_linked_entries():
    le = LinkedEntry(param_name='base', class_name='BaseLoader', state_hash='x', params_summary={'year': '2020'})
    entry = make_entry(record_id='rec1', class_name='MyLoader', linked_entries=[le])
    cls = make_class_info('MyLoader')
    state = make_state(entries={'rec1': entry}, classes={'MyLoader': cls})
    detail = _build_detail_payload(state=state, selected_entry_info=entry, selected_classes=[])
    linked = detail['selected_entry']['linked_entries']
    assert len(linked) == 1
    assert linked[0]['param_name'] == 'base'
    assert linked[0][JSONKeys.CLASS_NAME] == 'BaseLoader'


# ---------------------------------------------------------------------------
# build_browser_payload (top-level integration)
# ---------------------------------------------------------------------------


def test_build_browser_payload_structure():
    state = simple_state()
    payload = build_browser_payload(
        state,
        selected_classes=[],
        selected_entry=None,
        kind_filter='all',
        spec_filters={},
        filters=[],
        logic_mode='AND',
        row_display='none',
    )
    assert 'class_cards' in payload
    assert 'table_rows' in payload
    assert 'detail' in payload
    assert 'counts' in payload
    assert 'visible_entry_ids' in payload
    assert 'diagnostics' in payload
    assert 'spec_options' in payload


def test_build_browser_payload_counts():
    state = simple_state()
    payload = build_browser_payload(
        state,
        selected_classes=[], selected_entry=None, kind_filter='all',
        spec_filters={}, filters=[], logic_mode='AND', row_display='none',
    )
    counts = payload['counts']
    assert counts['classes'] == 1
    assert counts['entries'] == 1
    assert counts['visible_classes'] == 1
    assert counts['visible_entries'] == 1


def test_build_browser_payload_invalid_selected_entry_cleared():
    state = simple_state()
    payload = build_browser_payload(
        state,
        selected_classes=[], selected_entry='nonexistent', kind_filter='all',
        spec_filters={}, filters=[], logic_mode='AND', row_display='none',
    )
    assert payload['selected_entry'] is None


def test_build_browser_payload_filters_parsed_from_dicts():
    state = simple_state()
    payload = build_browser_payload(
        state,
        selected_classes=[], selected_entry=None, kind_filter='all',
        spec_filters={},
        filters=[{'target': 'class', 'operator': 'equals', 'value': 'OtherLoader'}],
        logic_mode='AND',
        row_display='none',
    )
    assert payload['counts']['visible_entries'] == 0


def test_build_browser_payload_visible_entry_ids():
    state = simple_state()
    payload = build_browser_payload(
        state,
        selected_classes=[], selected_entry=None, kind_filter='all',
        spec_filters={}, filters=[], logic_mode='AND', row_display='none',
    )
    assert payload['visible_entry_ids'] == ['rec1']
