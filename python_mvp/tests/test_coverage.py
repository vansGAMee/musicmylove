from types import SimpleNamespace


def test_coverage_distinguishes_ambiguity_filtered_and_absent():
    from python_mvp.coverage import diagnose
    tracks = [dict(id='mbid:1', artist='A', title='One'),
              dict(id='mbid:2', artist='B', title='Ambiguous'), dict(id='mbid:3', artist='B', title='Ambiguous')]
    raw = tracks+[dict(id='rare', artist='C', title='Rare')]
    engine = SimpleNamespace(tracks=tracks, meta={'seen':[0,1,2], 'train_users':['rep']},
                             data={'tracks':raw, 'users':[{'id':'rep','tracks':[0,3]}, {'id':'dev','tracks':[3]}]})
    report = diagnose(engine, ['A - One', 'B - Ambiguous', 'C - Rare', 'D - Missing'])
    assert report['counts'] == {'resolved': 1, 'ambiguous': 1, 'in_raw_not_catalog': 1, 'absent_from_raw': 1}
    row = next(r for r in report['tracks'] if r['input']=='C - Rare')
    assert row['representation_support'] == 1  # DEV must not inflate support
    assert report['expansion_targets'] == [{'artist':'C','tracks':1}, {'artist':'D','tracks':1}]
