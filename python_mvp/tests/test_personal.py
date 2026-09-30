import numpy as np
import pytest


def rows():
    return [{'id': str(i), 'features': [float(i%2), float(i%3)], 'label': i%2,
             'sequence': i, 'exposure_id': str(i//5), 'base_score':0.,'personal_scale':1.} for i in range(40)]


def test_temporal_split_keeps_whole_exposures_and_no_track_overlap():
    from python_mvp.personal import temporal_split
    train, validation = temporal_split(rows())
    assert max(r['sequence'] for r in train) < min(r['sequence'] for r in validation)
    assert not {r['exposure_id'] for r in train} & {r['exposure_id'] for r in validation}
    assert not {r['id'] for r in train} & {r['id'] for r in validation}
    with pytest.raises(ValueError):
        temporal_split(rows()[:5])


def test_personal_fit_learns_only_train_and_reports_later_validation():
    from python_mvp.personal import fit_rows, predict
    model = fit_rows(rows())
    assert model['report']['validation_loss'] < model['report']['baseline_loss']
    assert model['report']['train_count'] == 30
    assert model['report']['validation_count'] == 8
    values = predict(model, np.array([[0., 1.], [1., 1.]]))
    assert values[1] > values[0]
    assert set(model['training_labels']) == {str(i) for i in range(30)}


def test_active_model_is_invalidated_by_undo_and_rollback_preserves_ratings(tmp_path):
    from python_mvp.feedback import record, undo, summary
    from python_mvp.personal import train, load_active, rollback
    for i in range(40):
        result = {'checkpoint_sha256':'base', 'exposure_id':str(i//5), 'top50':[
            {'rank':1,'id':str(i),'artist':'artist'+str(i), 'title':'song',
             'personal_features':[float(i%2),float(i%3)], 'base_score':0., 'personal_scale':1.}]}
        record(tmp_path, result, 'like' if i%2 else 'dislike', '1')
    model = train(tmp_path, 'base')
    assert model['report']['accepted']
    assert load_active(tmp_path, 'base')[0] is not None
    assert load_active(tmp_path, 'other')[0] is None
    original = summary(tmp_path)
    rollback(tmp_path)
    assert load_active(tmp_path, 'base')[0] is None
    assert summary(tmp_path) == original
    train(tmp_path, 'base')
    # Correct an actual training label, not a held-out validation label.
    result = {'checkpoint_sha256':'base', 'exposure_id':'correction', 'top50':[
        {'rank':1,'id':'0','artist':'artist0','title':'song','personal_features':[0.,0.]}]}
    record(tmp_path, result, 'like', '1')
    assert load_active(tmp_path, 'base')[0] is None
    undo(tmp_path)
    assert load_active(tmp_path, 'base')[0] is not None


def test_late_revision_cannot_leak_future_training_labels():
    from python_mvp.personal import temporal_split
    data = rows()
    data[0]['sequence'] = 100
    train, validation = temporal_split(data)
    assert max(r['sequence'] for r in train) < min(r['sequence'] for r in validation)
