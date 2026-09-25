import numpy as np


def test_baselines_and_network_use_identical_candidates_and_playlist_rules():
    from python_mvp.audit_discovery import audit_query
    class Engine:
        tracks = [{'id': str(i), 'artist': f'A{i}', 'title': f'T{i}'} for i in range(4)]
        def score(self, model, features):
            return np.array([0., 4., 3., 100.])
    signal = np.array([0., 4., 3., 100.])
    result = audit_query(Engine(), None, {'user': 'u', 'seeds': [0], 'targets': {1}},
                         None, {1, 2}, {'ppr': signal, 'listeners': signal, 'graph': signal}, '1')
    for name in ('neural', 'listeners', 'ppr', 'graph'):
        assert result[name]['rank_ndcg'] == 1.
        assert result[name]['playlist_ndcg'] == 1.
        assert result[name]['size'] == 2
