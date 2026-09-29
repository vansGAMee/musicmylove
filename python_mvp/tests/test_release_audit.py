import numpy as np
from python_mvp.release_audit import less_familiar,values


def test_probe_is_generic_and_deterministic_with_families():
    tracks=[{'id':str(i)} for i in range(8)]
    artists=np.array([0,0,0,1,1,2,2,2]);families=np.array([0,1,2,3,3,5,6,7])
    scores=np.arange(8,0,-1)
    args=(scores,tracks,families,artists,[0])
    result=less_familiar(list(range(1,8)),*args)
    assert result==less_familiar(list(reversed(range(1,8))),*args)
    assert sum(artists[i]==0 for i in result)==1
    assert len({families[i] for i in result})==len(result)
    assert all(artists[a]!=artists[b] for a,b in zip(result,result[1:]))


def test_no_candidate_hit_is_zero_accuracy_not_dropped():
    from types import SimpleNamespace
    engine=SimpleNamespace(artists=np.arange(4),families=np.arange(4))
    result=values(engine,[],{2},[0])
    assert result['ndcg50']==result['recall50']==result['new_artist_share']==0
