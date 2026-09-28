import numpy as np
import torch
from scipy import sparse
from types import SimpleNamespace


def fixture():
    # Two independent tastes, plus familiar songs and unrelated highly scored bait.
    n = 83
    tracks = [dict(id=f't{i:03}', artist=f'artist{i}', title=f'song{i}') for i in range(n)]
    tracks[0]['artist']='Seed A'; tracks[1]['artist']='Seed B'
    for i in range(2,8): tracks[i]['artist']='Seed A' if i%2==0 else 'Seed B'
    tracks[8]['artist']='Repeat'; tracks[9]['artist']='REPEAT'
    embeddings = np.zeros((n,4),np.float32)
    embeddings[:42,0]=1; embeddings[42:82,1]=1
    embeddings[1]=[0,1,0,0]; embeddings[82]=[0,0,1,0]
    # Background expands the catalog so direct neighbors really are locally similar.
    background = 200
    for i in range(n,n+background):
        tracks.append(dict(id=f't{i:03}',artist=f'background{i}',title=f'song{i}'))
    embeddings = np.vstack([embeddings,np.tile([0,0,0,1],(background,1))]).astype(np.float32)
    rows=[0]*40+[1]*40+[0]; cols=list(range(2,42))+list(range(42,82))+[82]
    g=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(tracks),len(tracks)))
    pop=np.ones(len(tracks))*5; pop[2:12]=200
    return SimpleNamespace(tracks=tracks,embeddings=torch.from_numpy(embeddings),
        graph={'global':g,'confidence':g,'users':3*g},pop=pop), np.arange(len(tracks),0,-1,dtype=float)


def test_discovery_rejects_unrelated_scores_and_balances_prefixes():
    from python_mvp.discovery_policy import prepare_discovery, select_discovery
    engine,scores=fixture();scores[82]=1e6
    pool=prepare_discovery(engine,[0,1],['Seed A - song0','Seed B - song1'])
    ids,evidence,report=select_discovery(pool,scores)
    assert len(ids)==50 and 82 not in ids
    assert set(evidence[i]['direction'] for i in ids[:2])=={'seed a','seed b'}
    for k in range(1,len(ids)+1):
        assert sum(engine.tracks[i]['artist'].casefold() in pool.familiar for i in ids[:k]) <= int(k*.3)
        assert sum(i in pool.popular for i in ids[:k]) <= int(k*.2)
    names=[engine.tracks[i]['artist'].casefold() for i in ids]
    assert all(a!=b for a,b in zip(names,names[1:]))
    assert max(names.count(a) for a in names)<=2
    assert all(evidence[i]['shared_listeners']>=3 for i in ids)
    other=prepare_discovery(engine,[1,0,1],['Seed B - song1','Seed A - song0'])
    assert select_discovery(other,scores)[0]==ids
    assert report['policy']=='graph-discovery-v1'


def test_sparse_direction_never_gets_filled_with_unrelated_tracks():
    from python_mvp.discovery_policy import prepare_discovery, select_discovery
    engine,scores=fixture()
    engine.graph={k:v*0 for k,v in engine.graph.items()}
    pool=prepare_discovery(engine,[0],['Seed A - song0'])
    ids,_,report=select_discovery(pool,scores)
    assert ids==[]
    assert report['unsupported_directions']==['seed a']


def test_known_recording_variants_are_excluded_before_shortlisting():
    from python_mvp.discovery_policy import prepare_discovery, select_discovery
    engine,scores=fixture()
    engine.tracks[15].update(artist='already',title='heard - Remastered')
    engine.tracks[16].update(artist='already',title='heard')
    pool=prepare_discovery(engine,[0,1],['Seed A - song0','Seed B - song1','already - heard - 2011 Remaster'])
    ids,_,_=select_discovery(pool,scores)
    assert not {0,1,15,16}&set(ids)


def test_one_artist_cannot_hide_another_eligible_direction_neighbor():
    from python_mvp.discovery_policy import prepare_discovery, select_discovery
    engine,scores=fixture()
    for i in range(2,42):engine.tracks[i]['artist']='Repeat'
    engine.tracks[42]['artist']='Tail'
    engine.embeddings[42]=engine.embeddings[0]
    for key,value in [('global',1.),('confidence',1.),('users',3.)]:
        matrix=engine.graph[key].tolil();matrix[0,42]=value;engine.graph[key]=matrix.tocsr()
    pool=prepare_discovery(engine,[0],['Seed A - song0'])
    ids,_,_=select_discovery(pool,scores)
    assert 42 in ids and len(ids)==3


def test_weak_tail_cannot_win_only_because_of_high_neural_score():
    from python_mvp.discovery_policy import prepare_discovery
    engine,_=fixture();engine.embeddings[81]=engine.embeddings[0]
    for key,value in [('global',.01),('confidence',1.),('users',3.)]:
        matrix=engine.graph[key].tolil();matrix[0,81]=value;engine.graph[key]=matrix.tocsr()
    pool=prepare_discovery(engine,[0],['Seed A - song0'])
    assert 81 not in pool.candidates


def test_popular_artist_cannot_evade_limit_with_low_count_track():
    from python_mvp.discovery_policy import prepare_discovery, select_discovery
    engine,scores=fixture()
    engine.artist_ids=np.arange(len(engine.tracks))
    engine.artist_pop=np.ones(len(engine.tracks))*5
    engine.artist_pop[12]=1000
    pool=prepare_discovery(engine,[0,1],['Seed A - song0','Seed B - song1'])
    assert engine.pop[12]==5 and 12 in pool.popular
    ids,_,_=select_discovery(pool,scores)
    assert 12 not in ids[:4]


def test_cli_uses_graph_gate_before_neural_playlist_selection(tmp_path):
    from python_mvp.cli import generate
    engine,scores=fixture();scores[82]=1e6
    engine.features=lambda seeds:(np.ones((len(engine.tracks),22)),set(range(len(engine.tracks))),{})
    engine.score=lambda model,features:scores
    result=generate(engine,None,['Seed A - song0','Seed B - song1'],[],'2',
                    feedback_dir=tmp_path,personal=False)
    assert result['selection_policy']['policy']=='graph-discovery-v1'
    assert len(result['top50'])==50
    assert all(t['id']!='t082' and t['discovery_evidence'] for t in result['top50'])
