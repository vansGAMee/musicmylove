import numpy as np
import torch


def test_joint_audio_changes_learned_score_and_receives_gradient():
    from python_mvp.joint import JointRanker
    torch.manual_seed(9);m=JointRanker()
    x=torch.randn(7,610,requires_grad=True);z=m.encode(x)
    scores=m.score(z[:2][None],z[2:][None],torch.ones(1,5,4))
    scores.sum().backward()
    assert x.grad[:,96:608].abs().sum()>0
    altered=x.detach().clone();altered[:,96:608]=0
    assert not torch.allclose(m.encode(altered),z)


def test_joint_candidates_protect_whole_library_and_are_permutation_invariant():
    from python_mvp.joint import retrieve,select_seeds
    artists=np.arange(100)%7;families=np.arange(100);families[9]=0
    neighbors=np.tile(np.arange(24,dtype=np.int32),(100,1))
    assert select_seeds([7,1,3,7],artists)==select_seeds([3,1,7],artists)
    candidates=retrieve([0,1],neighbors,families,artists)
    assert not {0,1,9}&set(candidates)


def test_joint_training_pairs_never_use_known_positive_family_as_negative():
    from python_mvp.joint_training import pairs
    families=np.arange(20);families[8]=2
    p,n=pairs({1}, {0,1,2},range(20),families,np.random.default_rng(1))
    assert len(p)>0 and not {0,1,2,8}&set(n)


def test_browser_export_python_typescript_parity(tmp_path):
    import json,subprocess
    from python_mvp.joint import JointRanker,JointEngine,extras,numpy_scores,select_seeds,select
    from python_mvp.joint_export import export_browser
    torch.manual_seed(7);n=100;x=np.random.default_rng(7).normal(size=(n,610)).astype(np.float32)
    x[:,608]=np.arange(n)%2;x[:,609]=np.arange(n)/n
    model=JointRanker().eval()
    with torch.no_grad():z=model.encode(torch.tensor(x)).numpy()
    pack=dict(x=x,neighbors=np.tile(np.arange(20,44,dtype=np.int32),(n,1)),
              tracks=[dict(id=f't{i:03}',artist=f'A{i%40}',title=f'T{i}') for i in range(n)],
              artists=np.arange(n)%40,families=np.arange(n))
    weights=dict(w1=model.rank[0].weight.detach().tolist(),b1=model.rank[0].bias.detach().tolist(),
                 w2=model.rank[2].weight.detach().tolist(),b2=model.rank[2].bias.detach().tolist())
    export_browser(tmp_path,pack,z,weights,'TEST',{})
    engine=JointEngine(tmp_path);seeds=[3,0,2,1];features,candidates,_=engine.features(seeds)
    selected=select_seeds(seeds,pack['artists']);ids=sorted(candidates)
    with torch.no_grad():expected=model.score(torch.tensor(z[selected])[None],torch.tensor(z[ids])[None],
        torch.tensor(extras(selected,ids,pack['artists'],x[:,609],x[:,608]))[None])[0].numpy()
    assert np.max(np.abs(expected-features[ids,0]))<1e-5
    input_path=tmp_path/'input.json';input_path.write_text(json.dumps(seeds))
    ts=json.loads(subprocess.check_output(['node_modules/.bin/tsx','scripts/joint-parity.ts',str(tmp_path),str(input_path)],text=True))
    assert ts['seeds']==selected and ts['candidates']==ids
    assert np.max(np.abs(np.asarray(ts['scores'])-features[ids,0]))<1e-5
    assert ts['playlist']==select(ids,features[:,0],pack['tracks'],pack['families'],pack['artists'])
    # Source CLAP vectors and model weights are absent from deployment artifacts.
    assert (tmp_path/'vectors.f32').stat().st_size==n*32*4
    from python_mvp.cli import generate
    result=generate(engine,None,['A0 - T0','A1 - T1'],[],'2',personal=False,feedback_dir=tmp_path/'feedback')
    assert result['top50'] and result['sound_report'] is None


def test_joint_small_optimization_and_checkpoint_reuse(tmp_path):
    from python_mvp.joint_training import fit
    torch.set_num_threads(2)
    n=50;rng=np.random.default_rng(5);x=rng.normal(size=(n,610)).astype(np.float32)
    x[:,608]=np.arange(n)%2;x[:,609]=np.arange(n)/n
    pack=dict(x=x,artists=np.arange(n),families=np.arange(n),
              tracks=[dict(id=str(i),artist=str(i),title=str(i)) for i in range(n)],
              seeds=np.tile([0,1],(12,1)),mask=np.ones((12,2),bool),pos=np.full(12,2),neg=np.arange(12)+3,
              dev=[dict(user='dev',seeds=[0,1],candidates=list(range(2,n)),targets=[2],discovery=True)])
    state=fit(pack,tmp_path,'cpu',2,True)
    assert state['epoch']>=1 and np.isfinite(state['history'][0]['loss'])
    assert fit(pack,tmp_path,'cpu',2,True)['dev']==state['dev']
