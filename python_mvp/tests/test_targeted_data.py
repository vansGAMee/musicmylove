from types import SimpleNamespace
import pytest


def test_selection_only_uses_representation_users_and_missing_directions():
    from python_mvp.targeted_data import select_users
    tracks = [dict(id='a', artist='Wanted',title='Known'),dict(id='b',artist='Wanted',title='Rare'),dict(id='c',artist='Other',title='No')]
    engine = SimpleNamespace(meta={'train_users':['rep1','rep2']},data={'tracks':tracks,'users':[
        {'id':'rep1','tracks':[0,1]}, {'id':'rep2','tracks':[2]}, {'id':'dev','tracks':[0,1]}, {'id':'ranker','tracks':[1]}]})
    coverage = {'tracks':[{'input':'Wanted - Rare','status':'in_raw_not_catalog','raw_ids':['b']}]}
    assert select_users(engine, coverage, 100) == ['rep1']


def test_api_pages_are_bounded_and_ignore_out_of_window_listens(tmp_path):
    from python_mvp.targeted_data import collect_user
    calls = []
    def request(url):
        calls.append(url)
        return {'payload':{'listens':[
            {'listened_at':90,'track_metadata':{'artist_name':'A','track_name':'New'}},
            {'listened_at':1,'track_metadata':{'artist_name':'A','track_name':'Old'}}]}}
    rows = collect_user('a/b', tmp_path, upper=100, lower=50, pages=3, request=request)
    assert len(rows) == 1 and rows[0]['user_name']=='a/b'
    assert len(calls) == 1 and 'a%2Fb' in calls[0]
    # Completed API cache is reused; no request on repeated invocation.
    def forbidden(url):
        raise AssertionError('cached request repeated')
    assert collect_user('a/b',tmp_path,upper=100,lower=50,pages=3,request=forbidden) == rows


@pytest.mark.parametrize('frozen_in_model', [False, True])
def test_targeted_pipeline_merges_only_train_history_and_resumes(tmp_path, monkeypatch, frozen_in_model):
    import hashlib
    import json
    import urllib.parse
    from python_mvp import targeted_data as module
    from python_mvp import cli
    from python_mvp.prepare_data import prepare
    from python_mvp.config import fingerprint, write_json
    source, run, model = tmp_path/'source', tmp_path/'targeted', tmp_path/'model'
    path = tmp_path/'original.jsonl'
    rows = [{'user_name':f'user{i}', 'listened_at':1700000000+j,
             'track_metadata':{'artist_name':f'artist{i%20}','track_name':f'song{j}'}}
            for i in range(600) for j in range(5)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    prepare([path],source/'data',source/'reports')
    write_json(source/'reports/collection.json',[{'output':str(path),'sha256':fingerprint(path)}])
    if not frozen_in_model:
        (source/'reports/final_consumed.lock').write_text('consumed')
    data=json.loads((source/'data/dataset.json').read_text())
    engine=SimpleNamespace(data=data,tracks=data['tracks'],meta={
        'fingerprint':data['fingerprint'],'seen':list(range(len(data['tracks']))),
        'train_users':[u['id'] for u in data['users'] if u['split']=='train']})
    monkeypatch.setattr(cli,'open_engine',lambda *a:(engine,None))
    model.mkdir(); (model/'ranker.pt').write_bytes(b'synthetic parent checkpoint')
    if frozen_in_model:
        (model/'reports').mkdir()
        (model/'reports/final_consumed.lock').write_text('consumed')
    library=tmp_path/'liked.txt'; library.write_text('artist0 - missing\n')
    original_collector=module.collect_user
    calls=[]
    def request(url):
        calls.append(url)
        ts=int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)['max_ts'][0])-1
        return {'payload':{'listens':[{'listened_at':ts,'track_metadata':{'artist_name':'artist0','track_name':'missing'}}]}}
    monkeypatch.setattr(module,'collect_user',lambda *a:original_collector(*a,request=request))
    monkeypatch.setattr('sys.argv',['targeted_data',str(library),'--source',str(source),'--run',str(run),
                                   '--model',str(model),'--users','3','--collect'])
    module.main()
    assert len(calls)==3
    expanded=json.loads((run/'data/dataset.json').read_text())
    missing=next(i for i,t in enumerate(expanded['tracks']) if t['title']=='missing')
    owners=[u for u in expanded['users'] if missing in u['tracks']]
    assert len(owners)==3 and all(u['split']=='train' for u in owners)
    assert (run/'reports/final_consumed.lock').read_text()=='consumed'
    module.main()
    assert len(calls)==3
