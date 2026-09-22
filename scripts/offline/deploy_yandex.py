"""Deploy the prepared gateway to Yandex Cloud (no Vercel Functions).
YC_IAM_TOKEN is deployment-only. Never written to disk or printed.
Optional YC_FOLDER_ID / FRONTEND_ORIGIN / SERPER_API_KEY.
"""
import base64
import json
import os
import time
from pathlib import Path
import requests

TOKEN = os.environ.get('YC_IAM_TOKEN')
if not TOKEN:
    raise SystemExit('Yandex Cloud — YC_IAM_TOKEN — https://yandex.cloud/ru/docs/iam/operations/iam-token/create')
SESSION = requests.Session()
SESSION.headers.update({'Authorization': 'Bearer '+TOKEN})
BASE = 'https://serverless-functions.api.cloud.yandex.net/functions/v1'
STATE = Path('data/cache/yandex-deployment.json')


def api(method, url, **kwargs):
    response = SESSION.request(method, url, timeout=30, **kwargs)
    if not response.ok:
        raise RuntimeError(f'Yandex Cloud HTTP {response.status_code}: {response.text[:300]}')
    return response.json()


def operation(value):
    for _ in range(90):
        if value.get('done'):
            if value.get('error'):
                raise RuntimeError(value['error']['message'])
            return value.get('response', {})
        time.sleep(2)
        value = api('GET', 'https://operation.api.cloud.yandex.net/operations/'+value['id'])
    raise TimeoutError('Deployment is still running; do not create a second function')


folder = os.environ.get('YC_FOLDER_ID')
if not folder:
    clouds = api('GET', 'https://resource-manager.api.cloud.yandex.net/resource-manager/v1/clouds').get('clouds', [])
    if len(clouds) != 1:
        raise SystemExit('Set YC_FOLDER_ID to select the deployment folder')
    folders = api('GET', 'https://resource-manager.api.cloud.yandex.net/resource-manager/v1/folders', params={'cloudId': clouds[0]['id']}).get('folders', [])
    defaults = [f for f in folders if f.get('name') == 'default']
    selected = defaults or folders
    if len(selected) != 1:
        raise SystemExit('Set YC_FOLDER_ID to select the deployment folder')
    folder = selected[0]['id']
origin = os.environ.get('FRONTEND_ORIGIN')
if not origin or not origin.startswith('https://'):
    raise SystemExit('Set FRONTEND_ORIGIN to the existing static Vercel site HTTPS origin')
state = json.loads(STATE.read_text()) if STATE.exists() else {}
if state and state.get('folderId') != folder:
    raise SystemExit('Existing deployment belongs to another folder')
if not state.get('functionId'):
    existing = api('GET', BASE+'/functions', params={'folderId': folder}).get('functions', [])
    matches = [f for f in existing if f.get('name') == 'musicmylove-batch']
    function = matches[0] if matches else operation(api('POST', BASE+'/functions', json={'folderId': folder, 'name': 'musicmylove-batch', 'description': 'Public playlist import and batched verified tracklist context'}))
    state = {'folderId': folder, 'functionId': function['id']}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))
content = Path('data/cache/gateway.zip').read_bytes()
if len(content) > 3_500_000:
    raise SystemExit('Gateway ZIP exceeds direct-upload limit')
env = {'FRONTEND_ORIGIN': origin, 'TRACK_CATALOG': 'catalog.json'}
if os.environ.get('SERPER_API_KEY'):
    env['SERPER_API_KEY'] = os.environ['SERPER_API_KEY']
version = operation(api('POST', BASE+'/versions', json={
    'functionId': state['functionId'], 'runtime': 'nodejs22', 'entrypoint': 'index.handler',
    'resources': {'memory': '536870912'}, 'executionTimeout': '30s',
    'content': base64.b64encode(content).decode(), 'environment': env,
}))
operation(api('POST', BASE+'/functions/'+state['functionId']+':updateAccessBindings', json={
    'accessBindingDeltas': [{'action': 'ADD', 'accessBinding': {'roleId': 'serverless.functions.invoker', 'subject': {'id': 'allUsers', 'type': 'system'}}}]
}))
state.update({'versionId': version['id'], 'url': 'https://functions.yandexcloud.net/'+state['functionId']})
STATE.write_text(json.dumps(state))
health = requests.get(state['url'], params={'route': '/health'}, timeout=30)
health.raise_for_status()
print('NEXT_PUBLIC_GATEWAY_URL='+state['url'])
print('Health:', health.json())
