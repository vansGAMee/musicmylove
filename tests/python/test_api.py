import json
import httpx

from ml.api import ApiClient, HttpResult, HttpxTransport, TransientApiError, curl_transport


def test_retries_429_and_5xx_then_returns_json():
    responses = iter([
        HttpResult(429, {"retry-after": "0"}, b"busy"),
        HttpResult(503, {}, b"down"),
        HttpResult(200, {}, json.dumps({"ok": True}).encode()),
    ])
    sleeps = []
    client = ApiClient(transport=lambda *_: next(responses), sleep=sleeps.append, jitter=lambda: 0, minimum_interval=0)
    assert client.request_json("GET", "https://example.test") == {"ok": True}
    assert len(sleeps) == 2


def test_enforces_one_second_between_successive_requests():
    now = [10.0]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    client = ApiClient(
        transport=lambda *_: HttpResult(200, {}, b"{}"),
        sleep=sleep,
        clock=lambda: now[0],
        minimum_interval=1.0,
    )
    client.request_json("GET", "https://example.test/one")
    now[0] += 0.25
    client.request_json("GET", "https://example.test/two")
    assert sleeps == [0.75]


def test_uses_cached_value_without_network(tmp_path):
    client = ApiClient(transport=lambda *_: (_ for _ in ()).throw(AssertionError("network used")))
    path = tmp_path / "cached.json"
    path.write_text('{"value": 7}')
    assert client.cached_json(path, "GET", "https://example.test") == {"value": 7}


def test_maps_successful_no_content_to_empty_object():
    client = ApiClient(transport=lambda *_: HttpResult(204, {}, b""), minimum_interval=0)
    assert client.request_json("GET", "https://example.test") == {}


def test_curl_transport_separates_body_from_status_marker():
    class Completed:
        returncode = 0
        stdout = b'{"ok":true}\n200'
        stderr = b""
    calls = []
    result = curl_transport("POST", "https://example.test", b"[]", runner=lambda *args, **kwargs: calls.append((args, kwargs)) or Completed())
    assert result == HttpResult(200, {}, b'{"ok":true}')
    assert calls[0][1]["input"] == b"[]"
    assert "--ipv4" in calls[0][0][0]


def test_httpx_transport_reuses_injected_client():
    class Response:
        status_code = 200
        headers = {"x-ratelimit-remaining": "29"}
        content = b"{}"
    class Client:
        def __init__(self): self.calls = []
        def request(self, *args, **kwargs): self.calls.append((args, kwargs)); return Response()
    client = Client()
    transport = HttpxTransport(client)
    assert transport("GET", "https://example.test", None).status == 200
    assert transport("GET", "https://example.test/two", None).status == 200
    assert len(client.calls) == 2


def test_stops_after_bounded_transport_retries():
    attempts = []
    def failing(*_):
        attempts.append(1)
        raise httpx.ConnectError("blocked")
    client = ApiClient(transport=failing, sleep=lambda _: None, jitter=lambda: 0, minimum_interval=0)
    try:
        client.request_json("GET", "https://example.test")
        assert False, "expected transient failure"
    except TransientApiError:
        pass
    assert len(attempts) == 3
