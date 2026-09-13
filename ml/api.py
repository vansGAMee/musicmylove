import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

USER_AGENT = "MusicMyLove/0.2 (real-cohort recommendation research; contact: dev@example.com)"


class PermanentApiError(RuntimeError):
    pass


class TransientApiError(RuntimeError):
    pass


@dataclass
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes


def urllib_transport(method: str, url: str, body: bytes | None) -> HttpResult:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return HttpResult(response.status, {key.lower(): value for key, value in response.headers.items()}, response.read())
    except urllib.error.HTTPError as error:
        return HttpResult(error.code, {key.lower(): value for key, value in error.headers.items()}, error.read())


def curl_transport(method: str, url: str, body: bytes | None, runner=subprocess.run) -> HttpResult:
    command = [
        "curl", "--ipv4", "--silent", "--show-error", "--max-time", "20",
        "--request", method,
        "--header", f"User-Agent: {USER_AGENT}",
        "--header", "Accept: application/json",
        "--write-out", "\n%{http_code}",
    ]
    if body is not None:
        command.extend(["--header", "Content-Type: application/json", "--data-binary", "@-"])
    command.append(url)
    completed = runner(command, input=body, capture_output=True)
    if completed.returncode != 0:
        raise OSError(completed.stderr.decode(errors="replace"))
    response_body, marker = completed.stdout.rsplit(b"\n", 1)
    return HttpResult(int(marker), {}, response_body)


class HttpxTransport:
    def __init__(self, client=None):
        self.client = client or httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})

    def __call__(self, method: str, url: str, body: bytes | None) -> HttpResult:
        headers = {"Content-Type": "application/json"} if body is not None else None
        response = self.client.request(method, url, content=body, headers=headers)
        return HttpResult(response.status_code, {key.lower(): value for key, value in response.headers.items()}, response.content)


class ApiClient:
    def __init__(self, transport: Callable[[str, str, bytes | None], HttpResult] | None = None, sleep=time.sleep, jitter=random.random, minimum_interval=1.0, clock=time.monotonic):
        self.transport = transport or HttpxTransport()
        self.sleep = sleep
        self.jitter = jitter
        self.minimum_interval = minimum_interval
        self.clock = clock
        self.last_request_at = None

    def request_json(self, method: str, url: str, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        last_error = None
        for attempt in range(3):
            if self.last_request_at is not None:
                delay = self.minimum_interval - (self.clock() - self.last_request_at)
                if delay > 0:
                    self.sleep(delay)
            try:
                result = self.transport(method, url, body)
                self.last_request_at = self.clock()
                if result.status == 204:
                    return {}
                if 200 <= result.status < 300:
                    return json.loads(result.body)
                if result.status != 429 and result.status < 500:
                    raise PermanentApiError(f"HTTP {result.status}")
                retry_after = float(result.headers.get("retry-after", "0") or 0)
                self.sleep(max(retry_after, 2 ** attempt + self.jitter() * 0.25))
                last_error = RuntimeError(f"HTTP {result.status}")
            except PermanentApiError:
                raise
            except (OSError, TimeoutError, json.JSONDecodeError, httpx.TransportError) as error:
                last_error = error
                self.sleep(2 ** attempt + self.jitter() * 0.25)
        raise TransientApiError(f"API request failed after 3 attempts: {last_error}")

    def cached_json(self, path: Path, method: str, url: str, payload=None):
        if path.exists():
            return json.loads(path.read_text())
        value = self.request_json(method, url, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
        return value
