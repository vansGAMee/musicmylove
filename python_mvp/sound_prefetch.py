"""Bounded network prefetch. The caller remains the only owner of GPU inference."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import tempfile


@contextmanager
def ordered_prefetch(items,prepare,workers):
    if not 1<=workers<=8:raise ValueError('workers must be between 1 and 8')
    iterator=iter(items);pool=ThreadPoolExecutor(max_workers=workers)
    pending=deque()
    def results():
        pending.extend(pool.submit(prepare,item) for item in islice(iterator,workers))
        while pending:
            yield pending.popleft().result()
            for item in islice(iterator,1):pending.append(pool.submit(prepare,item))
    stream=results()
    try:yield stream
    finally:
        stream.close()
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


@dataclass
class PreviewJob:
    track: dict
    result: dict | None = None
    path: Path | None = None
    error: Exception | None = None
    cached: bool = False


@contextmanager
def preview_jobs(tracks,cache,search,workers,fetch):
    with tempfile.TemporaryDirectory(prefix='music-prefetch-') as temp:
        def prepare(track):
            job=PreviewJob(track)
            try:
                job.cached=cache.get(track['id']) is not None
                if not job.cached:
                    job.result=search.search(track)
                    if job.result:
                        with tempfile.NamedTemporaryFile(dir=temp,delete=False) as f:job.path=Path(f.name)
                        fetch(job.result['preview'],job.path)
            except Exception as exc:job.error=exc
            return job
        with ordered_prefetch(tracks,prepare,workers) as jobs:
            def results():
                for job in jobs:
                    try:yield job
                    finally:
                        if job.path:job.path.unlink(missing_ok=True)
            stream=results()
            try:yield stream
            finally:stream.close()
