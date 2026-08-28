"""Direct LAN Fast Parallel vLLM Embedding Client with zero proxy overhead."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests


def wait_for_server(base_url, api_key, timeout=600.0, poll=3.0):
    """Block until the embedding server answers /v1/models. Returns the served id."""
    t0 = time.time()
    session = requests.Session()
    session.trust_env = False  # Ensure direct LAN communication
    headers = {"Authorization": f"Bearer {api_key}"}
    while True:
        try:
            resp = session.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=poll)
            if resp.status_code == 200:
                data = resp.json()
                return data["data"][0]["id"]
        except Exception as e:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"embedding server {base_url} not up after {timeout}s: {e}")
        time.sleep(poll)


def embed_texts(texts, *, base_url="http://127.0.0.1:8001/v1", api_key="EMPTY",
                model=None, batch_size=512, max_workers=8, normalize=True, show_every=5, dimensions=None):
    """Embed a list of strings -> (N, dim) float32 with high-throughput concurrent batching.

    Uses requests.Session with trust_env=False for direct high-speed LAN calls.
    """
    if model is None:
        model = wait_for_server(base_url, api_key)

    n_total = len(texts)
    if n_total == 0:
        return np.empty((0, 1024), dtype=np.float32)

    dim = 1024 if dimensions is None else int(dimensions)
    out_arr = np.empty((n_total, dim), dtype=np.float32)

    batches = []
    for bi, s in enumerate(range(0, n_total, batch_size)):
        e = min(s + batch_size, n_total)
        batches.append((bi, s, e, texts[s:e]))

    n_batches = len(batches)
    t0 = time.time()
    lock = threading.Lock()
    completed_cnt = 0
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Thread-local sessions for safe concurrency without proxy
    local_sessions = threading.local()

    def get_session():
        if not hasattr(local_sessions, "session"):
            s = requests.Session()
            s.trust_env = False
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
            s.mount("http://", adapter)
            local_sessions.session = s
        return local_sessions.session

    def _worker(b_arg):
        nonlocal completed_cnt
        bi, s, e, batch_texts = b_arg
        cleaned = [str(t)[:1500] if t is not None else "" for t in batch_texts]
        kw = {"input": cleaned, "model": model}
        if dimensions is not None:
            kw["dimensions"] = int(dimensions)

        session = get_session()
        for attempt in range(3):
            try:
                resp = session.post(url, json=kw, headers=headers, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    break
                else:
                    time.sleep(1.0)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0)

        batch_embs = [None] * len(batch_texts)
        for item in data["data"]:
            batch_embs[int(item["index"])] = item["embedding"]

        out_arr[s:e] = np.asarray(batch_embs, dtype=np.float32)

        with lock:
            completed_cnt += 1
            cnt = completed_cnt
            if show_every and (cnt % show_every == 0 or cnt == n_batches):
                elapsed = time.time() - t0
                speed = (cnt * batch_size) / max(1e-3, elapsed)
                pct = (cnt / n_batches) * 100.0
                print(f"  [embed] {cnt}/{n_batches} batches ({pct:.1f}%) | {speed:.1f} texts/s | elapsed: {elapsed:.1f}s", flush=True)
        return bi

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker, batches))

    if normalize:
        norms = np.linalg.norm(out_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out_arr /= norms

    return out_arr
