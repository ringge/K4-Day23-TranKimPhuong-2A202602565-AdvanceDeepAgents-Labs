"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import random
import re
import threading
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Retry transient HTTP statuses (429/500/502/503/504), transport failures,
    and RetryableError. Honor numeric Retry-After; otherwise use capped
    exponential backoff with jitter. Other exceptions propagate immediately.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError, httpx.HTTPStatusError) as exc:
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                header = exc.response.headers.get("Retry-After")
                if header is not None:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        pass
            elif isinstance(exc, RetryableError):
                retry_after = exc.retry_after
            if attempt == attempts - 1:
                raise
            delay = (retry_after if retry_after is not None
                     else base * 2**attempt + random.uniform(0, base))
            time.sleep(max(0.0, min(cap, delay)))


_arxiv_lock = threading.Lock()
_arxiv_last_call = None

# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _arxiv_last_call
    try:
        terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", query)
        terms = [term for term in terms if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max(1, min(30, max_results)),
        }

        def fetch():
            global _arxiv_last_call
            with _arxiv_lock:
                if _arxiv_last_call is not None:
                    delay = 3.0 - (time.monotonic() - _arxiv_last_call)
                    if delay > 0:
                        time.sleep(delay)
                _arxiv_last_call = time.monotonic()
                response = httpx.get(ARXIV_URL, params=params, timeout=30.0)
            response.raise_for_status()
            return response

        response = with_retry(fetch, attempts=7, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            paper_id = entry.findtext("atom:id", default="", namespaces=ns).split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": entry.findtext("atom:published", default="", namespaces=ns)[:10],
                "title": " ".join(entry.findtext("atom:title", default="", namespaces=ns).split()),
                "summary": " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())[:600],
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def _hf_records(url, params, *, keyword="", prefer_ai=False):
    def fetch():
        response = httpx.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        return response.json()

    records = []
    for item in with_retry(fetch):
        paper = item.get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        title = " ".join((paper.get("title") or item.get("title") or "").split())
        summary = ((paper.get("ai_summary") if prefer_ai else None)
                   or paper.get("summary") or item.get("summary") or "")
        summary = " ".join(summary.split())
        if keyword and keyword.casefold() not in (title + " " + summary).casefold():
            continue
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": title,
            "summary": summary[:600],
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0,
        })
    return records


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            params["date"] = date
        records = _hf_records(HF_DAILY_URL, params, keyword=keyword)
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        records = _hf_records(HF_SEARCH_URL, {"q": query, "limit": max(1, min(50, limit))}, prefer_ai=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def _exa_call(name, arguments, *, max_chars=None):
    key = os.environ.get("EXA_API_KEY", "")
    endpoint = httpx.URL(EXA_URL)
    if key:
        endpoint = endpoint.copy_add_param("exaApiKey", key)

    def rate_limited(meta):
        if isinstance(meta, dict):
            for field, value in meta.items():
                normalized = re.sub(r"[^a-z]", "", field.casefold())
                if "ratelimit" in normalized and value not in (False, None, 0, "", "false"):
                    return True
                if rate_limited(value):
                    return True
        elif isinstance(meta, list):
            return any(rate_limited(value) for value in meta)
        elif isinstance(meta, str):
            return bool(re.search(r"rate[ _-]*limit(?:ed|ing|exceeded)?", meta, re.I))
        return False

    def fetch():
        response = httpx.post(
            endpoint,
            headers={"Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}},
            timeout=60.0,
        )
        response.raise_for_status()
        if "text/event-stream" in response.headers.get("content-type", "") or response.text.lstrip().startswith("event:"):
            messages = []
            for event in re.split(r"\r?\n\r?\n", response.text):
                lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")]
                if lines:
                    messages.append(json.loads("\n".join(lines)))
            message = next((item for item in reversed(messages) if "result" in item or "error" in item), None)
            if message is None:
                raise ValueError("Exa returned no JSON-RPC response")
        else:
            message = response.json()
        if "error" in message:
            raise RuntimeError(f"Exa JSON-RPC error: {message['error']}")
        result = message["result"]
        if rate_limited(result.get("_meta", {})):
            raise RetryableError("Exa free-tier rate limited", retry_after=20.0)
        text = "\n\n".join(item["text"] for item in result.get("content", []) if item.get("type") == "text")
        if result.get("isError"):
            raise RuntimeError(f"Exa tool error: {text}")
        return text

    try:
        text = with_retry(fetch, attempts=7, cap=60.0)
        return (text[:max_chars] if max_chars is not None else text) if text.strip() else "NO RESULTS"
    except Exception as exc:
        message = f"ERROR: {type(exc).__name__}: {exc}"
        if key:
            encoded_key = str(endpoint).split("exaApiKey=", 1)[1].split("&", 1)[0]
            for secret in (key, encoded_key):
                message = message.replace(secret, "[REDACTED]")
        return message


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    return _exa_call("web_search_exa", {
        "query": query,
        "objective": objective or f"Find reliable web sources about {query}",
        "numResults": num_results,
    })


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    return _exa_call("web_fetch_exa", {"urls": [url]}, max_chars=12000)


# The researcher receives exactly these host-side source tools.
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
