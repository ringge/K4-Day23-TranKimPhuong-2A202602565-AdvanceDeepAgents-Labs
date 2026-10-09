"""Validate report citations and sources using only the standard library.

Runs inside the sandbox as:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
Prints problems and exits 1, or prints "OK: ..." and exits 0.
"""
import json
import re
import sys
from urllib.parse import urlsplit

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_NUMBER = r"[+-]?\d+"
_PART = rf"{_NUMBER}(?:\s*[-–]\s*{_NUMBER})?"
_CITATION = re.compile(rf"\[({_PART}(?:\s*,\s*{_PART})*)\]")
_RANGE = re.compile(rf"({_NUMBER})\s*[-–]\s*({_NUMBER})")
_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*\r?$")
_REFERENCE = re.compile(rf"^\[({_NUMBER})\]")
_URL = re.compile(r'https?://[^\s<>`"\[\]]+')


def _without_code(text):
    """Mask fenced blocks and backtick spans while retaining line boundaries."""
    def mask(value):
        return re.sub(r"[^\r\n]", " ", value)

    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        if fence is not None:
            lines.append(mask(line))
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{fence[1]},}}[ \t]*",
                line.rstrip("\r\n"),
            ):
                fence = None
            continue
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)", line)
        if opening and not (
            opening.group(1)[0] == "`" and "`" in opening.group(2)
        ):
            fence = (opening.group(1)[0], len(opening.group(1)))
            lines.append(mask(line))
        else:
            lines.append(line)
    text = "".join(lines)
    spans = re.compile(r"(`+)(?!`)(.*?)(?<!`)\1(?!`)", re.DOTALL)
    return spans.sub(lambda match: mask(match.group(0)), text)


def _urls(text):
    """Extract bare or Markdown URLs, excluding surrounding punctuation."""
    urls = []
    for match in _URL.finditer(text):
        url = match.group(0).rstrip(".,;:!?")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        while url.endswith("}") and url.count("}") > url.count("{"):
            url = url[:-1]
        urls.append(url)
    return urls


def check(report_text, sources):
    """Return source-schema, body-citation, and reference-list problems.

    Comma groups and inclusive ranges count as body citations; code and
    Markdown link labels do not. Each numbered source must be cited and have
    exactly one reference line containing its one, exact HTTP(S) URL.
    """
    problems = []
    if not isinstance(sources, list):
        problems.append("sources.json must be a JSON list of objects")
        sources = []
    if not sources:
        problems.append("no sources in sources.json")

    by_n = {}
    seen_urls = set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        valid_n = isinstance(n, int) and not isinstance(n, bool)
        if not valid_n:
            problems.append(f"source entry {index} has non-integer n: {n!r}")
        elif n in by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_n[n] = source

        valid_url = isinstance(url, str) and url.startswith(("http://", "https://"))
        if valid_url:
            try:
                valid_url = bool(urlsplit(url).hostname) and not any(
                    char.isspace() for char in url
                )
            except ValueError:
                valid_url = False
        if not valid_url:
            problems.append(f"source entry {index} has invalid HTTP(S) url: {url!r}")
        if isinstance(url, str):
            if url in seen_urls:
                problems.append(f"duplicate source url: {url}")
            seen_urls.add(url)

    visible = _without_code(report_text)
    heading = _HEADING.search(visible)
    if heading is None:
        problems.append("report is missing ## References")
        body, references = visible, ""
    else:
        body, references = visible[:heading.start()], visible[heading.end():]
        next_section = re.search(r"(?m)^#{1,2}[ \t]+", references)
        if next_section:
            references = references[:next_section.start()]

    # Explicit reference-style Markdown links also have non-citation labels.
    link_labels = {
        match.group(1).strip().casefold()
        for match in re.finditer(r"(?m)^ {0,3}\[([^\]\n]+)\]:[ \t]*\S+", visible)
    }
    cited = set()
    for match in _CITATION.finditer(body):
        following = body[match.end():]
        label = match.group(1).strip().casefold()
        if re.match(r"[ \t]*\(", following):
            continue
        reference_link = re.match(r"[ \t]*\[([^\]\n]*)\]", following)
        if reference_link and (
            reference_link.group(1).strip().casefold() or label
        ) in link_labels:
            continue
        if label in link_labels or following.startswith(":"):
            continue
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = _RANGE.fullmatch(part)
            if span:
                start, end = map(int, span.groups())
                if end < start:
                    problems.append(f"invalid descending citation range [{part}]")
                else:
                    cited.update(range(start, end + 1))
            else:
                cited.add(int(part))

    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited")

    reference_counts = {}
    for line in references.splitlines():
        match = _REFERENCE.match(line)
        if match is None:
            continue
        n = int(match.group(1))
        reference_counts[n] = reference_counts.get(n, 0) + 1
        if n not in by_n:
            problems.append(f"reference [{n}] missing from sources.json")
        urls = _urls(line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL (found {len(urls)})")
        elif n in by_n and urls[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] URL does not match sources.json")
    for n in sorted(by_n):
        if n not in reference_counts:
            problems.append(f"source [{n}] missing from References")
    for n, count in sorted(reference_counts.items()):
        if count > 1:
            problems.append(f"reference [{n}] appears {count} times (expected exactly once)")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
