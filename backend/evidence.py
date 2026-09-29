"""Source-specific evidence and URL normalization.

Before this module, every grounding URL got the same first 300 characters of
the model's answer as its "snippet", so a citation [N] pointed at text that
had nothing to do with source N. Gemini's grounding metadata links each
supported answer segment to the chunk indices that support it; we invert that
map so each source carries only the passages attributed to it.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid"}
MAX_PASSAGE_CHARS = 600


def normalize_url(url: str) -> str:
    """Dedup key: lowercase scheme/host, drop fragment, tracking params, trailing slash."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or ""
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(sorted(query)), "")
    )


def passages_by_chunk(metadata) -> dict[int, list[str]]:
    """chunk index -> answer segments that Gemini says the chunk supports."""
    out: dict[int, list[str]] = {}
    for support in getattr(metadata, "grounding_supports", None) or []:
        text = (getattr(getattr(support, "segment", None), "text", "") or "").strip()
        if not text:
            continue
        for idx in getattr(support, "grounding_chunk_indices", None) or []:
            bucket = out.setdefault(int(idx), [])
            if text not in bucket:
                bucket.append(text)
    return out


def join_passages(passages: list[str]) -> str:
    return " … ".join(passages)[:MAX_PASSAGE_CHARS]
