"""Deterministic URL-lexical feature extraction (no network, no label use)."""
from __future__ import annotations

import hashlib
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

SPECIAL_RE = re.compile(r"[^A-Za-z0-9]")

STRICT_COLS = [
    "url_length", "host_length", "path_length", "query_length", "fragment_length",
    "num_dots_host", "num_hyphens_host", "num_subdomain_levels",
    "num_digits_url", "digits_ratio", "num_special_chars_url",
    "num_at", "num_percent", "num_equal", "num_question", "num_ampersand",
    "num_slashes_path", "num_query_params",
    "is_https", "host_is_ip", "has_port", "has_punycode",
    "url_entropy", "path_present", "query_present",
    "tld_length", "has_www", "parse_error_flag",
]
COLLECTION_PROXY = [
    "has_www", "is_https", "path_present", "query_present",
    "path_length", "query_length", "num_slashes_path", "num_query_params",
    "num_question", "num_ampersand",
]
NO_CUES_COLS = [c for c in STRICT_COLS if c not in set(COLLECTION_PROXY)]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in freq.values()))


def normalize_url(url: str) -> str:
    """Lowercase scheme/host, drop default ports. Keeps path/query order intact."""
    try:
        p = urlsplit(url.strip())
    except Exception:
        return url.strip()
    scheme = (p.scheme or "").lower()
    host = (p.hostname or "").lower()
    if not host:
        # relative / malformed: return stripped as-is
        return url.strip()
    port = None
    try:
        port = p.port
    except Exception:
        port = None
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    # preserve userinfo? drop for grouping simplicity but keep host+port
    new_netloc = host if port is None else f"{host}:{port}"
    path = p.path or ""
    query = f"?{p.query}" if p.query else ""
    # drop fragment for grouping
    return f"{scheme}://{new_netloc}{path}{query}" if scheme else f"{new_netloc}{path}{query}"


def row_id_for(normalized: str, salt: str = "") -> str:
    h = hashlib.sha1((normalized + "|" + salt).encode("utf-8")).hexdigest()[:16]
    return h


def _host_is_ip(host: str) -> int:
    h = host.strip().strip("[]").lower()
    try:
        ipaddress.ip_address(h)
        return 1
    except Exception:
        return 0


def extract_url_features(url: object) -> dict:
    s = "" if url is None else str(url)
    feats: dict = {}
    parse_error = 0
    try:
        p = urlsplit(s.strip())
        scheme = (p.scheme or "").lower()
        netloc = p.netloc or ""
        host = (p.hostname or "")
        path = p.path or ""
        query = p.query or ""
        fragment = p.fragment or ""
        if not netloc and not scheme:
            # likely relative URL
            parse_error = 1
            host = ""
    except Exception:
        parse_error = 1
        scheme, host, path, query, fragment = "", "", "", "", ""
    host_l = (host or "").lower()
    is_ip = _host_is_ip(host_l) if host_l else 0
    # port
    has_port = 0
    try:
        port = urlsplit(s.strip()).port if s.strip() else None
        has_port = 1 if port is not None else 0
    except Exception:
        # manual fallback: colon in netloc beyond host
        has_port = 0
    n = len(s)
    feats["url_length"] = float(n)
    feats["host_length"] = float(len(host_l))
    feats["path_length"] = float(len(path))
    feats["query_length"] = float(len(query))
    feats["fragment_length"] = float(len(fragment))
    feats["num_dots_host"] = float(host_l.count("."))
    feats["num_hyphens_host"] = float(host_l.count("-"))
    if is_ip or not host_l:
        feats["num_subdomain_levels"] = 0.0
    else:
        parts = [x for x in host_l.split(".") if x]
        feats["num_subdomain_levels"] = float(max(0, len(parts) - 2))
    ndig = sum(ch.isdigit() for ch in s)
    feats["num_digits_url"] = float(ndig)
    feats["digits_ratio"] = float(ndig / n) if n else 0.0
    feats["num_special_chars_url"] = float(len(SPECIAL_RE.findall(s)))
    feats["num_at"] = float(s.count("@"))
    feats["num_percent"] = float(s.count("%"))
    feats["num_equal"] = float(s.count("="))
    feats["num_question"] = float(s.count("?"))
    feats["num_ampersand"] = float(s.count("&"))
    feats["num_slashes_path"] = float(path.count("/"))
    feats["num_query_params"] = float(len([x for x in query.split("&") if x]) if query else 0)
    feats["is_https"] = 1.0 if scheme == "https" else 0.0
    feats["host_is_ip"] = float(is_ip)
    feats["has_port"] = float(has_port)
    feats["has_punycode"] = 1.0 if "xn--" in host_l else 0.0
    feats["url_entropy"] = float(shannon_entropy(s))
    feats["path_present"] = 1.0 if len(path) > 1 else 0.0  # '/' alone = homepage
    feats["query_present"] = 1.0 if len(query) > 0 else 0.0
    # TLD length: last label after dot (offline heuristic, no network)
    if is_ip or not host_l or "." not in host_l:
        feats["tld_length"] = 0.0
    else:
        feats["tld_length"] = float(len(host_l.rsplit(".", 1)[-1]))
    feats["has_www"] = 1.0 if host_l == "www" or host_l.startswith("www.") else 0.0
    feats["parse_error_flag"] = float(parse_error)
    # ensure finite
    for k, v in list(feats.items()):
        try:
            f = float(v)
            if not (f == f and abs(f) != float("inf")):
                feats[k] = 0.0
        except Exception:
            feats[k] = 0.0
    return feats


_DOMAIN_CACHE: dict[str, str] = {}


def registrable_domain(url: str, normalized: str = "") -> str:
    """eTLD+1 via tldextract offline; fallback last-2-labels. IP -> host. Unknown -> host/URL."""
    s = str(url)
    try:
        p = urlsplit(s.strip())
        host = (p.hostname or "").lower().strip("[]")
    except Exception:
        host = ""
    if not host:
        return "UNKNOWN:" + (normalized or s.strip())[:128]
    # IP -> host group
    try:
        ipaddress.ip_address(host.strip("[]"))
        return host
    except Exception:
        pass
    cached = _DOMAIN_CACHE.get(host)
    if cached is not None:
        return cached
    try:
        import tldextract

        ext = tldextract.extract(s, include_psl_private_domains=True)
        dom = getattr(ext, "top_domain_under_public_suffix", "") or ""
        if dom:
            out = dom.lower()
            _DOMAIN_CACHE[host] = out
            return out
    except Exception:
        pass
    parts = [x for x in host.split(".") if x]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host
