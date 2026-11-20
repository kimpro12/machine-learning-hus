"""PhishVN-compatible lexical feature extractor (scheme-stripped, CompPhish-aligned).

Rules replicate data/phishvn/docs/schema.md + the released vn_compphish.csv values:
- All length/count/entropy features are computed on the SCHEME-STRIPPED URL.
- `is_https` is a collection artefact (kept in the CSV for CompPhish compatibility)
  and is EXCLUDED from modelling — it is not part of PHISHVN_FEATURES.
- TLD = public suffix with private domains OFF (e.g. vercel.app -> suffix 'app',
  dom 'vercel.app'), except Vietnamese 2nd-level suffixes: only
  {biz,com,edu,gov,info,net,org,pro}.vn count as 2-level; any other X.vn
  collapses to tld='vn' (matches the dataset snapshot).
- Unknown suffix (tldextract finds nothing): dom=full host, tld='', tld_len=0,
  subdom_cnt = labels-1 (matches the 2 such rows in the release).
- path_len excludes a trailing '/' (dataset convention).
"""
from __future__ import annotations
import ipaddress
import math
from collections import Counter
from urllib.parse import urlsplit

import tldextract as _tldextract

# 21 modelling features (22-column CompPhish schema minus is_https).
PHISHVN_FEATURES = [
 'url_len','dom_len','is_ip','tld_len','subdom_cnt','letter_cnt','digit_cnt',
 'special_cnt','eq_cnt','qm_cnt','amp_cnt','dot_cnt','dash_cnt','under_cnt',
 'letter_ratio','digit_ratio','spec_ratio','slash_cnt','entropy','path_len','query_len',
]

# Vietnamese 2nd-level suffixes treated as 2-level TLDs by the dataset snapshot.
KNOWN_VN_2LD = frozenset(['biz.vn','com.vn','edu.vn','gov.vn','info.vn','net.vn','org.vn','pro.vn'])

# Single shared extractor: public PSL only (private domains OFF, matching the
# release), snapshot cached by tldextract after first use. Constructed once so
# repeated calls don't re-fetch/re-parse the suffix list.
_TLD_EXTRACTOR = _tldextract.TLDExtract(
    cache_dir=None,
    suffix_list_urls=None,
    include_psl_private_domains=False,
)


def strip_scheme(u: str) -> str:
    s = (u or '').strip()
    low = s.lower()
    if low.startswith('http://'):
        return s[7:]
    if low.startswith('https://'):
        return s[8:]
    return s


def _host_of(url: str) -> str:
    s = (url or '').strip()
    if '://' not in s:
        s = 'http://' + s
    try:
        p = urlsplit(s)
        return (p.hostname or '').lower()
    except Exception:
        return ''


def _split_host(host: str):
    """Return (registrable_domain, tld) following the dataset conventions."""
    h = (host or '').lower()
    if not h:
        return ('', '')
    try:
        ipaddress.ip_address(h.strip('[]'))
        return (h, '')
    except Exception:
        pass
    try:
        ext = _TLD_EXTRACTOR(h)
        suffix = (ext.suffix or '').lower()
        reg = (ext.top_domain_under_public_suffix or '').lower()
    except Exception:
        suffix, reg = '', ''
    if not suffix:
        # Unknown suffix: dom = full host, tld empty (matches release rows).
        return (h, '')
    if suffix.endswith('.vn') and suffix != 'vn' and suffix not in KNOWN_VN_2LD:
        # Newer-PSL city-level .vn suffixes collapse to 'vn' per dataset snapshot.
        suffix = 'vn'
        parts = [x for x in h.split('.') if x]
        reg = '.'.join(parts[-2:]) if len(parts) >= 2 else h
    if not reg:
        parts = [x for x in h.split('.') if x]
        reg = '.'.join(parts[-2:]) if len(parts) >= 2 else h
    return (reg, suffix)


def shannon(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    f = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in f.values()))


def phishvn_features(url: object) -> dict:
    raw = '' if url is None else str(url)
    s = strip_scheme(raw)
    n = len(s)
    host = _host_of(raw)
    reg, tld = _split_host(host)
    is_ip = 0
    try:
        ipaddress.ip_address(host.strip('[]'))
        is_ip = 1
    except Exception:
        is_ip = 0
    if is_ip or not host or not reg:
        sub = 0
    elif not tld:
        # Unknown-suffix convention (release rows): subdom = labels - 1.
        sub = max(0, len([x for x in host.split('.') if x]) - 1)
    elif host == reg:
        sub = 0
    elif host.endswith('.' + reg):
        sub = len([x for x in host[:-(len(reg) + 1)].split('.') if x])
    else:
        parts = [x for x in host.split('.') if x]
        sub = max(0, len(parts) - 2)
    qlen = 0
    plen = 0
    if '?' in s:
        before_q, after_q = s.split('?', 1)
        qlen = len(after_q.split('#')[0])
        rest = before_q
    else:
        rest = s.split('#')[0]
    if '/' in rest:
        # Dataset convention: trailing '/' does not count toward path_len.
        plen = len(rest.split('/', 1)[1].rstrip('/'))
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    special = n - letters - digits
    return {
     'url_len': float(n),
     'dom_len': float(len(reg)),
     'is_ip': float(is_ip),
     'tld_len': float(len(tld)),
     'subdom_cnt': float(sub),
     'letter_cnt': float(letters),
     'digit_cnt': float(digits),
     'special_cnt': float(special),
     'eq_cnt': float(s.count('=')),
     'qm_cnt': float(s.count('?')),
     'amp_cnt': float(s.count('&')),
     'dot_cnt': float(s.count('.')),
     'dash_cnt': float(s.count('-')),
     'under_cnt': float(s.count('_')),
     'letter_ratio': float(letters / n) if n else 0.0,
     'digit_ratio': float(digits / n) if n else 0.0,
     'spec_ratio': float(special / n) if n else 0.0,
     'slash_cnt': float(s.count('/')),
     'entropy': float(shannon(s)),
     'path_len': float(plen),
     'query_len': float(qlen),
    }
