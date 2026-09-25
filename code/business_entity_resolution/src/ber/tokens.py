"""Order-free retrieval tokens; rules and the learned table use supplied training data only.

Native-script names are token-for-token transliterations of the reference name, so a
dictionary learned from training pairs (``ber.translit``) maps them back before AnyAscii.
Tokens are country-scoped: ``country|kind|value`` with kind n=name, c=concatenated name
(domain/hashtag forms), a=address word, h=house/plot number.
"""
import json
import os
import re
from functools import lru_cache
from anyascii import anyascii

NATIVE = re.compile(r"[\u0900-\u0DFF]")
_RAW = re.compile(r"[\w\u0900-\u0DFF]+", re.UNICODE)
_ALNUM = re.compile(r"\d+|[a-z]+")
_DOMAIN = re.compile(r"\.(com|net|org|in|co|biz|info|fr)\b")
_ALIAS = re.compile(r"\b(?:formerly known as|formerly|f/k/a|fka|doing business as|d/b/a|dba|a/k/a|aka|t/a|trading as)\b:?", re.I)
LEGAL = frozenset("private limited pvt ltd llc inc incorporated corp corporation co company llp lp plc pllc pc pa the and of "
                  "sarl sas sasu eurl sa sci gmbh com net org www in biz info m s dr smt shri sri mr mrs ms".split())
ABBR = dict(pair.split(":") for pair in (
    "st:street rd:road ave:avenue av:avenue blvd:boulevard ln:lane dr:drive ct:court cir:circle hwy:highway "
    "pkwy:parkway ter:terrace trl:trail pl:place sq:square mt:mount ft:fort n:north s:south e:east w:west "
    "nr:near opp:opposite marg:road r:rue bd:boulevard rte:route che:chemin imp:impasse all:allee").split())
_ORDINAL = re.compile(r"(\d)(?:st|nd|rd|th)\b")
# E.U.R.L. / L.L.C. / [S.A.S] -> EURL / LLC / SAS, so dotted legal forms match LEGAL.
_DOTTED = re.compile(r"(?<![^\W\d_])((?:[^\W\d_]\.){2,}[^\W\d_]?)(?![^\W\d_])")
# "N° 86": the degree sign is not a word character, so a bare "n" would become "north".
_NUMERO = re.compile(r"\bn\s*°", re.I)
ENV = "BER_TRANSLIT"


@lru_cache(maxsize=1)
def _tables():
    path = os.environ.get(ENV)
    if not path or not os.path.exists(path):
        return {}, {}
    table = json.loads(open(path, encoding="utf-8").read())
    return table["name"], {**table["name"], **table["addr"]}


def configure(path):
    """Point this process and its future workers at a learned transliteration table."""
    os.environ[ENV] = str(path)
    _tables.cache_clear()


def latin(text, address=False):
    table = _tables()[1 if address else 0]
    words = [table.get(w, w) if NATIVE.search(w) else w for w in _RAW.findall(str(text or "").lower())]
    return anyascii(" ".join(words)).lower()


def name_parts(name):
    """Return (tokens, concatenated stems, core tokens) over alias variants of one name."""
    variants = [v for v in _ALIAS.split(str(name or "")) if v.strip()] or [""]
    tokens, stems = [], []
    for variant in variants:
        variant = _DOTTED.sub(lambda m: m.group(1).replace(".", ""), variant.replace("&", " and "))
        words = _ALNUM.findall(latin(variant))
        tokens += words
        stem = "".join(w for w in words if w not in LEGAL)
        if len(stem) >= 6:
            stems.append(stem)
    return tokens, stems, [t for t in tokens if t not in LEGAL]


def address_parts(address):
    """Return (words, numbers in order); numbers lose leading zeros, words expand abbreviations."""
    words, numbers = [], []
    raw = _NUMERO.sub(" no ", str(address or "").replace("null", " ").replace("NULL", " "))
    text = _ORDINAL.sub(r"\1", latin(raw, address=True))
    for t in _ALNUM.findall(text):
        if t.isdigit():
            numbers.append(t.lstrip("0") or "0")
        else:
            words.append(ABBR.get(t, t))
    return words, numbers


def is_domain(name):
    name = str(name or "").strip()
    return bool(_DOMAIN.search(name.lower())) or (name[:1] in "#@" and " " not in name)


def record_tokens(record):
    country = latin(record.get("country", ""))
    tokens, stems, _ = name_parts(record.get("business_name", ""))
    words, numbers = address_parts(record.get("business_address", ""))
    out = {f"{country}|n|{t}" for t in tokens} | {f"{country}|c|{t}" for t in stems}
    out |= {f"{country}|a|{t}" for t in words} | {f"{country}|h|{t}" for t in numbers}
    return sorted(out)
