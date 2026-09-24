"""Unicode-safe normalization; rules contain no external business identity data."""
import re
import unicodedata
from anyascii import anyascii

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
LEGAL = frozenset("ltd limited llc inc incorporated corp corporation co company plc pvt private llp sarl sas sa sci eurl pte bv gmbh and the".split())
ADDRESS_MAP = dict(pair.split(":") for pair in (
    "rd:road st:street ave:avenue av:avenue blvd:boulevard ln:lane dr:drive ct:court "
    "cir:circle hwy:highway pkwy:parkway ste:suite apt:apartment fl:floor flr:floor "
    "n:north s:south e:east w:west nr:near opp:opposite marg:road r:rue bd:boulevard "
    "boul:boulevard rte:route imp:impasse che:chemin pl:place res:residence"
).split())
ADDRESS_COMMON = frozenset(ADDRESS_MAP.values()) | frozenset(
    "no number near opposite unit building plot sector block floor de du des la le les et a au aux".split())


def normalize(value):
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    out = []
    latin_base = False
    for ch in text:
        if not unicodedata.combining(ch):
            latin_base = "LATIN" in unicodedata.name(ch, "")
        if unicodedata.combining(ch) and latin_base:
            continue
        # Preserve Indic marks, which Python's \w does not match.
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M"):
            out.append(ch)
        elif ch == "&":
            out.append(" and ")
        else:
            out.append(" ")
    return _SPACE.sub(" ", "".join(out)).strip()


def latin(value):
    return normalize(anyascii(str(value or "")))


def canonical(value):
    return " ".join(t for t in latin(value).split() if t not in LEGAL)


def address(value):
    return " ".join(ADDRESS_MAP.get(t, t) for t in latin(value).split())


def tokens(value):
    return normalize(value).split()


def salient_tokens(value, min_len=3):
    return sorted({t for t in value.split() if len(t) >= min_len and not t.isdigit() and t not in LEGAL},
                  key=lambda token: (-len(token), token))
