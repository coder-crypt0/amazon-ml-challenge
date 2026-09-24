"""Country-agnostic features; record IDs are never model inputs."""
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from .text import normalize, latin, canonical, address

SIM_NAMES = ("ratio", "token_sort", "token_set", "partial", "jaro", "length_ratio", "equal")
FEATURE_NAMES = tuple(f"{field}_{metric}" for field in ("name", "core", "address") for metric in SIM_NAMES) + (
    "unicode_name_ratio", "name_token_jaccard", "name_token_containment", "address_token_jaccard",
    "address_token_containment", "address_digit_jaccard", "address_digit_overlap", "address_digit_conflict",
    "house_equal", "house_conflict", "postcode_equal", "postcode_conflict", "name_digit_jaccard",
    "left_name_tokens", "right_name_tokens", "left_address_tokens", "right_address_tokens",
    "left_name_missing", "right_name_missing", "left_address_missing", "right_address_missing",
    "country_equal", "name_address_min", "name_address_product", "initials_equal", "name_fuzzy_tokens")


def prepare_record(row):
    if "_core" in row:
        return row
    out = dict(row)
    out["_name"] = latin(row.get("business_name", ""))
    out["_unicode"] = normalize(row.get("business_name", ""))
    out["_core"] = canonical(row.get("business_name", ""))
    out["_address"] = address(row.get("business_address", ""))
    for field in ("name", "core", "address"):
        out[f"_{field}_tokens"] = frozenset(out[f"_{field}"].split())
        out[f"_{field}_sorted"] = " ".join(sorted(out[f"_{field}_tokens"]))
    out["_nd"] = frozenset(re.findall(r"\d+", out["_name"]))
    numbers = re.findall(r"\d+", out["_address"])
    out["_ad"] = frozenset(numbers)
    out["_house"] = numbers[0] if numbers else ""
    out["_postcode"] = frozenset(t for t in numbers if len(t) in (5, 6))
    out["_initials"] = "".join(t[0] for t in out["_core"].split())
    return out


def _overlap(x, y):
    intersection = len(x & y)
    return intersection / max(1, len(x | y)), intersection / max(1, min(len(x), len(y)))


def preliminary_similarity(left, right):
    left, right = prepare_record(left), prepare_record(right)
    n = fuzz.ratio(left["_core_sorted"], right["_core_sorted"]) / 100 if left["_core"] and right["_core"] else 0
    a = fuzz.token_set_ratio(left["_address"], right["_address"]) / 100 if left["_address"] and right["_address"] else 0
    # A missing field must not suppress an otherwise excellent name match.
    return .55*n + .35*a + .1*min(n, a) if left["_address"] and right["_address"] else .75*n


def _sim(left, right, field):
    a, b = left["_"+field], right["_"+field]
    if not a or not b:
        return [0.] * len(SIM_NAMES)
    return [fuzz.ratio(a,b)/100, fuzz.ratio(left["_"+field+"_sorted"],right["_"+field+"_sorted"])/100,
            fuzz.token_set_ratio(a,b)/100, fuzz.partial_ratio(a,b)/100,
            JaroWinkler.normalized_similarity(a,b), min(len(a),len(b))/max(len(a),len(b)), float(a==b)]


def pair_features(left, right):
    l, r = prepare_record(left), prepare_record(right)
    ns, cs, ads = _sim(l,r,"name"), _sim(l,r,"core"), _sim(l,r,"address")
    nd, ad = _overlap(l["_nd"],r["_nd"]), _overlap(l["_ad"],r["_ad"])
    house_available = bool(l["_house"] and r["_house"])
    post_available = bool(l["_postcode"] and r["_postcode"])
    nt, at = _overlap(l["_core_tokens"], r["_core_tokens"]), _overlap(l["_address_tokens"],r["_address_tokens"])
    short, long = sorted((l["_core_tokens"], r["_core_tokens"]), key=len)
    fuzzy = sum(max((fuzz.ratio(x,y) for y in long), default=0) for x in short) / (100*max(1,len(short)))
    return [*ns, *cs, *ads,
        fuzz.ratio(l["_unicode"],r["_unicode"])/100 if l["_unicode"] and r["_unicode"] else 0,
        *nt, *at, *ad, float(bool(l["_ad"] and r["_ad"]) and not l["_ad"] & r["_ad"]),
        float(house_available and l["_house"]==r["_house"]), float(house_available and l["_house"]!=r["_house"]),
        float(post_available and bool(l["_postcode"] & r["_postcode"])),
        float(post_available and not l["_postcode"] & r["_postcode"]), nd[0],
        len(l["_core_tokens"]),len(r["_core_tokens"]),len(l["_address_tokens"]),len(r["_address_tokens"]),
        float(not l["business_name"]),float(not r["business_name"]),
        float(not l["business_address"]),float(not r["business_address"]),
        float(l["country"]==r["country"]),min(cs[1],ads[2]),cs[1]*ads[2],
        float(bool(l["_initials"]) and l["_initials"]==r["_initials"]),fuzzy]


def extract_features(left_records, right_records, pairs):
    return np.asarray([pair_features(left_records[int(i)],right_records[int(j)]) for i,j in pairs],dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
