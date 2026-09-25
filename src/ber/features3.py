"""Retrieval-aware pair, context and name-word odds features on top of ``ber.features``.

IDF comes from the target-corpus token index, so rare shared evidence counts more than
common words.  House-number features separate digit noise in true copies (3207 vs 4207,
00239 vs 239) from the small forward shifts used by near-duplicate sibling businesses
(818 vs 821).  Context features compare a candidate with the other candidates of the same
query.  Odds features score name words present on only one side ("Midtown", "Towing"
mark siblings; "Center", "Services" are copy noise), learned from training pairs only.
Record IDs are never inputs.
"""
import math

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from .blocking import key_hash
from .features import FEATURE_NAMES as BASE_NAMES, pair_features, prepare_record
from .retrieval import token_keys
from .tokens import NATIVE, address_parts, is_domain, latin, name_parts

PAIR_NAMES = ("n_idf_shared", "n_cov_q", "n_cov_t", "n_idf_extra", "n_idf_missing", "n_fuzzy_q", "n_fuzzy_t",
              "n_tsr", "n_tset", "n_concat_ratio", "n_concat_partial", "t_domain", "t_native", "t_empty_addr",
              "a_idf_shared", "a_cov_q", "a_cov_t", "a_idf_extra", "num_shared", "num_jacc", "num_idf_shared",
              "house_eq", "house_in", "house_logdiff", "house_edit", "house_shift", "n_extra_cnt", "n_missing_cnt")
CONTEXT_NAMES = ("rank_all", "score_all", "score_rel", "channel", "score_name", "ctx_near_dups", "ctx_same_house_as_t",
                 "ctx_house_eq_q", "ctx_best_other_tsr", "ctx_rank_tsr", "ctx_rank_addr", "ctx_n")
ODDS_NAMES = ("odds_extra_min", "odds_extra_max", "odds_extra_mean", "odds_miss_min", "odds_miss_mean")
FEATURE_NAMES = tuple(BASE_NAMES) + PAIR_NAMES + CONTEXT_NAMES
MODEL_NAMES = FEATURE_NAMES + ODDS_NAMES
_TSR, _ADDR = PAIR_NAMES.index("n_tsr"), PAIR_NAMES.index("a_idf_shared")


def prepare(row, index):
    """Base preparation plus token sets carrying IDF from the target-corpus ``index``."""
    out = prepare_record(row)
    if "_v3" in out:
        return out
    out = dict(out)
    country = latin(row.get("country", ""))
    _, _, core = name_parts(row.get("business_name", ""))
    words, numbers = address_parts(row.get("business_address", ""))
    core_set, word_set, number_set = set(core), set(words), set(numbers)
    keys = ([f"{country}|n|{t}" for t in core_set] + [f"{country}|a|{t}" for t in word_set]
            + [f"{country}|h|{t}" for t in number_set])
    df, _ = index.document_frequency(token_keys(keys))
    idf = index.idf(df).tolist()
    a, b = len(core_set), len(core_set) + len(word_set)
    name = row.get("business_name", "") or ""
    out.update(_v3=True, _c=core_set, _cseq=" ".join(core), _cat="".join(core),
               _cidf=dict(zip(core_set, idf[:a])), _w=word_set, _widf=dict(zip(word_set, idf[a:b])),
               _n=number_set, _nidf=dict(zip(number_set, idf[b:])), _first=numbers[0] if numbers else "",
               _domain=is_domain(name), _native=bool(NATIVE.search(name)),
               _empty=not str(row.get("business_address", "") or "").strip())
    return out


def _pair(q, t):
    sh = q["_c"] & t["_c"]
    s = sum(q["_cidf"][x] for x in sh)
    qi, ti = sum(q["_cidf"].values()) or 1e-9, sum(t["_cidf"].values()) or 1e-9
    if q["_c"] and t["_c"]:
        fq = sum(q["_cidf"][x] * max(fuzz.ratio(x, y) for y in t["_c"]) for x in q["_c"]) / (100 * qi)
        ft = sum(t["_cidf"][y] * max(fuzz.ratio(x, y) for x in q["_c"]) for y in t["_c"]) / (100 * ti)
        tsr = fuzz.token_sort_ratio(q["_cseq"], t["_cseq"]) / 100
        tset = fuzz.token_set_ratio(q["_cseq"], t["_cseq"]) / 100
    else:
        fq = ft = tsr = tset = 0.
    both = bool(q["_cat"] and t["_cat"])
    cr = fuzz.ratio(q["_cat"], t["_cat"]) / 100 if both else 0.
    cp = fuzz.partial_ratio(t["_cat"], q["_cat"]) / 100 if both else 0.
    ash = q["_w"] & t["_w"]
    a = sum(q["_widf"][x] for x in ash)
    aq, at = sum(q["_widf"].values()) or 1e-9, sum(t["_widf"].values()) or 1e-9
    nsh, nun = q["_n"] & t["_n"], q["_n"] | t["_n"]
    qh = q["_first"]
    house_in = float(bool(qh) and qh in t["_n"])
    if qh and t["_n"]:
        d = min(abs(int(qh[:9]) - int(x[:9])) for x in t["_n"])
        edit = min(Levenshtein.distance(qh, x) for x in t["_n"])
    else:
        d = edit = -1
    return [s, s / qi, s / ti, ti - s, qi - s, fq, ft, tsr, tset, cr, cp, float(t["_domain"]), float(t["_native"]),
            float(t["_empty"]), a, a / aq, a / at, at - a, len(nsh), len(nsh) / max(1, len(nun)),
            sum(q["_nidf"][x] for x in nsh), float(bool(qh) and qh == t["_first"]), house_in,
            math.log1p(d) if d >= 0 else -1., edit, float(0 < d <= 40 and not house_in),
            len(t["_c"] - q["_c"]), len(q["_c"] - t["_c"])]


def query_features(query, targets, score_all, score_name, channel):
    """FEATURE_NAMES rows for one prepared query and its prepared candidates, plus the
    name words only the target has (extra) and only the query has (missing) per pair."""
    if not targets:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), [], []
    base = np.asarray([pair_features(query, t) for t in targets], np.float32).reshape(-1, len(BASE_NAMES))
    pair = np.asarray([_pair(query, t) for t in targets], np.float32)
    n = len(targets)
    best = float(np.max(score_all)) or 1.
    tsr, addr = pair[:, _TSR], pair[:, _ADDR]
    rank_tsr = np.empty(n)
    rank_tsr[np.argsort(-tsr, kind="stable")] = np.arange(n)
    rank_addr = np.empty(n)
    rank_addr[np.argsort(-addr, kind="stable")] = np.arange(n)
    houses = {}
    for t in targets:
        if t["_first"]:
            houses[t["_first"]] = houses.get(t["_first"], 0) + 1
    same_q = sum(1 for t in targets if query["_first"] and t["_first"] == query["_first"])
    top = np.sort(tsr)[::-1]
    first, second = top[0], (top[1] if n > 1 else 0.)
    context = np.column_stack([
        np.arange(n), score_all, np.asarray(score_all) / best, channel, score_name,
        np.full(n, (tsr >= .85).sum()), [houses.get(t["_first"], 0) if t["_first"] else 0 for t in targets],
        np.full(n, same_q), np.where(tsr == first, second, first), rank_tsr, rank_addr, np.full(n, n)]).astype(np.float32)
    extra = [sorted(t["_c"] - query["_c"]) for t in targets]
    missing = [sorted(query["_c"] - t["_c"]) for t in targets]
    return np.hstack([base, pair, context]), extra, missing


def word_keys(words, side):
    """Flat keys and per-pair counts for lists of one-sided name words; side is 'e' or 'm'."""
    keys = [key_hash(f"{side}|{w}") for ws in words for w in ws]
    return np.asarray(keys, dtype=np.uint64), np.asarray([len(ws) for ws in words], dtype=np.int32)


class TokenOdds:
    """Smoothed log-odds, relative to the prior, that a pair matches given a one-sided name word."""

    def __init__(self, keys, values):
        self.keys, self.values = np.asarray(keys, np.uint64), np.asarray(values, np.float32)

    @classmethod
    def fit(cls, extra, extra_counts, missing, missing_counts, labels, strength=20.):
        labels = np.asarray(labels, np.float64)
        keys = np.r_[extra, missing].astype(np.uint64)
        per_key = np.r_[np.repeat(labels, extra_counts), np.repeat(labels, missing_counts)]
        prior = float(np.clip(labels.mean() if len(labels) else .5, 1e-6, 1 - 1e-6))
        if not len(keys):
            return cls(np.empty(0, np.uint64), np.empty(0, np.float32))
        unique, inverse = np.unique(keys, return_inverse=True)
        total = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        positive = np.bincount(inverse, weights=per_key, minlength=len(unique))
        values = (np.log((positive + strength * prior) / (total - positive + strength * (1 - prior)))
                  - math.log(prior / (1 - prior)))
        return cls(unique, values)

    def _lookup(self, keys):
        out = np.zeros(len(keys), np.float32)
        if len(keys) and len(self.keys):
            pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
            hit = self.keys[pos] == keys
            out[hit] = self.values[pos[hit]]
        return out

    @staticmethod
    def _reduce(values, counts):
        n = len(counts)
        low, high, mean = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
        nonempty = counts > 0
        if nonempty.any():
            starts = np.r_[0, np.cumsum(counts)[:-1]][nonempty]
            low[nonempty] = np.minimum.reduceat(values, starts)
            high[nonempty] = np.maximum.reduceat(values, starts)
            mean[nonempty] = np.add.reduceat(values, starts) / counts[nonempty]
        return low, high, mean

    def features(self, extra, extra_counts, missing, missing_counts):
        e_low, e_high, e_mean = self._reduce(self._lookup(extra), np.asarray(extra_counts))
        m_low, _, m_mean = self._reduce(self._lookup(missing), np.asarray(missing_counts))
        return np.column_stack([e_low, e_high, e_mean, m_low, m_mean]).astype(np.float32)

    def save(self, path):
        np.savez(path, keys=self.keys, values=self.values)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["keys"], data["values"])
