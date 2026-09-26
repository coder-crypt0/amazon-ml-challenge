#!/usr/bin/env python3
"""BER v5: vectorized business entity resolution (Amazon ML Challenge 2026).

    python ber5.py --data DATA_DIR --work WORK_DIR [--stage all|train|test] [--cfg JSON]

DATA_DIR holds train/ and test/ with the official TSV files.

train  learn a native-script -> Latin token table from train pairs; build the train graph with
       19% of S1 removed (their S2/S3 copies become distractors, matching the higher distractor
       density of test); retrieve candidates (target->S1 reverse view + S1->target combo/name/
       address views, numba sparse top-k over country-scoped TF-IDF tokens); featurize a sample
       of S1 entities; cross-fit two LightGBM models; tune the threshold on out-of-fold
       predictions under target exclusivity; report macro F0.5 on a held-out validation slice.
test   same retrieval and features on the full test graph, average the two models, keep each
       target only for its best S1, apply the threshold, write output/matching_results.tsv and
       output/candidate_pairs.tsv (the exact pairs the model scored).

Only the supplied data is used: no external data, no pretrained weights.
"""
import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii
from numba import njit, prange, set_num_threads

CFG = {
    "drop": 0.19,        # share of train S1 removed from the graph (test-like distractor density)
    "sample": 0.30,      # share of train S1 whose candidate pairs are featurized
    "val_share": 0.2,    # part of the sample held out for the final validation report
    "k_rev": 8, "k_fwd": 12, "k_name": 6, "k_addr": 6,
    "cap_s1": 20000, "cap_t": 50000,  # postings longer than this are skipped during retrieval
    "budget_s1": 10000, "budget_t": 30000,  # per-query posting budget per field group, rarest tokens first
    "wide": 4,           # candidate pool = wide x k by partial cosine, re-ranked by exact cosine
    "skip2": 0.005,      # stage 2 re-scores only pairs with stage-1 p >= skip2 (others keep p1)
    "rev_min": 0.1,
    "neg_keep": 0.3,     # share of easy negatives kept for training (weighted back up)
    "chunk": 3_000_000,
    "lr": 0.06, "leaves": 255, "min_leaf": 100, "rounds": 4000, "es": 100,
    "seed": 20260926,
    "translit_pairs": 600_000,
}
NT = os.cpu_count() or 4
T0 = time.time()
DEADLINE = float("inf")  # wall-clock limit (epoch seconds); scoring stops early and still writes outputs


def log(*args):
    try:
        import psutil
        mem = f"{psutil.Process().memory_info().rss / 2**30:5.1f}G"
    except Exception:
        mem = ""
    print(f"[{time.time() - T0:7.0f}s {mem}]", *args, flush=True)


# ----------------------------------------------------------------------------- normalization
NATIVE = re.compile(r"[ऀ-ൿ]")
_RAW = re.compile(r"[\wऀ-ൿ]+")
_ALNUM = re.compile(r"[a-z]+|\d+")
_ALIAS = re.compile(r"\b(?:formerly known as|formerly|f/k/a|fka|doing business as|d/b/a|dba|a/k/a|aka|"
                    r"t/a|trading as)\b:?", re.I)
_DOTTED = re.compile(r"(?<![^\W\d_])((?:[^\W\d_]\.){2,}[^\W\d_]?)(?![^\W\d_])")
_DOMAIN = re.compile(r"www\.|\.(?:com|net|org|in|co|biz|info|fr)\b", re.I)
_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnone\b", re.I)
_NUMERO = re.compile(r"\bn\s*°", re.I)
_ORDINAL = re.compile(r"(\d)(?:st|nd|rd|th)\b")
_REPEAT = re.compile(r"(.)\1+")
_VOWELS = str.maketrans("", "", "aeiouy")
LEET = str.maketrans("013457", "oieast")
LEGAL = frozenset(
    "private limited pvt ltd llc inc incorporated corp corporation co company llp lp plc pllc pc pa the and of "
    "sarl sas sasu eurl sa sci snc selarl ei scp gie scop sel ets gmbh com net org www in biz info m s dr smt shri sri "
    "mr mrs ms".split())
# Normalization v1 (models trained before 26 Sep 18:00) expands St -> street and Ste -> suite; v2 maps
# Street/Saint -> st and Suite/Sainte -> ste, so French "St-Denis"/"Saint-Denis" (and "St. Louis") agree.
ABBR_V1 = dict(p.split(":") for p in (
    "st:street str:street rd:road ave:avenue av:avenue blvd:boulevard ln:lane dr:drive ct:court cir:circle "
    "hwy:highway pkwy:parkway ter:terrace trl:trail pl:place sq:square mt:mount ft:fort n:north s:south "
    "e:east w:west ne:northeast nw:northwest se:southeast sw:southwest nr:near opp:opposite marg:road "
    "r:rue bd:boulevard rte:route che:chemin ch:chemin imp:impasse fbg:faubourg ste:suite apt:apartment "
    "fl:floor bldg:building ctr:center cntr:center expy:expressway fwy:freeway jct:junction jn:junction "
    "sec:sector stn:station rly:railway ngr:nagar hno:no dno:no").split())
ABBR = dict(p.split(":") for p in (
    "street:st str:st saint:st sainte:ste suite:ste rd:road ave:avenue av:avenue blvd:boulevard bld:boulevard "
    "boul:boulevard ln:lane dr:drive ct:court cir:circle qu:quai crs:cours chem:chemin mte:montee res:residence "
    "hwy:highway pkwy:parkway ter:terrace trl:trail pl:place sq:square mt:mount ft:fort n:north s:south "
    "e:east w:west ne:northeast nw:northwest se:southeast sw:southwest nr:near opp:opposite marg:road "
    "r:rue bd:boulevard rte:route che:chemin ch:chemin imp:impasse fbg:faubourg apt:apartment "
    "fl:floor bldg:building ctr:center cntr:center expy:expressway fwy:freeway jct:junction jn:junction "
    "sec:sector stn:station rly:railway ngr:nagar hno:no dno:no").split())
# Full state names -> postal codes (generic address normalization; both spellings occur in the data).
STATES = dict(p.split(":") for p in (
    "alabama:al alaska:ak arizona:az arkansas:ar california:ca colorado:co connecticut:ct delaware:de "
    "florida:fl georgia:ga hawaii:hi idaho:id illinois:il indiana:in iowa:ia kansas:ks kentucky:ky "
    "louisiana:la maine:me maryland:md massachusetts:ma michigan:mi minnesota:mn mississippi:ms "
    "missouri:mo montana:mt nebraska:ne nevada:nv new_hampshire:nh new_jersey:nj new_mexico:nm "
    "new_york:ny north_carolina:nc north_dakota:nd ohio:oh oklahoma:ok oregon:or pennsylvania:pa "
    "rhode_island:ri south_carolina:sc south_dakota:sd tennessee:tn texas:tx utah:ut vermont:vt "
    "virginia:va washington:wa west_virginia:wv wisconsin:wi wyoming:wy district_of_columbia:dc "
    "andhra_pradesh:ap arunachal_pradesh:ar assam:as bihar:br chhattisgarh:cg goa:ga gujarat:gj "
    "haryana:hr himachal_pradesh:hp jharkhand:jh karnataka:ka kerala:kl madhya_pradesh:mp "
    "maharashtra:mh manipur:mn meghalaya:ml mizoram:mz nagaland:nl odisha:od orissa:od punjab:pb "
    "rajasthan:rj sikkim:sk tamil_nadu:tn tamilnadu:tn telangana:ts tripura:tr uttar_pradesh:up "
    "uttarakhand:uk west_bengal:wb delhi:dl jammu_and_kashmir:jk puducherry:py chandigarh:ch").split())
STATES = {k.replace("_", " "): v for k, v in STATES.items()}
_STATES_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATES), key=len, reverse=True)) + r")\b")
# Token fields: 0 name word, 1 name skeleton, 2 whole concatenated name, 3 its 6-char prefix,
# 4 address word, 5 address number.  Fields 0-3 form the name vector, 4-5 the address vector.
FIELD_W = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 1.0])
_TAB = ({}, {}, None)


def _init_worker(tables):
    global _TAB
    _TAB = tables


def _latin(text, table):
    words = []
    for w in _RAW.findall(text.lower()):
        if NATIVE.search(w):
            w = table.get(w, w)
        words.append(w)
    return anyascii(" ".join(words)).lower()


def norm_name(raw, table):
    raw = raw or ""
    toks = []
    for part in _ALIAS.split(raw):
        if not part.strip():
            continue
        part = _DOTTED.sub(lambda m: m.group(1).replace(".", ""), part.replace("&", " and "))
        for w in _latin(part, table).split():
            if len(w) > 2 and any(c.isdigit() for c in w) and sum(c.isalpha() for c in w) >= 2:
                w = w.translate(LEET)
            toks.extend(_ALNUM.findall(w))
    core = [t for t in toks if t not in LEGAL]
    return toks, core


def norm_addr(raw, table, abbr=None):
    abbr = ABBR if abbr is None else abbr
    raw = _NUMERO.sub(" no ", _NULLS.sub(" ", raw or ""))
    text = _ORDINAL.sub(r"\1", _latin(raw, table))
    text = _STATES_RE.sub(lambda m: STATES[m.group(0)], text)
    seq, words, nums = [], [], []
    for t in _ALNUM.findall(text):
        if t[0].isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
        else:
            t = abbr.get(t, t)
            words.append(t)
        seq.append(t)
    return seq, words, nums


def skeleton(t):
    return _REPEAT.sub(r"\1", t[0] + t[1:].translate(_VOWELS))


def _norm_chunk(args):
    names, addrs, countries = args
    tname, taddr, abbr = _TAB
    n = len(names)
    toks_all, fields, counts = [], [], np.zeros(n, np.int32)
    nm, core_s, cat_s, ad, hn, ini = [], [], [], [], [], []
    nums = np.full((n, 4), -1, np.int32)
    flags = np.zeros((n, 8), np.int16)
    for i in range(n):
        rn, ra, c = names[i] or "", addrs[i] or "", countries[i]
        toks, core = norm_name(rn, tname)
        seq, words, numl = norm_addr(ra, taddr, abbr)
        base = core or toks
        cat = "".join(base)
        before = len(toks_all)
        for t in set(base):
            toks_all.append(f"{c}|n|{t}"); fields.append(0)
        for s in {skeleton(t) for t in base if len(t) >= 4}:
            toks_all.append(f"{c}|k|{s}"); fields.append(1)
        if len(cat) >= 5:
            toks_all.append(f"{c}|c|{cat}"); fields.append(2)
        if len(cat) >= 7:
            toks_all.append(f"{c}|p|{cat[:6]}"); fields.append(3)
        for w in set(words):
            if len(w) >= 2:
                toks_all.append(f"{c}|a|{w}"); fields.append(4)
        for x in set(numl):
            toks_all.append(f"{c}|h|{x}"); fields.append(5)
        counts[i] = len(toks_all) - before
        nm.append(" ".join(toks)); core_s.append(" ".join(core)); cat_s.append(cat)
        ini.append("".join(t[0] for t in base))
        ad.append(" ".join(seq)); hn.append(numl[0] if numl else "")
        for j, x in enumerate(numl[:4]):
            nums[i, j] = int(x[:9])
        flags[i] = (bool(NATIVE.search(rn)), bool(_ALIAS.search(rn)),
                    bool(_DOMAIN.search(rn)) or rn[:1] in "@#", not (words or numl),
                    bool(NATIVE.search(ra)), len(core), len(nm[-1]), len(ad[-1]))
    hashes = pd.util.hash_array(np.array(toks_all, dtype=object)) if toks_all else np.zeros(0, np.uint64)
    return hashes, np.array(fields, np.int8), counts, nm, core_s, cat_s, ad, hn, nums, flags, ini


def _core_chunk(args):
    """Address without the country's common words (region/department/city/street type); numbers kept."""
    countries, ads, common = args
    words = [a.split() for a in ads]
    toks, pos = [], []
    for i, (c, ws) in enumerate(zip(countries, words)):
        for j, w in enumerate(ws):
            if not w[0].isdigit():
                toks.append(f"{c}|a|{w}")
                pos.append((i, j))
    drop = set()
    if toks:
        hit = np.isin(pd.util.hash_array(np.array(toks, dtype=object)), common)
        drop = {pos[k] for k in np.flatnonzero(hit)}
    return [" ".join(w for j, w in enumerate(ws) if (i, j) not in drop) for i, ws in enumerate(words)]


# ----------------------------------------------------------------------------- data loading
def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, engine="c")


def load_split(data, split):
    frames = [read_tsv(os.path.join(data, split, f"{split}_source{s}.tsv")).sort_values(
        "country", kind="stable", ignore_index=True) for s in (1, 2, 3)]
    sizes = [len(f) for f in frames]
    codes, labels = pd.factorize(pd.concat([f["country"] for f in frames], ignore_index=True))
    labels = [str(x).strip().lower() or "?" for x in labels]  # one shared string object per label
    rec = {
        "ids": np.concatenate([f["entity_id"].to_numpy(object) for f in frames]),
        "names": [x for f in frames for x in f["business_name"].tolist()],
        "addrs": [x for f in frames for x in f["business_address"].tolist()],
        "countries": [labels[c] for c in codes],
        "src": np.repeat(np.array([1, 2, 3], np.int8), sizes),
        "n1": sizes[0],
    }
    del frames, codes
    log(f"{split}: S1 {sizes[0]:,} S2 {sizes[1]:,} S3 {sizes[2]:,}")
    return rec


def load_gt(data, ids):
    gt = read_tsv(os.path.join(data, "train", "train_ground_truth.tsv"))
    a, b = [], []
    for s1, m in zip(gt["source1_entity_id"].tolist(), gt["matched_entity_ids"].tolist()):
        if m:
            for t in m.split(","):
                a.append(s1); b.append(t.strip())
    index = pd.Index(ids)
    q, t = index.get_indexer(a), index.get_indexer(b)
    ok = (q >= 0) & (t >= 0)
    log(f"ground truth: {len(a):,} pairs, {int((~ok).sum())} unresolved")
    return q[ok].astype(np.int64), t[ok].astype(np.int64)


def subset(rec, keep):
    idx = np.nonzero(keep)[0]
    return {"ids": rec["ids"][idx], "names": [rec["names"][i] for i in idx],
            "addrs": [rec["addrs"][i] for i in idx], "countries": [rec["countries"][i] for i in idx],
            "src": rec["src"][idx], "n1": int(keep[:rec["n1"]].sum())}


def unit_hash(ids, salt=""):
    h = pd.util.hash_array(np.array([salt + x for x in ids], dtype=object))
    return (h % np.uint64(1_000_003)).astype(np.float64) / 1_000_003.0


# ----------------------------------------------------------------------------- transliteration table
_TOKEN = re.compile(r"[\wऀ-ൿ]+")


def learn_translit(rec, gq, gt, limit):
    """Native-script target names transliterate the S1 name token for token; native address
    tokens are mapped by co-occurrence with the S1 address.  Train ground truth only."""
    from collections import Counter, defaultdict
    names, addrs = rec["names"], rec["addrs"]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(gq))
    pair_c, co, latin_df, native_df = Counter(), Counter(), Counter(), Counter()
    n_name = n_addr = 0
    for j in order:
        if n_name >= limit and n_addr >= limit:
            break
        s, t = gq[j], gt[j]
        tn, ta = names[t], addrs[t]
        if n_name < limit and NATIVE.search(tn):
            a, b = _TOKEN.findall(tn.lower()), _TOKEN.findall(names[s].lower())
            if len(a) == len(b):
                n_name += 1
                for x, y in zip(a, b):
                    if NATIVE.search(x) and not NATIVE.search(y):
                        pair_c[(x, y)] += 1
        if n_addr < limit and NATIVE.search(ta):
            n_addr += 1
            lat = set(_TOKEN.findall(addrs[s].lower()))
            latin_df.update(lat)
            for x in {w for w in _TOKEN.findall(ta.lower()) if NATIVE.search(w)}:
                native_df[x] += 1
                for y in lat:
                    co[(x, y)] += 1
    table = {"name": {}, "addr": {}}
    opts = defaultdict(list)
    for (x, y), c in pair_c.items():
        opts[x].append((c, y))
    for x, o in opts.items():
        o.sort(reverse=True)
        if o[0][0] >= 2 and o[0][0] >= 0.5 * sum(c for c, _ in o):
            table["name"][x] = o[0][1]
    opts = defaultdict(list)
    for (x, y), c in co.items():
        if c >= 3:
            opts[x].append((c / native_df[x], c / latin_df[y] ** 0.5, y))
    for x, o in opts.items():
        share, _, y = max(o)
        if share >= 0.6:
            table["addr"][x] = y
    log(f"translit: {len(table['name']):,} name tokens from {n_name:,} pairs, "
        f"{len(table['addr']):,} address tokens from {n_addr:,} pairs")
    return table


# ----------------------------------------------------------------------------- numba kernels
@njit(nogil=True)
def _heap_push(hs, hi, size, k, s, d):
    if size < k:
        j = size
        hs[j] = s
        hi[j] = d
        while j > 0:
            p = (j - 1) // 2
            if hs[p] <= hs[j]:
                break
            ts = hs[p]; hs[p] = hs[j]; hs[j] = ts
            ti = hi[p]; hi[p] = hi[j]; hi[j] = ti
            j = p
        return size + 1
    if s <= hs[0]:
        return size
    hs[0] = s
    hi[0] = d
    j = 0
    while True:
        left = 2 * j + 1
        if left >= size:
            break
        m = left
        if left + 1 < size and hs[left + 1] < hs[left]:
            m = left + 1
        if hs[j] <= hs[m]:
            break
        ts = hs[m]; hs[m] = hs[j]; hs[j] = ts
        ti = hi[m]; hi[m] = hi[j]; hi[j] = ti
        j = m
    return size


@njit(nogil=True)
def _heap_out(hs, hi, size, out_i, out_s):
    if size == 0:
        return
    o = np.argsort(-hs[:size])
    for r in range(size):
        out_i[r] = hi[o[r]]
        out_s[r] = hs[o[r]]


@njit(nogil=True)
def _exact_cos(r1, r2, rec_ptr, rec_tok, rec_w, tok_grp):
    """Exact name and address cosines of two records over all their tokens."""
    a, a1, b, b1 = rec_ptr[r1], rec_ptr[r1 + 1], rec_ptr[r2], rec_ptr[r2 + 1]
    dn = 0.0
    da = 0.0
    while a < a1 and b < b1:
        ta = rec_tok[a]
        tb = rec_tok[b]
        if ta < tb:
            a += 1
        elif tb < ta:
            b += 1
        else:
            if tok_grp[ta] == 0:
                dn += rec_w[a] * rec_w[b]
            else:
                da += rec_w[a] * rec_w[b]
            a += 1
            b += 1
    return dn, da


@njit(parallel=True)
def _retrieve(q_rows, rec_ptr, rec_tok, rec_w, tok_grp, inv_ptr, inv_doc, inv_w, n_index, d_off, cap, budget,
              q_has_a, d_has_a, k_c, k_n, k_a, wide, n_chunks):
    """Top-k retrieval. Each query spends a posting budget on its rarest tokens (per field group) to
    build wide candidate pools by partial cosine, then re-ranks the pools by exact cosine over all tokens."""
    n_q = q_rows.shape[0]
    kn = max(k_n, 1)
    ka = max(k_a, 1)
    wc = k_c * wide
    wn = kn * wide
    wa = ka * wide
    oc_i = np.full((n_q, k_c), -1, np.int32)
    oc_s = np.zeros((n_q, k_c), np.float32)
    on_i = np.full((n_q, kn), -1, np.int32)
    on_s = np.zeros((n_q, kn), np.float32)
    oa_i = np.full((n_q, ka), -1, np.int32)
    oa_s = np.zeros((n_q, ka), np.float32)
    step = (n_q + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        lo = c * step
        hi = min(n_q, lo + step)
        if lo >= hi:
            continue
        acc = np.zeros(2 * n_index, np.float32)  # interleaved name/address accumulators
        touched = np.empty(n_index, np.int32)
        mark = np.zeros(n_index, np.bool_)
        hs_c = np.empty(wc, np.float32)
        hi_c = np.empty(wc, np.int32)
        hs_n = np.empty(wn, np.float32)
        hi_n = np.empty(wn, np.int32)
        hs_a = np.empty(wa, np.float32)
        hi_a = np.empty(wa, np.int32)
        cand = np.empty(wc + wn + wa, np.int32)
        en = np.empty(wc + wn + wa, np.float32)
        ea = np.empty(wc + wn + wa, np.float32)
        ec = np.empty(wc + wn + wa, np.float32)
        for qq in range(lo, hi):
            r = q_rows[qq]
            m = 0
            p0 = rec_ptr[r]
            nt_q = rec_ptr[r + 1] - p0
            lens = np.empty(nt_q, np.int64)
            for j in range(nt_q):
                t = rec_tok[p0 + j]
                lens[j] = inv_ptr[t + 1] - inv_ptr[t]
            used_n = 0
            used_a = 0
            for j in np.argsort(lens):
                ln = lens[j]
                if ln == 0:
                    continue
                if ln > cap:
                    break
                p = p0 + j
                t = rec_tok[p]
                g = tok_grp[t]
                if g == 0:
                    if used_n > 0 and used_n + ln > budget:
                        continue
                    used_n += ln
                else:
                    if used_a > 0 and used_a + ln > budget:
                        continue
                    used_a += ln
                w = rec_w[p]
                for x in range(inv_ptr[t], inv_ptr[t + 1]):
                    d = inv_doc[x]
                    if acc[2 * d] == 0.0 and acc[2 * d + 1] == 0.0:
                        touched[m] = d
                        m += 1
                    acc[2 * d + g] += w * inv_w[x]
            qa = q_has_a[r]
            sc = 0
            sn = 0
            sa = 0
            for j in range(m):
                d = touched[j]
                vn = acc[2 * d]
                va = acc[2 * d + 1]
                if not qa:
                    comb = vn
                elif d_has_a[d]:
                    comb = 0.5 * (vn + va)
                else:
                    comb = 0.5 * vn + 0.2
                sc = _heap_push(hs_c, hi_c, sc, wc, comb, d)
                if k_n > 0 and vn > 0:
                    sn = _heap_push(hs_n, hi_n, sn, wn, vn, d)
                if k_a > 0 and va > 0:
                    sa = _heap_push(hs_a, hi_a, sa, wa, va, d)
                acc[2 * d] = 0.0
                acc[2 * d + 1] = 0.0
            u = 0
            for j in range(sc):
                d = hi_c[j]
                if not mark[d]:
                    mark[d] = True
                    cand[u] = d
                    u += 1
            for j in range(sn):
                d = hi_n[j]
                if not mark[d]:
                    mark[d] = True
                    cand[u] = d
                    u += 1
            for j in range(sa):
                d = hi_a[j]
                if not mark[d]:
                    mark[d] = True
                    cand[u] = d
                    u += 1
            if u == 0:
                continue
            for j in range(u):
                d = cand[j]
                mark[d] = False
                x, y = _exact_cos(r, d_off + d, rec_ptr, rec_tok, rec_w, tok_grp)
                en[j] = x
                ea[j] = y
                if not qa:
                    ec[j] = x
                elif d_has_a[d]:
                    ec[j] = 0.5 * (x + y)
                else:
                    ec[j] = 0.5 * x + 0.2
            o = np.argsort(-ec[:u])
            for t in range(min(k_c, u)):
                oc_i[qq, t] = cand[o[t]]
                oc_s[qq, t] = ec[o[t]]
            if k_n > 0:
                o = np.argsort(-en[:u])
                kk = 0
                for t in range(u):
                    if kk >= k_n or en[o[t]] <= 0:
                        break
                    on_i[qq, kk] = cand[o[t]]
                    on_s[qq, kk] = en[o[t]]
                    kk += 1
            if k_a > 0:
                o = np.argsort(-ea[:u])
                kk = 0
                for t in range(u):
                    if kk >= k_a or ea[o[t]] <= 0:
                        break
                    oa_i[qq, kk] = cand[o[t]]
                    oa_s[qq, kk] = ea[o[t]]
                    kk += 1
    return oc_i, oc_s, on_i, on_s, oa_i, oa_s


@njit(parallel=True)
def _sort_within(ptr, tok, w):
    for r in prange(ptr.shape[0] - 1):
        a = ptr[r]
        b = ptr[r + 1]
        for i in range(a + 1, b):
            t = tok[i]
            x = w[i]
            j = i - 1
            while j >= a and tok[j] > t:
                tok[j + 1] = tok[j]
                w[j + 1] = w[j]
                j -= 1
            tok[j + 1] = t
            w[j + 1] = x


N_TOK = 35


@njit(parallel=True)
def _pair_tok(pq, pt, rec_ptr, rec_tok, rec_w, tok_field, tok_idf, n_chunks):
    """Exact per-field cosines and token-set statistics for aligned (q, t) record pairs."""
    n = pq.shape[0]
    out = np.zeros((n, N_TOK), np.float32)
    step = (n + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        lo = c * step
        hi = min(n, lo + step)
        st = np.zeros((7, 9), np.float64)
        for i in range(lo, hi):
            st[:, :] = 0.0
            a = rec_ptr[pq[i]]
            a1 = rec_ptr[pq[i] + 1]
            b = rec_ptr[pt[i]]
            b1 = rec_ptr[pt[i] + 1]
            dn = 0.0
            da = 0.0
            while a < a1 or b < b1:
                if b >= b1 or (a < a1 and rec_tok[a] < rec_tok[b]):
                    t = rec_tok[a]
                    f = tok_field[t]
                    v = tok_idf[t]
                    st[f, 0] += 1
                    st[f, 3] += v
                    if v > st[f, 7]:
                        st[f, 7] = v
                    a += 1
                elif a >= a1 or rec_tok[b] < rec_tok[a]:
                    t = rec_tok[b]
                    f = tok_field[t]
                    v = tok_idf[t]
                    st[f, 1] += 1
                    st[f, 4] += v
                    if v > st[f, 8]:
                        st[f, 8] = v
                    b += 1
                else:
                    t = rec_tok[a]
                    f = tok_field[t]
                    v = tok_idf[t]
                    st[f, 0] += 1
                    st[f, 1] += 1
                    st[f, 2] += 1
                    st[f, 3] += v
                    st[f, 4] += v
                    st[f, 5] += v
                    if v > st[f, 6]:
                        st[f, 6] = v
                    if f < 4:
                        dn += rec_w[a] * rec_w[b]
                    else:
                        da += rec_w[a] * rec_w[b]
                    a += 1
                    b += 1
            out[i, 0] = dn
            out[i, 1] = da
            col = 2
            for f in (0, 4, 5):
                nq = st[f, 0]
                nt = st[f, 1]
                ni = st[f, 2]
                u = nq + nt - ni
                out[i, col] = nq
                out[i, col + 1] = nt
                out[i, col + 2] = ni
                out[i, col + 3] = ni / u if u > 0 else np.nan
                out[i, col + 4] = st[f, 5] / st[f, 3] if st[f, 3] > 0 else np.nan
                out[i, col + 5] = st[f, 5] / st[f, 4] if st[f, 4] > 0 else np.nan
                out[i, col + 6] = st[f, 6]
                out[i, col + 7] = st[f, 7]
                out[i, col + 8] = st[f, 8]
                col += 9
            for f in (1, 2, 3):
                ni = st[f, 2]
                u = st[f, 0] + st[f, 1] - ni
                out[i, col] = ni
                out[i, col + 1] = ni / u if u > 0 else np.nan
                col += 2
    return out


@njit(parallel=True)
def _group_rank(order, gptr, s, rank, gap, margin, cnt):
    """Within each group (order[gptr[g]:gptr[g+1]]): rank by s (0 = best), gap to the best and
    margin over the best *other* member (positive only for a unique best)."""
    for g in prange(gptr.shape[0] - 1):
        a = gptr[g]
        b = gptr[g + 1]
        n = b - a
        if n == 0:
            continue
        sv = np.empty(n, np.float32)
        for j in range(n):
            sv[j] = s[order[a + j]]
        o = np.argsort(-sv)
        best = sv[o[0]]
        second = sv[o[1]] if n > 1 else np.float32(0.0)
        for r in range(n):
            p = order[a + o[r]]
            rank[p] = r
            gap[p] = best - sv[o[r]]
            margin[p] = (sv[o[r]] - second) if r == 0 else (sv[o[r]] - best)
            cnt[p] = n


N_SUP = 21


@njit(parallel=True)
def _support(qptr, pq, pt, psrc, rec_ptr, rec_tok, tok_field, tok_idf):
    """Cross-source support of a target's extra tokens (tokens the S1 lacks) and of its missing tokens
    (S1 tokens the target lacks) among the other candidates of the same S1.  Copies of a sibling business
    share their differing name word and house number across S2 and S3, and all lack the S1's own word and
    number, which the true copies carry; random noise on a true copy is not shared."""
    n = pq.shape[0]
    out = np.zeros((n, N_SUP), np.float32)
    for g in prange(qptr.shape[0] - 1):
        a = qptr[g]
        b = qptr[g + 1]
        if b <= a:
            continue
        q = pq[a]
        qa = rec_ptr[q]
        qb = rec_ptr[q + 1]
        nq = qb - qa
        qc2 = np.zeros(nq, np.int32)
        qc3 = np.zeros(nq, np.int32)
        has = np.zeros((b - a, nq), np.bool_)
        for i in range(a, b):
            x = qa
            for y in range(rec_ptr[pt[i]], rec_ptr[pt[i] + 1]):
                t = rec_tok[y]
                while x < qb and rec_tok[x] < t:
                    x += 1
                if x < qb and rec_tok[x] == t:
                    has[i - a, x - qa] = True
                    if psrc[i] == 2:
                        qc2[x - qa] += 1
                    else:
                        qc3[x - qa] += 1
        for i in range(a, b):
            for k in range(nq):
                t = rec_tok[qa + k]
                f = tok_field[t]
                if (f != 0 and f != 4 and f != 5) or has[i - a, k]:
                    continue
                other = qc3[k] if psrc[i] == 2 else qc2[k]
                if other > 0:
                    base = 12 + (0 if f == 0 else (3 if f == 4 else 6))
                    out[i, base] += 1
                    if other > out[i, base + 1]:
                        out[i, base + 1] = other
                    if tok_idf[t] > out[i, base + 2]:
                        out[i, base + 2] = tok_idf[t]
        tot = 0
        for i in range(a, b):
            tot += rec_ptr[pt[i] + 1] - rec_ptr[pt[i]]
        tk = np.empty(tot, np.int32)
        ow = np.empty(tot, np.int32)
        m = 0
        for i in range(a, b):
            x = qa
            for y in range(rec_ptr[pt[i]], rec_ptr[pt[i] + 1]):
                t = rec_tok[y]
                f = tok_field[t]
                if f != 0 and f != 4 and f != 5:
                    continue
                while x < qb and rec_tok[x] < t:
                    x += 1
                if x < qb and rec_tok[x] == t:
                    continue
                tk[m] = t
                ow[m] = i
                m += 1
        if m == 0:
            continue
        o = np.argsort(tk[:m])
        u = 0
        while u < m:
            v = u
            c2 = 0
            c3 = 0
            while v < m and tk[o[v]] == tk[o[u]]:
                if psrc[ow[o[v]]] == 2:
                    c2 += 1
                else:
                    c3 += 1
                v += 1
            t = tk[o[u]]
            f = tok_field[t]
            base = 0 if f == 0 else (4 if f == 4 else 8)
            idf = tok_idf[t]
            for r in range(u, v):
                i = ow[o[r]]
                if psrc[i] == 2:
                    other = c3
                    same = c2 - 1
                else:
                    other = c2
                    same = c3 - 1
                if other > 0:
                    out[i, base] += 1
                    if other > out[i, base + 1]:
                        out[i, base + 1] = other
                    if idf > out[i, base + 2]:
                        out[i, base + 2] = idf
                if same > 0:
                    out[i, base + 3] += 1
            u = v
    return out


@njit(nogil=True)
def _jacc3(r1, r2, rec_ptr, rec_tok, tok_field):
    """Jaccard of name words, address words and numbers between two records; both-have-numbers flag."""
    a, a1, b, b1 = rec_ptr[r1], rec_ptr[r1 + 1], rec_ptr[r2], rec_ptr[r2 + 1]
    na = np.zeros(3)
    nb = np.zeros(3)
    ni = np.zeros(3)
    while a < a1 or b < b1:
        if b >= b1 or (a < a1 and rec_tok[a] < rec_tok[b]):
            f = tok_field[rec_tok[a]]
            if f == 0 or f == 4 or f == 5:
                na[0 if f == 0 else f - 3] += 1
            a += 1
        elif a >= a1 or rec_tok[b] < rec_tok[a]:
            f = tok_field[rec_tok[b]]
            if f == 0 or f == 4 or f == 5:
                nb[0 if f == 0 else f - 3] += 1
            b += 1
        else:
            f = tok_field[rec_tok[a]]
            if f == 0 or f == 4 or f == 5:
                k = 0 if f == 0 else f - 3
                na[k] += 1
                nb[k] += 1
                ni[k] += 1
            a += 1
            b += 1
    out = np.zeros(3)
    for k in range(3):
        u = na[k] + nb[k] - ni[k]
        out[k] = ni[k] / u if u > 0 else 0.0
    return out[0], out[1], out[2], na[2] > 0 and nb[2] > 0


N_COL = 20
COL_NAMES = ["p1", "p1_rank", "p1_gap", "p1_margin", "q_psum", "q_n50", "anc_p", "anc_jn", "anc_ja", "anc_jh",
             "anc_same_src", "conf_n", "conf_sib_a", "conf_sib_n", "conf_num_agree", "conf_num_n",
             "best_same_a", "best_other_a", "wsup", "isolated"]


@njit(parallel=True)
def _collective(qptr, pt, psrc, p1, rec_ptr, rec_tok, tok_field):
    """Agreement of each candidate with the S1's confident co-candidates (stage-1 probabilities).
    True copies agree with the anchor (best other candidate) on house number and address; copies of a
    sibling business form their own coherent cluster that disagrees with it."""
    n = pt.shape[0]
    out = np.zeros((n, N_COL), np.float32)
    for g in prange(qptr.shape[0] - 1):
        a = qptr[g]
        b = qptr[g + 1]
        m = b - a
        if m == 0:
            continue
        pv = p1[a:b].copy()
        order = np.argsort(-pv)
        rank = np.empty(m, np.int32)
        for r in range(m):
            rank[order[r]] = r
        psum = pv.sum()
        n50 = 0
        for r in range(m):
            if pv[r] >= 0.5:
                n50 += 1
        best = pv[order[0]]
        second = pv[order[1]] if m > 1 else np.float32(0.0)
        top = min(m, 16)
        for li in range(m):
            i = a + li
            out[i, 0] = pv[li]
            out[i, 1] = rank[li]
            out[i, 2] = best - pv[li]
            out[i, 3] = pv[li] - (second if rank[li] == 0 else best)
            out[i, 4] = psum
            out[i, 5] = n50
            anc = order[0] if order[0] != li else (order[1] if m > 1 else -1)
            if anc >= 0:
                jn, ja, jh, _ = _jacc3(pt[i], pt[a + anc], rec_ptr, rec_tok, tok_field)
                out[i, 6] = pv[anc]
                out[i, 7] = jn
                out[i, 8] = ja
                out[i, 9] = jh
                out[i, 10] = 1.0 if psrc[i] == psrc[a + anc] else 0.0
            else:
                for c in range(6, 11):
                    out[i, c] = -1.0
            cn = 0
            sib_a = 0
            sib_n = 0
            agree = 0
            nnum = 0
            bs = -1.0
            bo = -1.0
            wnum = 0.0
            wden = 0.0
            for r in range(top):
                lj = order[r]
                if lj == li:
                    continue
                pj = pv[lj]
                if pj < 0.05:
                    break
                jn, ja, jh, both = _jacc3(pt[i], pt[a + lj], rec_ptr, rec_tok, tok_field)
                wnum += pj * 0.5 * (ja + jn)
                wden += pj
                if pj >= 0.5:
                    cn += 1
                    if ja >= 0.8:
                        sib_a += 1
                    if jn >= 0.8:
                        sib_n += 1
                    if both:
                        nnum += 1
                        if jh > 0:
                            agree += 1
                    if psrc[a + lj] == psrc[i]:
                        bs = max(bs, ja)
                    else:
                        bo = max(bo, ja)
            out[i, 11] = cn
            out[i, 12] = sib_a
            out[i, 13] = sib_n
            out[i, 14] = agree
            out[i, 15] = nnum
            out[i, 16] = bs
            out[i, 17] = bo
            out[i, 18] = wnum / wden if wden > 0 else -1.0
            out[i, 19] = 1.0 if (nnum > 0 and agree == 0) else 0.0
    return out


def collective(G, sel, p1):
    """Collective features for the pairs `sel` (whole S1 groups) given their stage-1 probabilities."""
    pq, pt = G["pq"][sel], G["pt"][sel]
    starts = np.flatnonzero(np.r_[True, pq[1:] != pq[:-1]])
    qptr = np.r_[starts, len(pq)].astype(np.int64)
    return _collective(qptr, pt, G["src"][pt], p1.astype(np.float32), G["rec_ptr"], G["tok"], G["tok_field"])


TCOMP_NAMES = ["p1_trank", "p1_tmargin", "p1_tgap", "p1_tn50"]
CPRIOR_NAMES = ["q_c2", "q_c3", "c_rank", "c_gap", "c_margin"]


def target_p_stats(G, p_all, cprior=False, rows=None):
    """Competition of stage-1 probabilities among all S1 claiming the same target (full graph).
    cprior adds a copy-count prior: an entity has ~3.5 copies, so among S1 competing for an ambiguous
    target, the one with fewer confident copies (excluding this target) is the likelier owner."""
    pt = G["pt"]
    P = len(pt)
    t_order = np.argsort(pt, kind="stable").astype(np.int64)
    t_ptr = np.zeros(G["n"] + 1, np.int64)
    t_ptr[1:] = np.cumsum(np.bincount(pt, minlength=G["n"]))
    rk, gp, mg, ct = (np.zeros(P, np.float32) for _ in range(4))
    _group_rank(t_order, t_ptr, p_all.astype(np.float32), rk, gp, mg, ct)
    n50 = np.bincount(pt, weights=(p_all >= 0.5), minlength=G["n"])[pt]
    cols = [rk, mg, gp, n50]
    if cprior:
        pq, n1 = G["pq"], G["n1"]
        conf = p_all >= 0.5
        s2 = G["src"][pt] == 2
        q_c2 = (np.bincount(pq, weights=conf & s2, minlength=n1)[pq] - (conf & s2)).astype(np.float32)
        q_c3 = (np.bincount(pq, weights=conf & ~s2, minlength=n1)[pq] - (conf & ~s2)).astype(np.float32)
        rk2, gp2, mg2, ct2 = (np.zeros(P, np.float32) for _ in range(4))
        _group_rank(t_order, t_ptr, -(q_c2 + q_c3), rk2, gp2, mg2, ct2)
        cols += [q_c2, q_c3, rk2, gp2, mg2]
    if rows is not None:  # materialize only the rows that are used (saves ~3 GB on the full graph)
        cols = [c[rows] for c in cols]
    return np.column_stack(cols).astype(np.float32)


# ----------------------------------------------------------------------------- graph construction
def build_graph(rec, tables, cfg, true_key=None):
    """Normalize, tokenize, retrieve candidates and compute graph-level (competition) features."""
    n, n1 = len(rec["ids"]), rec["n1"]
    names, addrs, countries = rec["names"], rec["addrs"], rec["countries"]
    step = 100_000
    chunks = [(names[i:i + step], addrs[i:i + step], countries[i:i + step]) for i in range(0, n, step)]
    with Pool(NT, initializer=_init_worker, initargs=((tables["name"], tables["addr"], ABBR if cfg.get("norm", 1) >= 2 else ABBR_V1),)) as pool:
        res = pool.map(_norm_chunk, chunks, chunksize=1)
    del chunks, names, addrs
    rec.pop("names", None)  # raw text is no longer needed; free it before the heavy stages
    rec.pop("addrs", None)
    H = np.concatenate([r[0] for r in res])
    F = np.concatenate([r[1] for r in res])
    cnt = np.concatenate([r[2] for r in res])
    R = {k: np.array([x for r in res for x in r[j]], dtype=object)
         for j, k in ((3, "nm"), (4, "core"), (5, "cat"), (6, "ad"), (7, "hn"), (10, "ini"))}
    nums = np.concatenate([r[8] for r in res])
    flags = np.concatenate([r[9] for r in res])
    del res
    gc.collect()
    log(f"normalized {n:,} records, {len(H):,} tokens")

    uniq, tok = np.unique(H, return_inverse=True)
    tok = tok.astype(np.int32).ravel()
    del H
    V = int(tok.max()) + 1
    tok_field = np.zeros(V, np.int8)
    tok_field[tok] = F
    rec_of = np.repeat(np.arange(n, dtype=np.int32), cnt)
    ccode = pd.factorize(pd.Series(countries))[0].astype(np.int32)
    tok_ctry = np.zeros(V, np.int32)
    tok_ctry[tok] = ccode[rec_of]
    df = np.bincount(tok, minlength=V)
    Nc = np.bincount(ccode)
    tok_idf = (np.log((1.0 + Nc[tok_ctry]) / (1.0 + df)) + 1.0).astype(np.float32)
    grp = (F >= 4).astype(np.int64)
    w = tok_idf[tok].astype(np.float64) * FIELD_W[F]
    key = rec_of.astype(np.int64) * 2 + grp
    norm2 = np.bincount(key, weights=w * w, minlength=2 * n)
    w = (w / np.sqrt(norm2[key])).astype(np.float32)
    has_a = norm2[1::2] > 0
    if cfg.get("core"):
        # address words present in more than core_thr of the country's records (regions, departments, big
        # cities, street types) become field 6: they keep their small IDF weight in the address cosine but
        # leave the overlap statistics, sibling support and string features (France region<->department swaps)
        common = (tok_field == 4) & (df > cfg.get("core_thr", 0.02) * Nc[tok_ctry])
        tok_field[common] = 6
        chash = np.sort(uniq[common])
        cchunks = [(countries[i:i + step], R["ad"][i:i + step].tolist(), chash) for i in range(0, n, step)]
        with Pool(NT) as pool:
            R["adc"] = np.array([x for part in pool.map(_core_chunk, cchunks, chunksize=1) for x in part],
                                dtype=object)
        del cchunks
        log(f"core address: {int(common.sum()):,} common address words")
    del key, grp, rec_of, F, df, tok_ctry, uniq
    rec_ptr = np.zeros(n + 1, np.int64)
    rec_ptr[1:] = np.cumsum(cnt)
    _sort_within(rec_ptr, tok, w)
    tok_grp = (tok_field >= 4).astype(np.int8)
    log(f"vocabulary {V:,}; records with address {has_a.mean():.3f}")

    def inverted(lo, hi):
        a, b = rec_ptr[lo], rec_ptr[hi]
        t = tok[a:b]
        doc = np.repeat(np.arange(hi - lo, dtype=np.int32), np.diff(rec_ptr[lo:hi + 1]))
        order = np.argsort(t, kind="stable")
        ptr = np.zeros(V + 1, np.int64)
        ptr[1:] = np.cumsum(np.bincount(t, minlength=V))
        return ptr, doc[order], w[a:b][order]

    nch = NT * 8
    ip, idoc, iw = inverted(n1, n)
    fc_i, fc_s, fn_i, fn_s, fa_i, fa_s = _retrieve(
        np.arange(n1, dtype=np.int64), rec_ptr, tok, w, tok_grp, ip, idoc, iw, n - n1, n1, cfg["cap_t"],
        cfg["budget_t"], has_a, has_a[n1:].copy(), cfg["k_fwd"], cfg["k_name"], cfg["k_addr"], cfg["wide"], nch)
    del ip, idoc, iw
    log("forward retrieval done")
    ip, idoc, iw = inverted(0, n1)
    rv_i, rv_s, _, _, _, _ = _retrieve(
        np.arange(n1, n, dtype=np.int64), rec_ptr, tok, w, tok_grp, ip, idoc, iw, n1, 0, cfg["cap_s1"],
        cfg["budget_s1"], has_a, has_a[:n1].copy(), cfg["k_rev"], 0, 0, cfg["wide"], nch)
    del ip, idoc, iw
    log("reverse retrieval done")

    def view_keys(ii, ss, rows_are_s1, min_s=0.0):
        k = ii.shape[1]
        rows = np.repeat(np.arange(ii.shape[0], dtype=np.int64), k)
        rank = np.tile(np.arange(k, dtype=np.int16), ii.shape[0])
        cols = ii.ravel().astype(np.int64)
        ok = (cols >= 0) & (ss.ravel() > min_s)
        q, t = (rows, cols) if rows_are_s1 else (cols, rows)
        return (q[ok] << 32) | t[ok], rank[ok]

    views = [view_keys(rv_i, rv_s, False, cfg["rev_min"]), view_keys(fc_i, fc_s, True),
             view_keys(fn_i, fn_s, True), view_keys(fa_i, fa_s, True)]
    del rv_i, rv_s, fc_i, fc_s, fn_i, fn_s, fa_i, fa_s
    if true_key is not None:  # retrieval diagnostics: recall of each view and of the union
        names = ("rev", "fwd_combo", "fwd_name", "fwd_addr")
        for nm_, (kk, rr) in zip(names, views):
            hit = np.isin(((kk >> 32) << 32) | ((kk & 0xFFFFFFFF) + n1), true_key)
            at = [round(float(hit[rr < r].sum() / len(true_key)), 4) for r in (1, 2, 4, 8, 12, 20) if r <= rr.max() + 1]
            log(f"DIAG view {nm_}: pairs {len(kk):,} recall {hit.sum() / len(true_key):.4f} recall@1,2,4,8,.. {at}")
    uk = np.unique(np.concatenate([v[0] for v in views]))
    ranks = np.full((len(uk), 4), 99, np.int8)
    for j, (kk, rr) in enumerate(views):
        ranks[np.searchsorted(uk, kk), j] = rr
    del views
    pq = (uk >> 32).astype(np.int32)
    pt = ((uk & 0xFFFFFFFF) + n1).astype(np.int32)
    del uk
    P = len(pq)
    log(f"candidates: {P:,} pairs, {P / max(n1, 1):.1f} per S1")

    cos = np.zeros((P, 3), np.float32)
    ch = cfg["chunk"]
    for a in range(0, P, ch):
        o = _pair_tok(pq[a:a + ch], pt[a:a + ch], rec_ptr, tok, w, tok_field, tok_idf, nch)
        cos[a:a + ch, 1] = o[:, 0]
        cos[a:a + ch, 2] = o[:, 1]
    qa, ta = has_a[pq], has_a[pt]
    cos[:, 0] = np.where(~qa, cos[:, 1], np.where(ta, 0.5 * (cos[:, 1] + cos[:, 2]), 0.5 * cos[:, 1] + 0.2))

    # competition features over the full candidate graph (label-free)
    comp = np.zeros((P, 16), np.float16)  # ranks/counts are small integers, gaps/margins in [-1, 1]
    t_order = np.argsort(pt, kind="stable").astype(np.int64)
    t_ptr = np.zeros(n + 1, np.int64)
    t_ptr[1:] = np.cumsum(np.bincount(pt, minlength=n))
    q_order = np.arange(P, dtype=np.int64)
    q_ptr = np.zeros(n1 + 1, np.int64)
    q_ptr[1:] = np.cumsum(np.bincount(pq, minlength=n1))
    rk, gp, mg, ct = (np.zeros(P, np.float32) for _ in range(4))
    col = 0
    for si, full in ((0, True), (1, False), (2, False)):
        s = np.ascontiguousarray(cos[:, si])
        _group_rank(t_order, t_ptr, s, rk, gp, mg, ct)
        comp[:, col], comp[:, col + 1] = rk, mg
        col += 2
        if full:
            comp[:, col] = gp
            comp[:, 15] = ct
            col += 1
        _group_rank(q_order, q_ptr, s, rk, gp, mg, ct)
        comp[:, col], comp[:, col + 1] = rk, gp
        col += 2
        if full:
            comp[:, col] = mg
            comp[:, 14] = ct
            col += 1
    del t_order, q_order, rk, gp, mg, ct
    gc.collect()
    log("graph features done")

    # name ambiguity counts (label-free)
    key = pd.Series([c + "|" + x for c, x in zip(countries, R["core"])])
    is1 = np.zeros(n, bool)
    is1[:n1] = True
    s1_counts = key[is1].value_counts()
    t_counts = key[~is1].value_counts()
    nf_s1 = key.map(s1_counts).fillna(0).to_numpy(np.float32)
    nf_t = key.map(t_counts).fillna(0).to_numpy(np.float32)
    if cfg.get("relfreq"):  # per 100k records of the country: comparable across country sizes (unseen France)
        cs = pd.Series(countries)
        nf_s1 = nf_s1 / cs.map(cs[is1].value_counts()).to_numpy(np.float32) * 1e5
        nf_t = nf_t / cs.map(cs[~is1].value_counts()).to_numpy(np.float32) * 1e5
    del key

    del cos
    return {"n": n, "n1": n1, "pq": pq, "pt": pt, "ranks": ranks, "comp": comp,
            "q_ptr": q_ptr, "rec_ptr": rec_ptr, "tok": tok, "w": w, "tok_field": tok_field,
            "tok_idf": tok_idf, "has_a": has_a, "R": R, "nums": nums, "flags": flags,
            "nf_s1": nf_s1, "nf_t": nf_t, "src": rec["src"], "nch": nch}


# ----------------------------------------------------------------------------- pair features
TOK_NAMES = ["cos_n", "cos_a"] + [f"{p}_{s}" for p in ("n", "a", "h") for s in
                                  ("nq", "nt", "ni", "jac", "covq", "covt", "mi", "mq", "mt")] + \
            ["k_ni", "k_jac", "c_ni", "c_jac", "p_ni", "p_jac"]
COMP_NAMES = ["c_trank", "c_tmargin", "c_tgap", "c_qrank", "c_qgap", "c_qmargin",
              "n_trank", "n_tmargin", "n_qrank", "n_qgap", "a_trank", "a_tmargin", "a_qrank", "a_qgap",
              "q_cnt", "t_cnt"]
SUP_NAMES = [f"x{f}_{s}" for f in ("n", "a", "h") for s in ("ocnt", "omax", "oidf", "scnt")] + [
    f"m{f}_{s}" for f in ("n", "a", "h") for s in ("ocnt", "omax", "oidf")]
RF_SPEC = [("nm", "ratio"), ("nm", "token_sort_ratio"), ("nm", "token_set_ratio"), ("nm", "partial_ratio"),
           ("nm", "jw"), ("core", "ratio"), ("core", "token_set_ratio"), ("cat", "ratio"),
           ("cat", "partial_ratio"), ("ad", "ratio"), ("ad", "token_set_ratio"), ("ad", "token_sort_ratio"),
           ("hn", "lev")]
RF_NAMES = [f"rf_{f}_{s}" for f, s in RF_SPEC]
NUM_NAMES = ["hn_both", "hn_eq", "hn_absdiff", "hn_band", "hn_mindiff", "acr"]
STRUCT_NAMES = ["t_src", "t_native", "t_alias", "t_dom", "t_aempty", "t_native_a", "q_alias", "q_dom",
                "q_ncore", "t_ncore", "q_lnm", "t_lnm", "q_lad", "t_lad", "q_nf_s1", "t_nf_s1", "t_nf_t"]
FEATS = ["r_rev", "r_fwd", "r_fname", "r_faddr", "cos_c"] + TOK_NAMES + COMP_NAMES + SUP_NAMES + \
        RF_NAMES + NUM_NAMES + STRUCT_NAMES


def _rf_scorer(name):
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    return {"ratio": fuzz.ratio, "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio, "partial_ratio": fuzz.partial_ratio,
            "jw": JaroWinkler.normalized_similarity, "lev": Levenshtein.distance}[name]


def featurize(G, sel, live=None):
    """Feature matrix for the pairs `sel` (index array, sorted, made of whole S1 groups)."""
    from rapidfuzz import process
    pq, pt = G["pq"][sel], G["pt"][sel]
    X = np.empty((len(sel), len(FEATS)), np.float32)
    c = 0
    X[:, c:c + 4] = G["ranks"][sel]; c += 4
    T = _pair_tok(pq, pt, G["rec_ptr"], G["tok"], G["w"], G["tok_field"], G["tok_idf"], G["nch"])
    qa, ta = G["has_a"][pq], G["has_a"][pt]
    X[:, c] = np.where(~qa, T[:, 0], np.where(ta, 0.5 * (T[:, 0] + T[:, 1]), 0.5 * T[:, 0] + 0.2)); c += 1
    X[:, c:c + N_TOK] = T; c += N_TOK
    del T
    X[:, c:c + 16] = G["comp"][sel]; c += 16
    starts = np.flatnonzero(np.r_[True, pq[1:] != pq[:-1]])
    qptr = np.r_[starts, len(pq)].astype(np.int64)
    X[:, c:c + N_SUP] = _support(qptr, pq, pt, G["src"][pt], G["rec_ptr"], G["tok"], G["tok_field"],
                                 G["tok_idf"]); c += N_SUP
    R = G["R"]
    lq, lt = (pq, pt) if live is None else (pq[live], pt[live])
    for field, sc in RF_SPEC:
        if field == "ad" and "adc" in R:
            field = "adc"
        a, b = R[field][lq].tolist(), R[field][lt].tolist()
        v = process.cpdist(a, b, scorer=_rf_scorer(sc), workers=NT, dtype=np.float32)
        if field == "hn":
            v = np.where((R["hn"][lq] == "") | (R["hn"][lt] == ""), np.nan, v)
        if live is None:
            X[:, c] = v
        else:
            X[:, c] = np.nan
            X[live, c] = v
        c += 1
    hq, ht = G["nums"][pq, 0], G["nums"][pt, 0]
    both = (hq >= 0) & (ht >= 0)
    d = np.abs(hq - ht).astype(np.float64)
    X[:, c] = both
    X[:, c + 1] = both & (hq == ht)
    X[:, c + 2] = np.where(both, np.log1p(d), np.nan)
    X[:, c + 3] = both & (d >= 3) & (d <= 21)
    A = G["nums"][pq].astype(np.float64)
    B = G["nums"][pt].astype(np.float64)
    A[A < 0] = np.nan
    B[B < 0] = np.nan
    D = np.abs(A[:, :, None] - B[:, None, :]).reshape(len(pq), 16)
    D = np.where(np.isnan(D), np.inf, D).min(1)
    X[:, c + 4] = np.where(np.isfinite(D), np.log1p(D), np.nan)
    del A, B, D
    qi_, tc_ = R["ini"][pq], R["cat"][pt]
    ti_, qc_ = R["ini"][pt], R["cat"][pq]
    lq = np.fromiter(map(len, qi_), np.int32, len(qi_))
    lt = np.fromiter(map(len, ti_), np.int32, len(ti_))
    X[:, c + 5] = ((qi_ == tc_) & (lq >= 2)) | ((ti_ == qc_) & (lt >= 2))
    c += 6
    fq, ft = G["flags"][pq], G["flags"][pt]
    X[:, c] = G["src"][pt]
    X[:, c + 1:c + 6] = ft[:, [0, 1, 2, 3, 4]]
    X[:, c + 6:c + 8] = fq[:, [1, 2]]
    X[:, c + 8] = fq[:, 5]
    X[:, c + 9] = ft[:, 5]
    X[:, c + 10] = fq[:, 6]
    X[:, c + 11] = ft[:, 6]
    X[:, c + 12] = fq[:, 7]
    X[:, c + 13] = ft[:, 7]
    X[:, c + 14] = G["nf_s1"][pq]
    X[:, c + 15] = G["nf_s1"][pt]
    X[:, c + 16] = G["nf_t"][pt]
    c += 17
    assert c == len(FEATS)
    return X


def group_chunks(pq, size):
    """Contiguous pair ranges of about `size` rows that never split an S1 group."""
    P = len(pq)
    out, a = [], 0
    while a < P:
        b = min(P, a + size)
        while b < P and pq[b] == pq[b - 1]:
            b += 1
        out.append((a, b))
        a = b
    return out


# ----------------------------------------------------------------------------- decision + metric
def exclusive(pt, p):
    """True for the single best-scoring S1 of every target."""
    order = np.lexsort((-p, pt))
    first = np.ones(len(order), bool)
    first[1:] = pt[order][1:] != pt[order][:-1]
    keep = np.zeros(len(pt), bool)
    keep[order[first]] = True
    return keep


def macro_f05(qi, y, pred, nt):
    """qi: dense S1 index of each pair; nt: true match count of every evaluated S1."""
    m = len(nt)
    npred = np.bincount(qi, weights=pred, minlength=m)
    tp = np.bincount(qi, weights=pred & y, minlength=m)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = tp / npred
        Rr = tp / nt
        f = np.where(tp > 0, 1.25 * P * Rr / (0.25 * P + Rr), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return float(f.mean())


def tune_tau(qi, y, p, keep, nt, grid=None):
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2) if grid is None else grid
    scores = {float(t): macro_f05(qi, y, keep & (p >= t), nt) for t in grid}
    best = max(scores, key=scores.get)
    return best, scores


# ----------------------------------------------------------------------------- models
def _gpu():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).returncode == 0
    except Exception:
        return False


GPU = _gpu()


class Model:
    """LightGBM or XGBoost booster behind one predict(); XGBoost trains and predicts on the GPU if present."""

    def __init__(self, kind, booster, best=None):
        self.kind, self.b, self.best = kind, booster, best
        self.fil = None  # RAPIDS FIL (GPU) for LightGBM models, verified against LightGBM on first use

    def _fil_predict(self, X):
        import cupy as cp
        out = self.fil.predict_proba(cp.asarray(np.ascontiguousarray(X, dtype=np.float32)))
        out = cp.asnumpy(out) if hasattr(out, "__cuda_array_interface__") else np.asarray(out)
        return out[:, 1] if out.ndim == 2 else out

    def predict(self, X):
        if self.kind == "lgb":
            if self.fil is not None:
                try:
                    if not getattr(self, "checked", False):
                        k = min(len(X), 4000)
                        diff = np.abs(self._fil_predict(X[:k]) - self.b.predict(X[:k], num_threads=NT)).max()
                        self.checked = True
                        log(f"FIL vs LightGBM max diff {diff:.2e}")
                        if diff > 1e-3:
                            raise ValueError("FIL mismatch")
                    return self._fil_predict(X)
                except Exception as e:
                    log(f"FIL disabled ({e}); using LightGBM on CPU")
                    self.fil = None
            return self.b.predict(X, num_iteration=self.best, num_threads=NT)
        X = np.ascontiguousarray(X, dtype=np.float32)
        if GPU:
            try:
                import cupy as cp
                return cp.asnumpy(self.b.inplace_predict(cp.asarray(X)))
            except ImportError:
                pass
        return self.b.inplace_predict(X)

    def gain(self, names):
        if self.kind == "lgb":
            return dict(zip(names, self.b.feature_importance("gain").round(0).tolist()))
        sc = self.b.get_score(importance_type="total_gain")
        return {n: round(float(sc.get(n, 0.0)), 0) for n in names}

    @staticmethod
    def load(md, stem):
        if os.path.exists(os.path.join(md, stem + ".json")):
            import xgboost as xgb
            b = xgb.Booster(model_file=os.path.join(md, stem + ".json"))
            b.set_param({"device": "cuda" if GPU else "cpu", "nthread": NT})
            return Model("xgb", b)
        import lightgbm as lgb
        path = os.path.join(md, stem + ".txt")
        m = Model("lgb", lgb.Booster(model_file=path))
        if GPU:
            try:
                from cuml.fil import ForestInference
                try:
                    m.fil = ForestInference.load(path, model_type="lightgbm", is_classifier=True)
                except TypeError:
                    m.fil = ForestInference.load(path, model_type="lightgbm", output_class=True)
                log(f"{stem}: GPU FIL inference")
            except Exception as e:
                log(f"{stem}: FIL unavailable ({e})")
        return m


def _Batches(X, idx, y, w, names, batch=1_000_000, gpu=True):
    """xgboost.DataIter over rows idx of a float16 matrix, converted to float32 one batch at a time."""
    import xgboost as xgb
    cp = None
    if GPU and gpu:
        try:
            import cupy as cp
        except ImportError:
            cp = None

    class It(xgb.DataIter):
        def __init__(self):
            self.pos = 0
            super().__init__()

        def next(self, input_data):
            if self.pos >= len(idx):
                return False
            sl = idx[self.pos:self.pos + batch]
            kw = {"data": X[sl].astype(np.float32), "label": y[sl], "feature_names": names}
            if w is not None:
                kw["weight"] = w[sl]
            if cp is not None:  # GPU: the quantized matrix is built on the device, not in host RAM
                kw = {k: (cp.asarray(v) if k != "feature_names" else v) for k, v in kw.items()}
            input_data(**kw)
            self.pos += batch
            return True

        def reset(self):
            self.pos = 0

    return It()


# ----------------------------------------------------------------------------- train stage
def run_train(data, work, cfg):
    import lightgbm as lgb
    os.makedirs(os.path.join(work, "model"), exist_ok=True)
    full = load_split(data, "train")
    gq, gtt = load_gt(data, full["ids"])
    if cfg.get("subsample", 1.0) < 1.0:  # smoke-test mode: a share of S1 with all their targets
        keep = unit_hash(full["ids"].tolist(), "sub") < cfg["subsample"]
        keep[gtt[keep[gq]]] = True
        remap = np.full(len(keep), -1, np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        ok = keep[gq]
        gq, gtt = remap[gq[ok]], remap[gtt[ok]]
        full = subset(full, keep)
        log(f"subsample: {full['n1']:,} S1, {len(full['ids']):,} records")
    tables = learn_translit(full, gq, gtt, cfg["translit_pairs"])
    with open(os.path.join(work, "model", "translit.json"), "w", encoding="utf-8") as fh:
        json.dump(tables, fh, ensure_ascii=False)

    n1 = full["n1"]
    u = unit_hash(full["ids"][:n1].tolist())
    keep = np.ones(len(full["ids"]), bool)
    keep[:n1] = u >= cfg["drop"]
    remap = np.full(len(full["ids"]), -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    ok = keep[gq]
    gq, gtt = remap[gq[ok]], remap[gtt[ok]]
    rec = subset(full, keep)
    u = u[keep[:n1]]
    del full, remap
    gc.collect()
    tgt_per_s1 = (len(rec["ids"]) - rec["n1"]) / rec["n1"]
    log(f"P-dense graph: {rec['n1']:,} S1 kept, targets per S1 {tgt_per_s1:.2f}")

    true_key = np.unique((gq << 32) | gtt)
    G = build_graph(rec, tables, cfg, true_key if cfg.get("diag") else None)
    rec.pop("ids", None)
    n1 = G["n1"]
    nt_all = np.bincount(gq, minlength=n1).astype(np.float64)
    pkey = (G["pq"].astype(np.int64) << 32) | G["pt"].astype(np.int64)
    y_all = np.isin(pkey, true_key)
    del pkey
    log(f"pair recall (all kept S1): {y_all.sum() / len(true_key):.4f}")
    if cfg.get("diag"):
        nt_q = np.bincount(gq, minlength=n1)
        hit_q = np.bincount(G["pq"], weights=y_all, minlength=n1)
        full_q = (hit_q == nt_q)
        log(f"DIAG entity full recall {full_q.mean():.4f}; pairs per S1 {len(G['pq']) / n1:.1f}")
        return

    # sample layout by position a in [0, 1): fold 0 = [0, .4) (last tenth = early stopping),
    # fold 1 = [.4, .8) (same), validation = [.8, 1)
    a = (u - cfg["drop"]) / cfg["sample"]
    fold = np.full(n1, -1, np.int8)
    in_s = (a >= 0) & (a < 1)
    half = (1 - cfg["val_share"]) / 2
    fold[in_s & (a < half)] = 0
    fold[in_s & (a >= half) & (a < 2 * half)] = 1
    fold[in_s & (a >= 2 * half)] = 2
    es_s1 = in_s & (((a >= 0.9 * half) & (a < half)) | ((a >= 1.9 * half) & (a < 2 * half)))
    pf = fold[G["pq"]]
    sel = np.flatnonzero(pf >= 0)
    log(f"sample: {int(in_s.sum()):,} S1 ({[int((fold == k).sum()) for k in range(3)]}), {len(sel):,} pairs")
    y = y_all[sel]
    f_s = pf[sel]
    es_p = es_s1[G["pq"][sel]]
    pq_s, pt_s = G["pq"][sel], G["pt"][sel]
    rank_min = np.minimum(G["ranks"][sel, :2].min(1), G["comp"][sel, 0].astype(np.float32))
    rng = np.random.default_rng(cfg["seed"])
    easy = (~y) & (rank_min > 3)
    take = (f_s < 2) & ~es_p & (~easy | (rng.random(len(sel)) < cfg["neg_keep"]))
    need = take | ((f_s < 2) & es_p)
    wts = np.where(easy, 1.0 / cfg["neg_keep"], 1.0).astype(np.float32)
    chunks = group_chunks(pq_s, cfg["chunk"])

    params = dict(objective="binary", learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
                  min_data_in_leaf=cfg["min_leaf"], feature_fraction=cfg.get("ff", 0.7),
                  bagging_fraction=cfg.get("bf", 0.8), bagging_freq=1, lambda_l2=cfg.get("l2", 1.0),
                  max_bin=255, num_threads=NT, verbose=-1, seed=cfg.get("model_seed", cfg["seed"]))
    s1_of = {k: np.flatnonzero(fold == k) for k in range(3)}
    ctry_all = np.array(rec["countries"][:n1], dtype=object)

    def fit_stage(tag, make_x, names, fallback=None):
        """Cross-fit two LightGBM models (fold 0 / fold 1). Pass 1 keeps only training and early-stopping
        rows in memory; pass 2 recomputes features for out-of-fold and validation predictions."""
        rows = np.flatnonzero(need)
        X = np.empty((len(rows), len(names)), np.float16)
        pos = 0
        for a0, b0 in chunks:
            m = need[a0:b0]
            k = int(m.sum())
            X[pos:pos + k] = make_x(a0, b0)[m]
            pos += k
        log(f"{tag}: training matrix {X.shape} (float16)")

        class Rows(lgb.Sequence):
            batch_size = 65536

            def __init__(self, arr):
                self.arr = arr

            def __getitem__(self, idx):
                return self.arr[idx].astype(np.float64)

            def __len__(self):
                return len(self.arr)

        models = []
        if cfg.get("model", "lgb") == "xgb":
            import xgboost as xgb
            xp = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                  "device": "cuda" if GPU else "cpu", "grow_policy": "lossguide", "max_depth": 0,
                  "max_leaves": cfg["leaves"], "eta": cfg["lr"], "subsample": cfg.get("bf", 0.8),
                  "colsample_bytree": cfg.get("ff", 0.7), "min_child_weight": cfg.get("mcw", 2.0),
                  "lambda": cfg.get("l2", 1.0), "max_bin": 256, "seed": cfg.get("model_seed", cfg["seed"]),
                  "nthread": NT}
            for f in (0, 1):
                tr = np.flatnonzero(take[rows] & (f_s[rows] == f))
                es = np.flatnonzero(es_p[rows] & (f_s[rows] == f))
                # batches from the float16 matrix: no full float32 copy of the training rows
                try:
                    dtr = xgb.QuantileDMatrix(_Batches(X, tr, y[rows], wts[rows], names), max_bin=256)
                    dva = xgb.QuantileDMatrix(_Batches(X, es, y[rows], None, names), ref=dtr)
                except Exception as e:  # device batches unavailable: host batches
                    log(f"GPU DMatrix failed ({e}); using host batches")
                    dtr = xgb.QuantileDMatrix(_Batches(X, tr, y[rows], wts[rows], names, gpu=False), max_bin=256)
                    dva = xgb.QuantileDMatrix(_Batches(X, es, y[rows], None, names, gpu=False), ref=dtr)
                b = xgb.train(xp, dtr, cfg["rounds"], evals=[(dva, "es")], early_stopping_rounds=cfg["es"],
                              verbose_eval=200)
                best = b.best_iteration
                b = b[: best + 1]
                b.save_model(os.path.join(work, "model", f"{tag}_{f}.json"))
                log(f"{tag} xgb model {f} ({xp['device']}): {best + 1} rounds, train rows {len(tr):,}")
                models.append(Model("xgb", b))
                del dtr, dva
                gc.collect()
            del X
            gc.collect()
        else:
            # bin once, drop the float matrix, then train each fold on subset views of the binned data
            full = lgb.Dataset(Rows(X), y[rows], weight=np.where(es_p[rows], 1.0, wts[rows]), feature_name=names,
                               params=params, free_raw_data=True)
            full.construct()
            del X
            gc.collect()
            log(f"{tag}: binned dataset built")
            for f in (0, 1):
                tr = np.flatnonzero(take[rows] & (f_s[rows] == f))
                es = np.flatnonzero(es_p[rows] & (f_s[rows] == f))
                dtr = full.subset(tr)
                dva = full.subset(es)
                mdl = lgb.train(params, dtr, cfg["rounds"], valid_sets=[dva],
                                callbacks=[lgb.early_stopping(cfg["es"], verbose=False), lgb.log_evaluation(200)])
                log(f"{tag} model {f}: {mdl.best_iteration} rounds, train rows {len(tr):,}, es rows {len(es):,}")
                mdl.save_model(os.path.join(work, "model", f"{tag}_{f}.txt"), num_iteration=mdl.best_iteration)
                models.append(Model("lgb", mdl, mdl.best_iteration))
                del dtr, dva
                gc.collect()
            del full
            gc.collect()
        p = np.zeros(len(sel), np.float32) if fallback is None else fallback.astype(np.float32).copy()
        for a0, b0 in chunks:
            live = np.ones(b0 - a0, bool) if fallback is None else fallback[a0:b0] >= cfg["skip2"]
            if not live.any():
                continue
            Xc = make_x(a0, b0, None if fallback is None else live)[live]
            fs = f_s[a0:b0][live]
            out = np.empty(len(fs), np.float32)
            for fv, mm in ((0, models[1]), (1, models[0])):
                m = fs == fv
                if m.any():
                    out[m] = mm.predict(Xc[m])
            m = fs == 2
            if m.any():
                out[m] = 0.5 * sum(mm.predict(Xc[m]) for mm in models)
            view = p[a0:b0]
            view[live] = out
        log(f"{tag}: out-of-fold predictions done")
        imp = models[0].gain(names)
        return models, p, sorted(imp.items(), key=lambda kv: -kv[1])[:30]

    def evaluate(p):
        """Threshold tuned on out-of-fold folds 0/1 (with exclusivity), reported on the validation fold."""
        rep, ev = {}, {}
        for name, fs in (("oof", (0, 1)), ("val", (2,))):
            s1s = np.concatenate([s1_of[k] for k in fs])
            dense = np.full(n1, -1, np.int64)
            dense[s1s] = np.arange(len(s1s))
            msk = np.isin(f_s, fs)
            qi = dense[pq_s[msk]]
            ev[name] = (qi, y[msk], p[msk], exclusive(pt_s[msk], p[msk]), nt_all[s1s], s1s)
        qi, yy, pp, kx, nt, _ = ev["oof"]
        tau, sweep = tune_tau(qi, yy, pp, kx, nt)
        rep["tau"] = tau
        rep["oof_f05"] = sweep[tau]
        rep["oof_f05_no_excl"] = macro_f05(qi, yy, pp >= tau, nt)
        qi, yy, pp, kx, nt, s1s = ev["val"]
        rep["val_f05"] = macro_f05(qi, yy, kx & (pp >= tau), nt)
        rep["val_oracle"] = macro_f05(qi, yy, yy.copy(), nt)
        rep["val_sweep"] = {t: macro_f05(qi, yy, kx & (pp >= t), nt)
                            for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)}
        ctry = ctry_all[s1s]
        for cname in np.unique(ctry):
            cm = ctry == cname
            dense = np.full(len(s1s), -1, np.int64)
            dense[cm] = np.arange(int(cm.sum()))
            pm = dense[qi] >= 0
            rep[f"val_f05_{cname}"] = macro_f05(dense[qi][pm], yy[pm], (kx & (pp >= tau))[pm], nt[cm])
        return tau, rep

    base = {"pairs_per_s1": len(G["pq"]) / n1, "pair_recall": float(y_all.sum() / len(true_key)),
            "targets_per_s1_train": tgt_per_s1}
    models1, p1, top1 = fit_stage("lgb", lambda a0, b0, live=None: featurize(G, sel[a0:b0], live), FEATS)
    np.save(os.path.join(work, "model", "val_p1.npy"), p1)
    # validation pairs for cross-run ensembling (keys, labels, fold) and true counts of sampled S1
    np.save(os.path.join(work, "model", "val_keys.npy"), (pq_s.astype(np.int64) << 32) | pt_s.astype(np.int64))
    np.save(os.path.join(work, "model", "val_y.npy"), y)
    np.save(os.path.join(work, "model", "val_fold.npy"), f_s)
    s1_all = np.flatnonzero(fold >= 0)
    np.save(os.path.join(work, "model", "val_s1.npy"), np.stack([s1_all, fold[s1_all], nt_all[s1_all]]).astype(np.int64))
    tau1, rep1 = evaluate(p1)
    meta = {"tau": tau1, "feats": FEATS, "cfg": cfg, "stage2": False, "report": {**base, **rep1},
            "top_features": top1}
    with open(os.path.join(work, "model", "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    log("STAGE 1 REPORT " + json.dumps(meta["report"], indent=1))
    log("stage-1 top features: " + ", ".join(k for k, _ in top1[:15]))
    if cfg.get("stage2", True):
        tcomp = cfg.get("tcomp", False)
        if tcomp:  # stage-1 probabilities for every graph pair (out-of-fold where the S1 was trained on)
            p_all = np.zeros(len(G["pq"]), np.float32)
            p_all[sel] = p1
            rest = np.flatnonzero(pf < 0)  # whole S1 groups outside the sample
            m0 = models1[0]
            for a0, b0 in group_chunks(G["pq"][rest], cfg["chunk"]):
                p_all[rest[a0:b0]] = m0.predict(featurize(G, rest[a0:b0]))
                log(f"  full-graph stage-1 {b0:,}/{len(rest):,}")
            TC = target_p_stats(G, p_all, cfg.get("cprior", False), rows=sel)
            del p_all

        def x2(a0, b0, live=None):
            parts = [featurize(G, sel[a0:b0], live), collective(G, sel[a0:b0], p1[a0:b0])]
            if tcomp:
                parts.append(TC[a0:b0])
            return np.hstack(parts)
        tc_names = (TCOMP_NAMES + (CPRIOR_NAMES if cfg.get("cprior") else [])) if tcomp else []
        models2, p2, top2 = fit_stage("lgb2", x2, FEATS + COL_NAMES + tc_names,
                                      fallback=p1)
        np.save(os.path.join(work, "model", "val_p2.npy"), p2)
        tau2, rep2 = evaluate(p2)
        meta.update(tau2=tau2, stage2=True, tcomp=tcomp, cprior=bool(tcomp and cfg.get("cprior")),
                    report2=rep2, top_features2=top2)
        with open(os.path.join(work, "model", "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        log("STAGE 2 REPORT " + json.dumps(rep2, indent=1))
        log("stage-2 top features: " + ", ".join(k for k, _ in top2[:15]))


# ----------------------------------------------------------------------------- test stage
def run_test(data, work, cfg, tau=None):
    md = os.path.join(work, "model")
    with open(os.path.join(md, "translit.json"), encoding="utf-8") as fh:
        tables = json.load(fh)
    with open(os.path.join(md, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["feats"] == FEATS, "feature list changed since training"
    mc = meta["cfg"]  # build test features exactly like the models were trained (defaults = older runs)
    cfg = {**cfg, "norm": mc.get("norm", 1), "relfreq": mc.get("relfreq", False), "core": mc.get("core", False),
           "core_thr": mc.get("core_thr", 0.02)}
    models1 = [Model.load(md, f"lgb_{f}") for f in (0, 1)]
    stage2 = bool(meta.get("stage2")) and all(
        os.path.exists(os.path.join(md, f"lgb2_{f}.txt")) or os.path.exists(os.path.join(md, f"lgb2_{f}.json"))
        for f in (0, 1))
    models2 = [Model.load(md, f"lgb2_{f}") for f in (0, 1)] if stage2 else []
    rec = load_split(data, "test")
    if cfg.get("subsample", 1.0) < 1.0:
        rec = subset(rec, unit_hash(rec["ids"].tolist(), "sub") < cfg["subsample"])
    ctry = pd.Series(rec["countries"][:rec["n1"]]).value_counts()
    ctt = pd.Series(rec["countries"][rec["n1"]:]).value_counts()
    log("test targets per S1 by country: " + json.dumps({c: round(ctt.get(c, 0) / v, 2) for c, v in ctry.items()}))
    s1_ctry = np.array(rec["countries"][:rec["n1"]], dtype=object)
    G = build_graph(rec, tables, cfg)
    P = len(G["pq"])
    chunks = group_chunks(G["pq"], cfg["chunk"])

    def score(models, make_x, tag, fallback=None):
        p = np.zeros(P, np.float32) if fallback is None else fallback.astype(np.float32).copy()
        skip2 = meta["cfg"].get("skip2", 0.005)
        for a0, b0 in chunks:
            if time.time() > DEADLINE:
                log(f"WARNING: time budget reached in {tag} at {a0:,}/{P:,} pairs")
                return p, False
            live = np.ones(b0 - a0, bool) if fallback is None else fallback[a0:b0] >= skip2
            if not live.any():
                continue
            X = make_x(a0, b0, None if fallback is None else live)[live]
            view = p[a0:b0]
            view[live] = np.mean([m.predict(X) for m in models], 0)
            log(f"  {tag} scored {b0:,}/{P:,} ({int(live.sum()):,} live)")
        return p, True

    def emit(p, tau_, out, variants):
        os.makedirs(out, exist_ok=True)
        write_outputs(rec["ids"], G["n1"], G["pq"], G["pt"], p, tau_, out, countries=s1_ctry)
        for t in variants:
            write_outputs(rec["ids"], G["n1"], G["pq"], G["pt"], p, t, out, cand=False, suffix=f"_t{t:.2f}")

    tau1 = float(meta["tau"] if tau is None else tau)
    p1, ok1 = score(models1, lambda a0, b0, live=None: featurize(G, np.arange(a0, b0), live), "stage1")
    np.save(os.path.join(work, "test_p1.npy"), p1)
    np.save(os.path.join(work, "test_pq.npy"), G["pq"])
    np.save(os.path.join(work, "test_pt.npy"), G["pt"])
    emit(p1, tau1, os.path.join(work, "output_s1"), sorted({round(tau1 + d, 2) for d in (-0.05, 0.05, 0.1)}))
    if stage2 and ok1:
        tau2 = float(meta["tau2"] if tau is None else tau)
        TC = target_p_stats(G, p1, meta.get("cprior", False)) if meta.get("tcomp") else None

        def x2(a0, b0, live=None):
            idx = np.arange(a0, b0)
            parts = [featurize(G, idx, live), collective(G, idx, p1[a0:b0])]
            if TC is not None:
                parts.append(TC[a0:b0])
            return np.hstack(parts)
        p2, ok2 = score(models2, x2, "stage2", fallback=p1)
        if ok2:
            np.save(os.path.join(work, "test_p2.npy"), p2)
            emit(p2, tau2, os.path.join(work, "output"), sorted({round(tau2 + d, 2) for d in (-0.05, 0.05, 0.1)}))
            return
    import shutil
    shutil.copytree(os.path.join(work, "output_s1"), os.path.join(work, "output"), dirs_exist_ok=True)
    log("final output = stage 1")


def write_outputs(ids, n1, pq, pt, p, tau, out, cand=True, suffix="", countries=None):
    keep = exclusive(pt, p) & (p >= tau)
    starts = np.zeros(n1 + 1, np.int64)
    starts[1:] = np.cumsum(np.bincount(pq, minlength=n1))
    tid = ids[pt]
    fc = open(os.path.join(out, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") if cand else None
    with open(os.path.join(out, f"matching_results{suffix}.tsv"), "w", encoding="utf-8", newline="\n") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        if fc:
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for q in range(n1):
            a, b = starts[q], starts[q + 1]
            if fc:
                fc.write(ids[q] + "\t" + ",".join(tid[a:b]) + "\n")
            fm.write(ids[q] + "\t" + ",".join(tid[a:b][keep[a:b]]) + "\n")
    if fc:
        fc.close()
    npred = np.bincount(pq, weights=keep, minlength=n1)
    log(f"wrote {out}/matching_results{suffix}: tau {tau}, matches {int(keep.sum()):,}, "
        f"empty S1 {(npred == 0).mean():.4f}, mean matches {npred.mean():.3f}")
    if countries is not None:
        for c in np.unique(countries):
            m = countries == c
            log(f"    {c}: S1 {int(m.sum()):,}, empty {(npred[m] == 0).mean():.4f}, mean matches {npred[m].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "train", "test"])
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="stop scoring after this many hours and still write valid outputs")
    args = ap.parse_args()
    cfg = {**CFG, **json.loads(args.cfg)}
    global DEADLINE
    if args.budget_hours:
        DEADLINE = T0 + args.budget_hours * 3600
    set_num_threads(NT)
    log(f"cpus {NT}; cfg {json.dumps(cfg)}")
    if args.stage in ("all", "train"):
        run_train(args.data, args.work, cfg)
        gc.collect()
    if args.stage in ("all", "test"):
        run_test(args.data, args.work, cfg, args.tau)
    log("done")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
