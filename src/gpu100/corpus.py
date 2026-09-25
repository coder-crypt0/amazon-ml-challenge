"""Compact, open-country text corpus for vectorized matching."""
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from sys import intern

import numpy as np
from anyascii import anyascii

LEGAL = frozenset("ltd limited llc inc incorporated corporation corp co company plc pvt private llp sarl sas sa sci eurl pte bv gmbh and the".split())
ADDR_MAP = {"rd":"road", "st":"street", "ave":"avenue", "av":"avenue",
 "blvd":"boulevard", "ln":"lane", "dr":"drive", "ct":"court", "hwy":"highway",
 "nr":"near", "opp":"opposite", "marg":"road", "r":"rue", "bd":"boulevard",
 "boul":"boulevard", "rte":"route", "che":"chemin", "pl":"place",
 "apt":"apartment", "ste":"suite", "fl":"floor", "flr":"floor"}
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SPACE = re.compile(r"\s+")
NUMBER = re.compile(r"\d+")
DOMAIN = re.compile(r"\b(?:www\.)?([\w-]+)\.(?:com|in|fr|org|net|co)\b", re.I)


def normalize(value):
    if not value:
        return ""
    value = str(value)
    value = value.lower() if value.isascii() else anyascii(value).lower()
    value = value.replace("&", " and ")
    value = PUNCT.sub(" ", value)
    return SPACE.sub(" ", value).strip()


def name_text(value):
    value = DOMAIN.sub(lambda match: match.group(1), str(value or ""))
    return " ".join(token for token in normalize(value).split() if token not in LEGAL)


def address_text(value):
    return " ".join(ADDR_MAP.get(token,token) for token in normalize(value).split() if token != "null")


@dataclass
class Corpus:
    ids: list
    names: list
    addresses: list
    raw_names: list
    raw_addresses: list
    countries: list
    house: np.ndarray
    postcode: np.ndarray
    source: np.ndarray
    name_len: np.ndarray
    address_len: np.ndarray
    name_hash: np.ndarray
    address_hash: np.ndarray

    @classmethod
    def from_tsv(cls, paths):
        ids=[];names=[];addresses=[];raw_names=[];raw_addresses=[]
        countries=[];house=[];postcodes=[];sources=[]
        name_len=[];address_len=[];name_hash=[];address_hash=[]
        def stable_hash(value):
            return int.from_bytes(hashlib.blake2b(value.encode("utf-8"),digest_size=8).digest(),"little") if value else 0
        for path in map(Path,paths):
            with path.open(encoding="utf-8-sig",newline="") as handle:
                reader=csv.DictReader(handle,delimiter="\t")
                if reader.fieldnames != ["entity_id","business_name","business_address","country"]:
                    raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Malformed record in {path}")
                    ids.append(row["entity_id"])
                    raw_names.append(row["business_name"])
                    raw_addresses.append(row["business_address"])
                    clean_name=name_text(row["business_name"])
                    names.append(clean_name)
                    text=address_text(row["business_address"])
                    addresses.append(text)
                    name_len.append(len(clean_name));address_len.append(len(text))
                    name_hash.append(stable_hash(clean_name));address_hash.append(stable_hash(text))
                    countries.append(intern(row["country"]))
                    digits=NUMBER.findall(text)
                    house.append(int(digits[0]) % 4294967296 if digits else 0)
                    long_digits=next((int(x) for x in digits if len(x) in (5,6)),0)
                    postcodes.append(long_digits % 4294967296)
                    sources.append(2 if row["entity_id"].startswith("S2-") else 3 if row["entity_id"].startswith("S3-") else 1)
                    if len(ids)%500000==0:
                        print(f"Normalized {len(ids):,} records",flush=True)
        return cls(ids,names,addresses,raw_names,raw_addresses,countries,np.asarray(house,dtype=np.uint32),
                   np.asarray(postcodes,dtype=np.uint32),np.asarray(sources,dtype=np.uint8),
                   np.asarray(name_len,dtype=np.uint16),np.asarray(address_len,dtype=np.uint16),
                   np.asarray(name_hash,dtype=np.uint64),np.asarray(address_hash,dtype=np.uint64))

    def __len__(self):
        return len(self.ids)

    def by_country(self):
        groups={}
        for i,country in enumerate(self.countries):
            groups.setdefault(country,[]).append(i)
        return {country:np.asarray(indices,dtype=np.uint32) for country,indices in groups.items()}

    def save(self,path):
        import polars as pl
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        frame=pl.DataFrame({"entity_id":self.ids,"name":self.names,"address":self.addresses,
                            "raw_name":self.raw_names,"raw_address":self.raw_addresses,"country":self.countries,
                            "house":self.house,"postcode":self.postcode,"source":self.source,
                            "name_len":self.name_len,"address_len":self.address_len,
                            "name_hash":self.name_hash,"address_hash":self.address_hash})
        frame.write_ipc(path,compression="zstd")

    @classmethod
    def load(cls,path):
        import polars as pl
        frame=pl.read_ipc(path,memory_map=False)
        return cls(frame["entity_id"].to_list(),frame["name"].to_list(),frame["address"].to_list(),
                   frame["raw_name"].to_list(),frame["raw_address"].to_list(),frame["country"].to_list(),
                   frame["house"].to_numpy(),frame["postcode"].to_numpy(),frame["source"].to_numpy(),
                   frame["name_len"].to_numpy(),frame["address_len"].to_numpy(),
                   frame["name_hash"].to_numpy(),frame["address_hash"].to_numpy())


def truth_for_queries(path,ids):
    wanted=set(ids)
    truth={}
    with Path(path).open(encoding="utf-8",newline="") as handle:
        reader=csv.DictReader(handle,delimiter="\t")
        for row in reader:
            q=row["source1_entity_id"]
            if q in wanted:
                truth[q]=set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    if set(truth)!=wanted:
        raise ValueError("Truth labels do not cover every selected reference")
    return truth
