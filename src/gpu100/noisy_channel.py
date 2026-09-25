"""Source-conditioned corruption model learned exclusively from fit positives."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

CHANNEL_FIELDS=("name","address")
CHANNEL_FEATURE_NAMES=("name_channel_cost","address_channel_cost","name_edit_support",
    "address_edit_support","name_token_equivalence","address_token_equivalence",
    "name_whole_alias","channel_observed")


def _op_key(operation,left,right):
    if operation.tag=="replace":
        return "R:"+left[operation.src_pos]+">"+right[operation.dest_pos]
    if operation.tag=="delete":
        return "D:"+left[operation.src_pos]
    return "I:"+right[operation.dest_pos]


class NoisyChannel:
    def __init__(self,op_counts,alias_counts,whole_aliases,fit_links):
        self.ops={key:Counter(values) for key,values in op_counts.items()}
        self.alias={key:dict(values) for key,values in alias_counts.items()}
        self.whole={key:set(values) for key,values in whole_aliases.items()}
        self.fit_links=fit_links
        self.total={key:sum(values.values()) for key,values in self.ops.items()}
        self.vocab={key:len(values) for key,values in self.ops.items()}
        self.default_cost={key:-math.log(.5/(self.total[key]+.5*(self.vocab[key]+1)))
                           for key in self.ops}
        self.edit_cost={key:{operation:-math.log((count+.5)/(self.total[key]+.5*(self.vocab[key]+1)))
                             for operation,count in values.items()} for key,values in self.ops.items()}
        self.alias_probability={key:{variant:count/(sum(variants.values())+2)
                                     for variant,count in variants.items()}
                                for key,variants in self.alias.items()}
        reverse=defaultdict(Counter)
        for key,variants in self.alias.items():
            source,field,reference_token=key.split(":",2)
            for observed_token,count in variants.items():
                reverse[(int(source),field,observed_token)][reference_token]+=count
        self.reverse_alias={}
        for key,candidates in reverse.items():
            winner,count=candidates.most_common(1)[0]
            if count>=3 and count/sum(candidates.values())>=.6:
                self.reverse_alias[key]=winner

    def canonical_target(self,text,source,field):
        return " ".join(self.reverse_alias.get((int(source),field,token),token)
                        for token in text.split())

    @classmethod
    def fit(cls,queries,targets,fit_rows,truth,min_alias_count=3,max_links=600000):
        wanted={target_id for qrow in fit_rows for target_id in truth[queries.ids[int(qrow)]]}
        target_lookup={entity_id:i for i,entity_id in enumerate(targets.ids) if entity_id in wanted}
        if len(target_lookup)!=len(wanted):
            raise ValueError("Fit truth references target IDs absent from the training corpus")
        ops=defaultdict(Counter)
        aliases=defaultdict(Counter)
        whole=defaultdict(Counter)
        fit_links=0
        for qrow in fit_rows:
            qrow=int(qrow)
            qid=queries.ids[qrow]
            for target_id in sorted(truth[qid]):
                trow=target_lookup[target_id]
                source=int(targets.source[trow])
                if source not in (2,3):
                    raise ValueError("Noisy-channel target must come from Source 2 or 3")
                for field in CHANNEL_FIELDS:
                    left=queries.names[qrow] if field=="name" else queries.addresses[qrow]
                    right=targets.names[trow] if field=="name" else targets.addresses[trow]
                    if not left or not right:
                        continue
                    left=left[:160];right=right[:160]
                    bucket=f"{source}:{field}"
                    for operation in Levenshtein.editops(left,right):
                        ops[bucket][_op_key(operation,left,right)]+=1
                    lt=set(left.split());rt=set(right.split())
                    unmatched=lt-rt
                    candidate_tokens=rt-lt
                    for token in unmatched:
                        if len(token)<3 or not candidate_tokens:
                            continue
                        nearest=max(candidate_tokens,key=lambda candidate:fuzz.ratio(token,candidate))
                        if fuzz.ratio(token,nearest)>=45:
                            aliases[f"{bucket}:{token}"][nearest]+=1
                    if field=="name" and fuzz.ratio(left,right)<65:
                        qaddress=queries.addresses[qrow]
                        taddress=targets.addresses[trow]
                        if qaddress and taddress and fuzz.token_set_ratio(qaddress,taddress)>=75:
                            whole[bucket][left+"\0"+right]+=1
                fit_links+=1
                if fit_links>=max_links:
                    break
            if fit_links>=max_links:
                break
        filtered={key:{variant:count for variant,count in variants.items() if count>=min_alias_count}
                  for key,variants in aliases.items()}
        filtered={key:values for key,values in filtered.items() if values}
        repeated={key:{pair for pair,count in variants.items() if count>=min_alias_count}
                  for key,variants in whole.items()}
        return cls(ops,filtered,repeated,fit_links)

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload={"fit_links":self.fit_links,"ops":{k:dict(v) for k,v in self.ops.items()},
                 "aliases":self.alias,"whole":{k:sorted(v) for k,v in self.whole.items()}}
        temp=path.with_suffix(".writing.json")
        temp.write_text(json.dumps(payload,separators=(",",":"),ensure_ascii=False),encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["ops"],data["aliases"],data["whole"],data["fit_links"])

    def profile(self):
        return {"fit_links":self.fit_links,"edit_operations":{key:self.total[key] for key in self.ops},
                "recurring_token_equivalences":{str(source):sum(len(v) for k,v in self.alias.items() if k.startswith(str(source)+":")) for source in (2,3)},
                "recurring_whole_name_aliases":{str(source):len(self.whole.get(f"{source}:name",())) for source in (2,3)}}

    def _field(self,left,right,source,field):
        if not left or not right:
            return 0.,0.,0.
        bucket=f"{source}:{field}"
        edit_cost=0.;supported=0
        counts=self.ops.get(bucket,{})
        costs=self.edit_cost.get(bucket,{})
        fallback=self.default_cost.get(bucket,0.)
        edit_operations=Levenshtein.editops(left[:160],right[:160])
        for operation in edit_operations:
            key=_op_key(operation,left,right)
            count=counts.get(key,0)
            if count>=3:
                supported+=1
            edit_cost+=costs.get(key,fallback)
        token_scores=[]
        right_tokens=set(right.split())
        for token in set(left.split()):
            if token in right_tokens:
                token_scores.append(1.)
                continue
            equivalents=self.alias_probability.get(f"{bucket}:{token}",{})
            token_scores.append(max((equivalents.get(candidate,0)
                                     for candidate in right_tokens),default=0.))
        return (edit_cost/max(1,max(len(left),len(right))),
                supported/max(1,len(edit_operations)),
                sum(token_scores)/max(1,len(token_scores)))

    def score_pair(self,query_name,query_address,target_name,target_address,source):
        name=self._field(query_name,target_name,source,"name")
        address=self._field(query_address,target_address,source,"address")
        whole=float(query_name+"\0"+target_name in self.whole.get(f"{source}:name",()))
        return (name[0],address[0],name[1],address[1],name[2],address[2],whole,1.)

    def score_batch(self,queries,targets,query_rows,target_rows,offsets,prior,anchor_flags,
                    top_per_query=8,probability_floor=.7):
        result=np.zeros((len(target_rows),len(CHANNEL_FEATURE_NAMES)),dtype=np.float32)
        for qrow,(start,end) in zip(query_rows,zip(offsets[:-1],offsets[1:])):
            qrow=int(qrow);start=int(start);end=int(end)
            qn=queries.names[qrow];qa=queries.addresses[qrow]
            for position in range(start,end):
                if position-start>=top_per_query and prior[position]<probability_floor:
                    continue
                trow=int(target_rows[position])
                result[position]=self.score_pair(qn,qa,targets.names[trow],targets.addresses[trow],
                                                  int(targets.source[trow]))
        return result
