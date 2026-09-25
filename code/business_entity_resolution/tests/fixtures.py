"""Synthetic data exclusively for pipeline verification."""
import csv
from pathlib import Path


def make_dataset(root,train_queries=100,test_queries=15):
    root=Path(root)
    for split,count in (("train",train_queries),("test",test_queries)):
        directory=root/split;directory.mkdir(parents=True,exist_ok=True)
        sources={1:[],2:[],3:[]};truth=[]
        for i in range(count):
            letters=chr(97+i//26)+chr(97+i%26)
            country="France" if split=="test" and i%3==0 else ("India" if i%2 else "US")
            name=f"Acme {letters} Workshop"
            address=f"{100+i} Pine Road, District {letters}"
            sources[1].append([f"S1-{i}",name,address,country])
            positives=[]
            if i%10:
                for source in (2,3):
                    entity_id=f"S{source}-{i}"
                    sources[source].append([entity_id,name+" Limited",address.replace("Road","Rd"),country])
                    positives.append(entity_id)
            sources[2].append([f"S2-x{i}",f"Unrelated {letters} Retail",f"{900+i} Cedar Street",country])
            truth.append([f"S1-{i}",",".join(positives)])
        for source,rows in sources.items():
            with (directory/f"{split}_source{source}.tsv").open("w",encoding="utf-8",newline="") as handle:
                writer=csv.writer(handle,delimiter="\t",lineterminator="\n")
                writer.writerow(["entity_id","business_name","business_address","country"])
                writer.writerows(rows)
        if split=="train":
            with (directory/"train_ground_truth.tsv").open("w",encoding="utf-8",newline="") as handle:
                writer=csv.writer(handle,delimiter="\t",lineterminator="\n")
                writer.writerow(["source1_entity_id","matched_entity_ids"])
                writer.writerows(truth)
    return root
