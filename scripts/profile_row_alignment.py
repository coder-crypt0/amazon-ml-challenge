"""Check whether supplied source order accidentally carries entity linkage signal."""
from pathlib import Path
import polars as pl
import numpy as np
from scipy.stats import spearmanr

base=Path("student_resource/dataset/train")
reference=pl.read_csv(base/"train_source1.tsv",separator="\t",n_rows=50000).with_row_index("s1_row")
truth=(pl.scan_csv(base/"train_ground_truth.tsv",separator="\t")
       .join(reference.select("entity_id","s1_row").lazy(),left_on="source1_entity_id",right_on="entity_id")
       .select("source1_entity_id","s1_row","matched_entity_ids").collect())
pairs=(truth.with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").filter(pl.col("matched_entity_ids").str.starts_with("S"))
       .select("source1_entity_id","s1_row",pl.col("matched_entity_ids").alias("target_id")))
for source in (2,3):
    ids=pairs.filter(pl.col("target_id").str.starts_with(f"S{source}-"))
    rows=(pl.scan_csv(base/f"train_source{source}.tsv",separator="\t")
          .with_row_index("target_row")
          .filter(pl.col("entity_id").is_in(ids["target_id"]))
          .select("entity_id","target_row").collect())
    joined=ids.join(rows,left_on="target_id",right_on="entity_id")
    x=joined["s1_row"].to_numpy();y=joined["target_row"].to_numpy()
    sibling=(joined.group_by("source1_entity_id")
             .agg(pl.col("target_row").min().alias("low"),
                  pl.col("target_row").max().alias("high"),pl.len().alias("count"))
             .filter(pl.col("count")>=2))
    distance=(sibling["high"]-sibling["low"]).to_numpy()
    print(source,"pairs",len(joined),"pearson",float(np.corrcoef(x,y)[0,1]),
          "spearman",float(spearmanr(x,y).statistic),
          "sibling_median_row_span",float(np.median(distance)) if len(distance) else None,
          "target_rows",int(rows["target_row"].max())+1,flush=True)
