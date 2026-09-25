"""Country-partitioned char TF-IDF retrieval with bounded postings.

The corpus contains only supplied records. Search runs in C++ through sparse_dot_topn.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from sparse_dot_topn import sp_matmul_topn


class HashView:
    def __init__(self, country, field, row_ids, matrix, idf, keep, grams, dimension, cap,
                 sort_tokens=False,analyzer="char",alias_target=False):
        self.country=country;self.field=field;self.row_ids=row_ids
        self.matrix=matrix;self.idf=idf;self.keep=keep
        self.grams=tuple(grams);self.dimension=dimension;self.cap=cap
        self.sort_tokens=bool(sort_tokens)
        self.analyzer=analyzer;self.alias_target=bool(alias_target)
        self.vectorizer=HashingVectorizer(analyzer=analyzer,ngram_range=self.grams,
                    n_features=dimension,alternate_sign=False,binary=True,norm=None,dtype=np.float32)

    @classmethod
    def build(cls, corpus, country, field, row_ids, grams=(4,4), dimension=1<<18, cap=1000,
              sort_tokens=False,analyzer="char",alias_target=False,channel=None):
        vectorizer=HashingVectorizer(analyzer=analyzer,ngram_range=grams,n_features=dimension,
                alternate_sign=False,binary=True,norm=None,dtype=np.float32)
        text=corpus.names if field=="name" else corpus.addresses
        def lookup(i):
            value=text[int(i)]
            if alias_target and channel is not None:
                value=channel.canonical_target(value,corpus.source[int(i)],field)
            return " ".join(sorted(value.split())) if sort_tokens else value
        raw=vectorizer.transform(lookup(i) for i in row_ids)
        df=np.bincount(raw.indices,minlength=dimension)
        keep=(df>=2)&(df<=cap)
        idf=(np.log((1+len(row_ids))/(1+df))+1).astype(np.float32)
        raw.data*=keep[raw.indices]*idf[raw.indices]
        raw.eliminate_zeros()
        l2_normalize(raw,copy=False)
        transposed=raw.T.tocsr()
        return cls(country,field,np.asarray(row_ids,dtype=np.uint32),transposed,idf,keep,grams,
                   dimension,cap,sort_tokens,analyzer,alias_target)

    def transform_queries(self,texts):
        matrix=self.vectorizer.transform((" ".join(sorted(text.split())) for text in texts) if self.sort_tokens else texts)
        matrix.data*=self.keep[matrix.indices]*self.idf[matrix.indices]
        matrix.eliminate_zeros()
        l2_normalize(matrix,copy=False)
        return matrix

    def search(self,texts,top_k=64,threads=4):
        if not texts:
            return sparse.csr_matrix((0,len(self.row_ids)),dtype=np.float32)
        q=self.transform_queries(texts)
        return sp_matmul_topn(q,self.matrix,top_n=top_k,n_threads=threads,sort=True)

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        sparse.save_npz(directory/"matrix.npz",self.matrix,compressed=False)
        np.save(directory/"rows.npy",self.row_ids)
        np.save(directory/"idf.npy",self.idf)
        np.save(directory/"keep.npy",self.keep)
        (directory/"metadata.json").write_text(json.dumps(dict(country=self.country,field=self.field,
            grams=self.grams,dimension=self.dimension,cap=self.cap,sorted_tokens=self.sort_tokens,
            analyzer=self.analyzer,alias_target=self.alias_target,records=len(self.row_ids),
            postings=self.matrix.nnz),indent=2))
        (directory/"COMPLETE").write_text("complete\n")

    @classmethod
    def load(cls,directory):
        directory=Path(directory)
        if not (directory/"COMPLETE").exists():raise FileNotFoundError(directory/"COMPLETE")
        info=json.loads((directory/"metadata.json").read_text())
        return cls(info["country"],info["field"],np.load(directory/"rows.npy",mmap_mode="r"),
                   sparse.load_npz(directory/"matrix.npz"),np.load(directory/"idf.npy",mmap_mode="r"),
                   np.load(directory/"keep.npy",mmap_mode="r"),info["grams"],info["dimension"],info["cap"],
                   info.get("sorted_tokens",False),info.get("analyzer","char"),info.get("alias_target",False))


def build_views(corpus,root,cap_rate=.0025,dimension=1<<20,channel=None,use_word=True):
    root=Path(root);views={}
    for country,rows in corpus.by_country().items():
        cap=max(20,min(15000,int(len(rows)*cap_rate)))
        for field,grams in (("name",(3,4)),("address",(3,4))):
            variants=["original","sorted"]
            if use_word:
                variants.append("rare_words" if field=="name" else "number_words")
            for variant in variants:
                directory=root/country/field/variant
                sorted_tokens=variant=="sorted"
                word_view=variant in ("rare_words","number_words")
                view_grams=(1,2) if word_view else grams
                analyzer="word" if word_view else "char"
                alias_target=word_view and channel is not None
                if (directory/"COMPLETE").exists():
                    view=HashView.load(directory)
                    if (view.cap!=cap or view.dimension!=dimension or len(view.row_ids)!=len(rows)
                            or view.grams!=view_grams or view.sort_tokens!=sorted_tokens
                            or view.analyzer!=analyzer or view.alias_target!=alias_target):
                        raise ValueError("Search index config/data changed; select a new run directory")
                else:
                    view=HashView.build(corpus,country,field,rows,view_grams,dimension,cap,
                                        sorted_tokens,analyzer,alias_target,channel)
                    view.save(directory)
                views[country,field,variant]=view
                print(f"Ready {country}/{field}/{variant}: {len(rows):,} records, {view.matrix.nnz:,} postings",flush=True)
    return views


def retrieve_batch(queries,query_indices,views,name_k=64,address_k=128,threads=4):
    """Return row-wise union of independently ranked name and address candidates."""
    groups={}
    for local,global_row in enumerate(query_indices):
        groups.setdefault(queries.countries[int(global_row)],[]).append(local)
    result=[None]*len(query_indices)
    for country,locals_ in groups.items():
        if not any(key[0]==country and key[1]=="name" for key in views):
            for i in locals_:result[i]={}
            continue
        qrows=[int(query_indices[i]) for i in locals_]
        named=[view for key,view in views.items() if key[0]==country and key[1]=="name"]
        addressed=[view for key,view in views.items() if key[0]==country and key[1]=="address"]
        name_results=[view.search([queries.names[i] for i in qrows],
                                  min(name_k,16) if view.analyzer=="word" else name_k,threads) for view in named]
        address_results=[view.search([queries.addresses[i] for i in qrows],
                                     min(address_k,24) if view.analyzer=="word" else address_k,threads) for view in addressed]
        components=([(matrix,view,0) for matrix,view in zip(name_results,named)]
                    +[(matrix,view,1) for matrix,view in zip(address_results,addressed)])
        for local_index,position in enumerate(locals_):
            scored={}
            for matrix,view,col in components:
                start,end=matrix.indptr[local_index:local_index+2]
                for rid,score in zip(matrix.indices[start:end],matrix.data[start:end]):
                    global_target=int(view.row_ids[rid])
                    entry=scored.setdefault(global_target,[0.,0.])
                    entry[col]=max(entry[col],float(score))
            result[position]=scored
    return result
