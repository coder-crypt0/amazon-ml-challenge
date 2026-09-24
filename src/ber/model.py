"""Native LightGBM model and macro entity metrics."""
import numpy as np
from pathlib import Path
from .checkpoint import restore_file, persist_file


def macro_f05(truth, pred, query_ids=None):
    keys = list(query_ids) if query_ids is not None else list(dict.fromkeys([*truth,*pred]))
    values=[]
    for key in keys:
        t, p = set(truth.get(key,())), set(pred.get(key,()))
        values.append(1.25*len(t&p)/(.25*len(t)+len(p)) if t or p else 1.)
    return float(np.mean(values)) if values else 0.


def calibrate_threshold(probabilities, pairs, query_ids, truth, thresholds=None):
    thresholds = np.r_[np.arange(.05,.96,.025),.975,.99,.995] if thresholds is None else thresholds
    best=(-1.,-1.)
    for threshold in thresholds:
        pred={}
        for pair, p in zip(pairs,probabilities):
            if p>=threshold:
                pred.setdefault(pair[0],set()).add(pair[1])
        score=macro_f05(truth,pred,query_ids)
        if score>best[1]:
            best=(float(threshold),score)
    return best


class PairClassifier:
    def __init__(self, **params):
        self.params=params
        self.model=None

    def fit_classifier(self,X,y,seed=42,threads=6,checkpoint=None):
        import lightgbm as lgb
        from .features import FEATURE_NAMES
        params=dict(objective="binary",learning_rate=.045,num_leaves=47,min_data_in_leaf=40,
                    feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,
                    seed=seed,num_threads=threads,verbosity=-1,deterministic=True,force_col_wise=True)
        rounds=self.params.pop("n_estimators",450)
        params.update(self.params)
        dataset=lgb.Dataset(X,label=y,feature_name=list(FEATURE_NAMES),free_raw_data=False)
        if checkpoint and restore_file(checkpoint):
            self.model=lgb.Booster(model_file=str(checkpoint))
        done=self.model.current_iteration() if self.model is not None else 0
        while done<rounds:
            self.model=lgb.train(params,dataset,num_boost_round=min(50,rounds-done),init_model=self.model,keep_training_booster=True)
            new_done=self.model.current_iteration()
            if checkpoint:
                temp=Path(str(checkpoint)+".tmp")
                self.model.save_model(str(temp))
                temp.replace(checkpoint)
                persist_file(checkpoint)
            if new_done<=done:
                break
            done=new_done
        return self

    def predict_proba(self,X):
        if self.model is None:
            raise RuntimeError("Classifier is not fitted")
        return self.model.predict(X,num_threads=1)

    def save(self,path):
        self.model.save_model(str(path))

    @classmethod
    def load(cls,path):
        import lightgbm as lgb
        result=cls()
        result.model=lgb.Booster(model_file=str(path))
        return result


def train_xgboost(X,y,checkpoint,device="cuda",rounds=500,seed=42,threads=4):
    """Resume in 50-tree units; CUDA histogram training when requested."""
    import xgboost as xgb
    from .features import FEATURE_NAMES
    params=dict(objective="binary:logistic",tree_method="hist",device=device,max_depth=7,
                learning_rate=.045,min_child_weight=8,subsample=.9,colsample_bytree=.9,
                reg_lambda=3.,seed=seed,nthread=threads)
    train=xgb.QuantileDMatrix(X,label=y,feature_names=list(FEATURE_NAMES),max_bin=256)
    model=None
    if restore_file(checkpoint):
        model=xgb.Booster(model_file=str(checkpoint))
    done=model.num_boosted_rounds() if model is not None else 0
    while done<rounds:
        model=xgb.train(params,train,num_boost_round=min(50,rounds-done),xgb_model=model)
        done=model.num_boosted_rounds()
        temp=Path(checkpoint).with_name("xgb_writing.ubj")
        model.save_model(temp)
        temp.replace(checkpoint)
        persist_file(checkpoint)
        print(f"XGBoost checkpoint: {done}/{rounds} trees ({device})",flush=True)
    return model


class InferenceModel:
    def __init__(self,directory):
        import json
        directory=Path(directory)
        self.config=json.loads((directory/"calibration.json").read_text())
        self.lgb=PairClassifier.load(directory/"model.txt")
        self.weight=self.config.get("xgboost_weight",0.)
        self.xgb=None
        if self.weight:
            import xgboost as xgb
            self.xgb=xgb.Booster(model_file=str(directory/"xgboost.ubj"))
            self.xgb.set_param({"device":"cpu","nthread":1})

    def predict_proba(self,X):
        probabilities=self.lgb.predict_proba(X)
        if self.xgb is not None:
            probabilities=(1-self.weight)*probabilities+self.weight*self.xgb.inplace_predict(X)
        return probabilities
