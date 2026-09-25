"""A permutation-equivariant GPU matcher over each query's candidate set."""
import copy
import json
from pathlib import Path
import numpy as np


def make_network(n_features,hidden=96,layers=2):
    import torch
    from torch import nn
    class SetMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.project=nn.Sequential(nn.LayerNorm(n_features),nn.Linear(n_features,hidden),nn.GELU())
            encoder=nn.TransformerEncoderLayer(hidden,nhead=4,dim_feedforward=hidden*2,
                    dropout=.08,batch_first=True,norm_first=True)
            self.context=nn.TransformerEncoder(encoder,num_layers=layers,enable_nested_tensor=False)
            self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        def forward(self,features,mask):
            safe_mask=mask.clone()
            safe_mask[~safe_mask.any(1),0]=True
            hidden=self.context(self.project(features),src_key_padding_mask=~safe_mask)
            return self.out(hidden).squeeze(-1).masked_fill(~mask,-30.)
    return SetMatcher()


def pack_queries(features,offsets,labels=None,k=64):
    n=len(offsets)-1
    width=features.shape[1]
    x=np.zeros((n,k,width),dtype=np.float32)
    mask=np.zeros((n,k),dtype=bool)
    y=np.zeros((n,k),dtype=np.float32)
    candidate_rows=np.full((n,k),-1,dtype=np.int64)
    for q,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        start,end=int(start),int(end)
        count=min(k,end-start)
        if count:
            x[q,:count]=features[start:start+count]
            mask[q,:count]=True
            candidate_rows[q,:count]=np.arange(start,start+count)
            if labels is not None:y[q,:count]=labels[start:start+count]
    return x,y,mask,candidate_rows


def macro_f05(true_counts,predicted,correct):
    denominator=.25*np.asarray(true_counts)+np.asarray(predicted)
    return float(np.mean(np.divide(1.25*np.asarray(correct),denominator,
                    out=np.ones_like(denominator,dtype=np.float64),where=denominator>0)))


def sweep(scores,labels,mask,true_counts,thresholds=None):
    thresholds=np.arange(.25,.991,.025) if thresholds is None else thresholds
    best=(-1.,-1.)
    for threshold in thresholds:
        predictions=(scores>=threshold)&mask
        correct=(predictions&(labels>0)).sum(1)
        total=predictions.sum(1)
        result=macro_f05(true_counts,total,correct)
        if result>best[1]:best=(float(threshold),result)
    return best


def train_set_model(x,y,mask,true_counts,train_end,tune_end,checkpoint,epochs=20,batch_size=128,
                    learning_rate=.0007,device="cuda",seed=42):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model=make_network(x.shape[-1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.015)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    scaler=torch.amp.GradScaler(device,enabled=device.startswith("cuda"))
    checkpoint=Path(checkpoint)
    best=-1.;best_threshold=.5;start_epoch=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch=state["epoch"]+1
        best=state["best_score"]
        best_threshold=state["best_threshold"]
        if state.get("best_state") is not None:
            best_state=state["best_state"]
    train_order=np.arange(train_end)
    for epoch in range(start_epoch,epochs):
        model.train()
        np.random.default_rng(seed+epoch).shuffle(train_order)
        loss_total=0.
        for start in range(0,train_end,batch_size):
            ids=train_order[start:start+batch_size]
            xb=torch.from_numpy(x[ids]).to(device)
            yb=torch.from_numpy(y[ids]).to(device)
            mb=torch.from_numpy(mask[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(':')[0],enabled=device.startswith("cuda")):
                logits=model(xb,mb)
                positive_weight=torch.tensor(2.,device=device)
                loss=F.binary_cross_entropy_with_logits(logits[mb],yb[mb],pos_weight=positive_weight)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            loss_total+=loss.detach().item()
        scheduler.step()
        tune_scores=predict_set_model(model,x[train_end:tune_end],mask[train_end:tune_end],device=device,batch_size=batch_size)
        threshold,score=sweep(tune_scores,y[train_end:tune_end],mask[train_end:tune_end],true_counts[train_end:tune_end])
        if score>best:
            best=score;best_threshold=threshold
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        temporary=checkpoint.with_suffix('.tmp')
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),best_score=best,best_threshold=best_threshold,
                best_state=best_state),temporary)
        temporary.replace(checkpoint)
        print(f"Set matcher epoch {epoch+1}/{epochs}: loss={loss_total/max(1,train_end//batch_size):.4f}, tune-F0.5={score:.6f}, best={best:.6f}",flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model,best_threshold,best


def predict_set_model(model,x,mask,device="cuda",batch_size=256):
    import torch
    outputs=np.zeros(mask.shape,dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            stop=min(start+batch_size,len(x))
            xb=torch.from_numpy(np.asarray(x[start:stop])).to(device)
            mb=torch.from_numpy(mask[start:stop]).to(device)
            with torch.autocast(device_type=device.split(':')[0],enabled=device.startswith("cuda")):
                output=model(xb,mb)
            outputs[start:stop]=torch.sigmoid(output.float()).cpu().numpy()
    return outputs
