"""Apache-2.0 TabM matcher with hard-negative mining and epoch checkpoints."""
from pathlib import Path
import numpy as np
from .setmodel import sweep


def hard_negative_indices(labels,mask,first_stage,k_hard=24,k_random=8,seed=42,strategy="hard"):
    rng=np.random.default_rng(seed)
    selected=[];positive=0;negative=0
    width=labels.shape[1]
    for q in range(len(labels)):
        active=np.flatnonzero(mask[q])
        good=active[labels[q,active]>0]
        bad=active[labels[q,active]==0]
        if strategy=="hard":
            bad=bad[np.argsort(-first_stage[q,bad],kind="stable")]
        elif strategy=="random":
            bad=rng.permutation(bad)
        else:
            raise ValueError(strategy)
        hard=bad[:k_hard]
        rest=bad[k_hard:]
        random_part=rng.choice(rest,size=min(k_random,len(rest)),replace=False) if len(rest) else np.empty(0,dtype=int)
        positions=np.r_[good,hard,random_part]
        selected.extend((q*width+positions).tolist())
        positive+=len(good);negative+=len(hard)+len(random_part)
    return np.asarray(selected,dtype=np.int64),{"positives":positive,"mined_hard_negatives":negative,
             "hard_per_query":k_hard,"random_per_query":k_random,"strategy":strategy}


def make_model(n_features,k=8):
    from tabm import TabM
    from rtdl_num_embeddings import LinearReLUEmbeddings
    return TabM.make(n_num_features=n_features,num_embeddings=LinearReLUEmbeddings(n_features,d_embedding=8),
                     d_out=1,k=k,n_blocks=2,d_block=128)


def _standardize_fit(x,indices):
    sample=x.reshape(-1,x.shape[-1])[indices]
    mean=sample.mean(0).astype(np.float32)
    std=np.maximum(sample.std(0),.05).astype(np.float32)
    return mean,std


def predict_tabm(model,x,mask,mean,std,device="cuda",batch_size=32768):
    import torch
    shape=mask.shape
    out=np.zeros(shape,dtype=np.float32)
    flat=x.reshape(-1,x.shape[-1])
    positions=np.flatnonzero(mask.reshape(-1))
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(positions),batch_size):
            ids=positions[start:start+batch_size]
            block=np.clip((flat[ids]-mean)/std,-5,5)
            data=torch.from_numpy(block).to(device)
            with torch.autocast(device_type=device.split(':')[0],enabled=device.startswith("cuda")):
                logits=model(data).squeeze(-1)
            out.reshape(-1)[ids]=torch.sigmoid(logits.float()).mean(1).cpu().numpy()
    return out


def train_tabm(x,y,mask,first_stage,true_counts,fit_end,tune_end,checkpoint,
               epochs=12,batch_size=8192,device="cuda",seed=42,strategy="hard"):
    import torch
    import torch.nn.functional as F
    checkpoint=Path(checkpoint)
    indices,mining=hard_negative_indices(y[:fit_end],mask[:fit_end],first_stage[:fit_end],seed=seed,strategy=strategy)
    mean,std=_standardize_fit(x[:fit_end],indices)
    np.savez(checkpoint.with_suffix(".scaler.npz"),mean=mean,std=std)
    model=make_model(x.shape[-1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    scaler=torch.amp.GradScaler(device,enabled=device.startswith("cuda"))
    start_epoch=0;best=-1.;best_threshold=.5;best_state=None
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch=state["epoch"]+1
        best=state["best_score"];best_threshold=state["best_threshold"]
        best_state=state["best_state"]
    flat_x=x[:fit_end].reshape(-1,x.shape[-1]);flat_y=y[:fit_end].reshape(-1)
    for epoch in range(start_epoch,epochs):
        model.train()
        order=np.random.default_rng(seed+epoch).permutation(len(indices))
        total_loss=0.
        for start in range(0,len(order),batch_size):
            ids=indices[order[start:start+batch_size]]
            block=np.clip((flat_x[ids]-mean)/std,-5,5)
            xb=torch.from_numpy(block).to(device)
            yb=torch.from_numpy(flat_y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(':')[0],enabled=device.startswith("cuda")):
                logits=model(xb).squeeze(-1)
                loss=F.binary_cross_entropy_with_logits(logits,yb[:,None].expand_as(logits),
                         pos_weight=torch.as_tensor(2.,device=device))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            total_loss+=loss.detach().item()
        scheduler.step()
        tune=predict_tabm(model,x[fit_end:tune_end],mask[fit_end:tune_end],mean,std,device)
        threshold,score=sweep(tune,y[fit_end:tune_end],mask[fit_end:tune_end],true_counts[fit_end:tune_end])
        if score>best:
            best=score;best_threshold=threshold
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        temporary=checkpoint.with_suffix(".writing.pt")
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
             scheduler=scheduler.state_dict(),best_score=best,best_threshold=best_threshold,
             best_state=best_state,mining=mining),temporary)
        temporary.replace(checkpoint)
        print(f"TabM epoch {epoch+1}/{epochs}: loss={total_loss/max(1,(len(order)+batch_size-1)//batch_size):.4f} tune-F0.5={score:.6f} best={best:.6f}",flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model,mean,std,best_threshold,best,mining


def load_tabm(checkpoint,n_features,device="cuda"):
    import torch
    checkpoint=Path(checkpoint)
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=make_model(n_features).to(device)
    model.load_state_dict(state["best_state"])
    model.eval()
    with np.load(checkpoint.with_suffix(".scaler.npz")) as values:
        mean=values["mean"];std=values["std"]
    return model,mean,std
