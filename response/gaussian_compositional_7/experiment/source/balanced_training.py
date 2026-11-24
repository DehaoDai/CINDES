"""CINDES/DSM training with matched positive-observation minibatch schedules.

Both methods iterate over the same supplied positive observations in batches
of at most 256, including the final partial batch. CINDES adds two fresh
standard-Gaussian negatives per positive and averages BCE over the resulting
1:2 pooled labels. DSM uses per-coordinate denoising MSE. This matches the
maximum positive-data exposure, not computation or the selected durations.

No target generator, graph, potential, score, or evaluation sample is imported.
"""

import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

import l1_training as architecture
import training_loop as loop


R = 25.0
T = 3.0
T_MIN = .001
WIDTH = 64
DEPTH = 2
SOFTPLUS_BETA = 5.0
VALIDATION_DRAWS = 8
NORMALIZER_SIZE = 65536
CV_HORIZON = 6000
FULL_HORIZON = 8000


def stream_seeds(method, seed):
    if method not in ("cindes", "dsm"):
        raise ValueError("method must be cindes or dsm")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 1000000:
        raise ValueError("seed must be an integer in [0,1000000]")
    namespace = 110000000 + 1000 * seed
    common = {"positive_permutations": namespace + 2}
    if method == "cindes":
        return dict(common, initialization=namespace+1, refreshed_negatives=namespace+3,
                    validation_negatives=namespace+4, validation_normalizer=namespace+5)
    return dict(common, initialization=namespace+11, training_times=namespace+13,
                training_noise=namespace+14, validation_times=namespace+15,
                validation_noise=namespace+16)


def _inputs(train, device):
    train = torch.as_tensor(train).detach().to(device="cpu", dtype=torch.float32).contiguous()
    if train.ndim != 2 or min(train.shape) < 1 or not bool(torch.isfinite(train).all()):
        raise ValueError("Training observations must be a finite nonempty matrix")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return train, target


def _setup(train, method, coefficient, seed, target):
    if isinstance(coefficient, bool) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("l1_lambda must be finite and nonnegative")
    seeds = stream_seeds(method, seed)
    spec = architecture.model_spec(method, train.shape[1],
                                  "classifier" if method == "cindes" else "residual",
                                  activation="softplus", softplus_beta=SOFTPLUS_BETA)
    model = loop._initialize(lambda: architecture.build_model(spec), seeds["initialization"], target)
    weights = architecture.linear_weights(model)
    config = {
        "protocol": "positive_balanced_generic_softplus_training_v1", "method": method,
        "model_spec": spec, "d": train.shape[1], "seed": seed, "n_train": len(train),
        "device": str(target), "dtype": "float32", "activation": "softplus",
        "softplus_beta": SOFTPLUS_BETA, "network_width": WIDTH, "network_depth": DEPTH,
        "input_features": "raw", "optimizer": "Adam", "learning_rate": .001,
        "weight_decay": 0., "l1_lambda": float(coefficient),
        "l1_penalty_definition": "lambda * mean absolute value of all Linear weights, including output; biases excluded",
        "l1_optimization": "Adam gradient; no proximal thresholding or pruning",
        "positive_batch_semantics": "batch_size counts positive observations; no final partial batch is dropped",
        "same_positive_permutation_stream_across_methods_and_penalties": True,
        "rng_stream_seeds": seeds, "training_input_fingerprint": loop._fingerprint(train),
        "selection_uses_target_truth": False, "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "weight_parameter_count": sum(w.numel() for _, w in weights),
        "penalized_weight_names": [name for name, _ in weights],
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            "balanced_training.py": Path(__file__), "l1_training.py": Path(architecture.__file__),
            "training_loop.py": Path(loop.__file__), "subgaussian_methods.py": Path(loop.base.__file__),
        }.items()},
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
    }
    if method == "cindes":
        negative_rng = loop._generator(seeds["refreshed_negatives"])
        config.update({"R": R, "fake_ratio": 2, "negative_mode": "refresh",
                       "training_objective": "mean pooled 1:2 BCE plus normalized L1",
                       "batch_semantics": "B positives plus exactly 2B fresh Gaussian negatives; BCE over 3B labels"})

        def batch_loss(net, indices):
            positives = train[indices]
            negatives = torch.randn((2*len(indices), train.shape[1]), generator=negative_rng,
                                    dtype=train.dtype)
            points = torch.cat([positives, negatives]).to(target)
            labels = torch.cat([torch.ones(len(indices)), torch.zeros(2*len(indices))]).to(target)
            logits = net(points).squeeze(-1).clamp(-R, R)
            data_loss = F.binary_cross_entropy_with_logits(logits, labels)
            return architecture.penalized_objective(net, data_loss, coefficient), len(indices), 2*len(indices)
    else:
        time_rng = loop._generator(seeds["training_times"])
        noise_rng = loop._generator(seeds["training_noise"])
        config.update({"T": T, "t_min": T_MIN, "dsm_parameterization": "residual",
                       "dsm_time_frequencies": 8, "time_distribution": "log-uniform[.001,3]",
                       "training_objective": "mean squared sigma*score+epsilon over observations AND coordinates plus normalized L1",
                       "batch_semantics": "B positive observations, each with one fresh time/noise draw"})
        positives = train.to(target)

        def batch_loss(net, indices):
            clean = positives[indices.to(target)]
            times = torch.exp(torch.rand((len(indices), 1), generator=time_rng)
                              * (math.log(T)-math.log(T_MIN)) + math.log(T_MIN)).to(target)
            epsilon = torch.randn((len(indices), train.shape[1]), generator=noise_rng).to(target)
            sigma = torch.sqrt(-torch.expm1(-2*times))
            noisy = torch.exp(-times)*clean + sigma*epsilon
            data_loss = ((sigma*net(noisy, times)+epsilon)**2).mean()
            return architecture.penalized_objective(net, data_loss, coefficient), len(indices), 0
    return model, spec, config, seeds, batch_loss


def _validator(validation, method, seeds, target, normalizer_size):
    if method == "cindes":
        q = torch.randn((normalizer_size, validation.shape[1]),
                        generator=loop._generator(seeds["validation_normalizer"]))
        negative_bank = torch.randn((2*len(validation), validation.shape[1]),
                                    generator=loop._generator(seeds["validation_negatives"]))
        config = {"validation_criterion": "normalized density NLL omitting common Gaussian term",
                  "density_normalizer_size": normalizer_size,
                  "normalizer_bank_fingerprint": loop._fingerprint(q),
                  "validation_negative_bank_fingerprint": loop._fingerprint(negative_bank)}

        def validate(net):
            positive_sum = positive_bce = negative_bce = 0.
            for start in range(0, len(validation), 1024):
                g = net(validation[start:start+1024].to(target)).squeeze(-1).clamp(-R,R)
                positive_sum += float(g.double().sum())
                positive_bce += float(F.softplus(-g).double().sum())
            for start in range(0, len(negative_bank), 1024):
                g = net(negative_bank[start:start+1024].to(target)).squeeze(-1).clamp(-R,R)
                negative_bce += float(F.softplus(g).double().sum())
            sum_w = sum_w2 = max_w = 0.
            clipped = 0
            for start in range(0, len(q), 8192):
                raw = net(q[start:start+8192].to(target)).squeeze(-1)
                g = raw.clamp(-R,R).double()
                weights = torch.exp(g)
                sum_w += float(weights.sum())
                sum_w2 += float(weights.square().sum())
                max_w = max(max_w, float(weights.max()))
                clipped += int((raw.abs() >= R).sum())
            logz = math.log(sum_w/len(q))
            density_loss = logz-positive_sum/len(validation)
            gaussian_nll = float((.5*validation.double().square().sum(1)
                                  + .5*validation.shape[1]*math.log(2*math.pi)).mean())
            return {"validation_loss": density_loss,
                    "validation_density_nll_omit_common_log_phi": density_loss,
                    "validation_density_nll_full": density_loss+gaussian_nll,
                    "validation_log_normalizer": logz,
                    "validation_positive_mean_clipped_logit": positive_sum/len(validation),
                    "validation_bce": (positive_bce+negative_bce)/(3*len(validation)),
                    "validation_normalizer_ess": sum_w*sum_w/sum_w2,
                    "validation_normalizer_max_normalized_weight": max_w/sum_w,
                    "validation_normalizer_clipped_fraction": clipped/len(q),
                    **architecture.weight_diagnostics(net)}
    else:
        clean = validation.repeat_interleave(VALIDATION_DRAWS, dim=0)
        times = torch.exp(torch.rand((len(clean),1), generator=loop._generator(seeds["validation_times"]))
                          * (math.log(T)-math.log(T_MIN))+math.log(T_MIN))
        epsilon = torch.randn(clean.shape, generator=loop._generator(seeds["validation_noise"]))
        sigma = torch.sqrt(-torch.expm1(-2*times))
        noisy = torch.exp(-times)*clean+sigma*epsilon
        config = {"validation_criterion": "unpenalized DSM mean squared error per coordinate",
                  "validation_draws_per_point": VALIDATION_DRAWS,
                  "validation_bank_fingerprints": {"time":loop._fingerprint(times),
                      "noise":loop._fingerprint(epsilon),"noisy":loop._fingerprint(noisy)}}
        times,epsilon,sigma,noisy=(x.to(target) for x in (times,epsilon,sigma,noisy))

        def validate(net):
            total=0.
            for start in range(0,len(noisy),1024):
                sl=slice(start,start+1024)
                errors=((sigma[sl]*net(noisy[sl],times[sl])+epsilon[sl])**2).mean(1)
                total+=float(errors.double().sum())
            value=total/len(noisy)
            return {"validation_loss":value,"validation_loss_per_coordinate":value,
                    **architecture.weight_diagnostics(net)}
    return validate,config


def train_cv(train, validation, *, method, l1_lambda, seed, output_dir,
             max_updates=CV_HORIZON, validation_checks=100, batch_size=256,
             normalizer_size=NORMALIZER_SIZE, retain_validation_states=None,
             device="cpu", verbose=False):
    """Fit one fold/candidate through the full common positive-epoch horizon."""
    train,target=_inputs(train,device)
    validation,_=_inputs(validation,device)
    if validation.shape[1]!=train.shape[1]:
        raise ValueError("Training and validation dimensions differ")
    loop._positive_int("normalizer_size",normalizer_size)
    model,spec,config,seeds,batch_loss=_setup(train,method,l1_lambda,seed,target)
    base=loop._config(method,train,validation,seed,target,max_updates,batch_size,
                      validation_checks,seeds)
    base.update(config)
    config=base
    validate,validation_config=_validator(validation,method,seeds,target,normalizer_size)
    config.update(validation_config)
    if retain_validation_states is None:
        retain_validation_states=method=="cindes"
    if not isinstance(retain_validation_states,bool):
        raise ValueError("retain_validation_states must be boolean")
    config.update({"retain_validation_states":retain_validation_states,
                   "n_val":len(validation),"validation_penalty_included":False,
                   "positive_batches_per_epoch":math.ceil(len(train)/batch_size),
                   "max_positive_epochs":max_updates/math.ceil(len(train)/batch_size)})
    return loop._run(model,batch_loss,validate,len(train),
                     loop._generator(seeds["positive_permutations"]),config,
                     output_dir,target,verbose)


def refit_fixed(train, *, method, l1_lambda, seed, output_dir, stop_updates,
                scheduler_updates=FULL_HORIZON,batch_size=256,device="cpu",verbose=False):
    """Refit all observed data at a frozen lambda and cosine-schedule prefix."""
    train,target=_inputs(train,device)
    loop._positive_int("scheduler_updates",scheduler_updates)
    loop._positive_int("batch_size",batch_size)
    if isinstance(stop_updates,bool) or not isinstance(stop_updates,int) or not 0<=stop_updates<=scheduler_updates:
        raise ValueError("stop_updates must be an integer between zero and scheduler_updates")
    model,spec,config,seeds,batch_loss=_setup(train,method,l1_lambda,seed,target)
    config.update({"protocol":"positive_balanced_fixed_duration_refit_v1",
                   "n_validation":0,"batch_size":batch_size,"stop_updates":stop_updates,
                   "scheduler_updates":scheduler_updates,
                   "scheduler":"CosineAnnealingLR T_max remains full horizon",
                   "positive_batches_per_epoch":math.ceil(len(train)/batch_size),
                   "selection_rule":"lambda and common stopping fraction frozen from this dataset's three-fold CV; no refit validation"})
    directory=Path(output_dir)
    with loop._locked(directory):
        metadata_path=directory/"metadata.json"
        checkpoint_path=directory/"model.pt"
        history_path=directory/"training_history.jsonl"
        if metadata_path.exists():
            old=json.loads(metadata_path.read_text())
            if old.get("config")!=config or old.get("status")!="complete":
                raise ValueError("Existing refit is incomplete or differs; use a new directory")
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=old["checkpoint_sha256"]:
                raise ValueError("Refit checkpoint changed")
            payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
            if payload.get("config")!=config or payload.get("update")!=stop_updates:
                raise ValueError("Refit checkpoint/config mismatch")
            model.load_state_dict(payload["state_dict"],strict=True)
            return model.eval(),dict(old,reused=True)
        if checkpoint_path.exists() or history_path.exists():
            raise ValueError("Unrecognized refit artifacts; use a new directory")
        initial_hash=loop._state_hash(model)
        batch_hash=hashlib.sha256()
        permutation_rng=loop._generator(seeds["positive_permutations"])
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=scheduler_updates)
        checks={math.ceil(k*stop_updates/20) for k in range(1,21)}
        updates=epoch_index=positives_seen=negatives_seen=0
        loss_total=0.;loss_count=0
        batches_per_epoch=math.ceil(len(train)/batch_size)
        started=time.perf_counter();epoch_fraction=0.

        def metadata(status):
            return {"status":status,"config":config,"actual_updates":updates,
                    "actual_epochs":epoch_fraction,"positive_occurrences":positives_seen,
                    "negative_occurrences":negatives_seen,"initial_state_sha256":initial_hash,
                    "batch_schedule_sha256":batch_hash.hexdigest(),
                    "wall_seconds":time.perf_counter()-started,
                    "learning_rate_next_update":float(optimizer.param_groups[0]["lr"]),
                    "timing_scope":"all refit operations; no validation",
                    "checkpoint":"model.pt","reused":False}

        loop._json(metadata_path,metadata("in_progress"))
        model.train()
        while updates<stop_updates:
            epoch_index+=1
            order=torch.randperm(len(train),generator=permutation_rng,device="cpu")
            for batch_index,start in enumerate(range(0,len(train),batch_size)):
                indices=order[start:start+batch_size]
                batch_hash.update(indices.numpy().tobytes())
                loss,positive_count,negative_count=batch_loss(model,indices)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite refit loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward();optimizer.step();scheduler.step()
                updates+=1;positives_seen+=positive_count;negatives_seen+=negative_count
                loss_total+=float(loss.detach())*len(indices);loss_count+=len(indices)
                epoch_fraction=epoch_index-1+(batch_index+1)/batches_per_epoch
                if updates in checks:
                    loop._sync(target)
                    row={"update":updates,"epoch":epoch_fraction,
                         "training_loss_since_previous_check":loss_total/loss_count,
                         "learning_rate_next_update":float(optimizer.param_groups[0]["lr"]),
                         "positive_occurrences":positives_seen,"negative_occurrences":negatives_seen}
                    with history_path.open("a",encoding="utf-8") as handle:
                        handle.write(json.dumps(row,allow_nan=False)+"\n")
                    loss_total=0.;loss_count=0
                    loop._json(metadata_path,metadata("in_progress"))
                    if verbose:print(json.dumps({"method":method,**row}),flush=True)
                if updates==stop_updates:break
        loop._sync(target);model.eval()
        payload={"format_version":2,"method":method,"config":config,"model_config":spec,
                 "state_dict":{name:t.detach().cpu().clone() for name,t in model.state_dict().items()},
                 "update":updates,"epoch":epoch_fraction,"initial_state_sha256":initial_hash,
                 "batch_schedule_sha256":batch_hash.hexdigest(),"selection":"fixed before refit"}
        loop._atomic(checkpoint_path,lambda handle:torch.save(payload,handle),binary=True)
        result=metadata("complete")
        result.update({"checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       "final_state_sha256":loop._state_hash(model),
                       "weight_diagnostics":architecture.weight_diagnostics(model)})
        loop._json(metadata_path,result)
        return model,result
