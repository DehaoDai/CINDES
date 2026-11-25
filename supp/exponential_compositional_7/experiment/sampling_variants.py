"""Common initialization and nested-grid variants of the frozen reverse sampler.

The score functions and models are unchanged. No target truth enters this file.
Coarse innovations reproduce the original sampler. Fine innovations use an
independent orthogonal bridge, and odd fine steps use separate particles.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from reference import LogExponentialReference, sample_reference
from sampling import ScoreDiagnostics, cindes_score, make_timesteps


def bridge_coefficients(h1, h2):
    total = math.sqrt(-math.expm1(-2*(h1+h2)))
    a = math.exp(-h2)*math.sqrt(-math.expm1(-2*h1))/total
    b = math.sqrt(-math.expm1(-2*h2))/total
    return a, b


def split_innovation(coarse, bridge, h1, h2):
    a, b = bridge_coefficients(h1, h2)
    return a*coarse+b*bridge, b*coarse-a*bridge


@torch.no_grad()
def sample_variant(model, method, n, d=7, *, initialization='reference_T',
                   M=200, seed=0, particle_seed=0, reference_seed=0,
                   bridge_seed=0, fine_particle_seed=0, K=1000, T=3.0,
                   delta=.001, batch=64, device='cpu', quadrature=None):
    if method not in ('cindes','dsm','reference','gaussian_null'):
        raise ValueError('Unknown method')
    if initialization not in ('normal','reference_T') or M not in (200,400):
        raise ValueError('This frozen diagnostic supports only its planned variants')
    if min(n,d,batch) < 1:
        raise ValueError('Positive counts required')
    device = torch.device(device)
    if model is not None:
        model = model.to(device).eval()
    reverse = torch.Generator(device=device).manual_seed(int(seed)+999)
    particles = torch.Generator(device=device).manual_seed(int(particle_seed))
    fine_particles = torch.Generator(device=device).manual_seed(int(fine_particle_seed))
    bridges = torch.Generator(device=device).manual_seed(int(bridge_seed))
    reference_rng = np.random.default_rng(int(reference_seed))
    posterior = LogExponentialReference(quadrature,device=device)
    diagnostic = ScoreDiagnostics() if method == 'cindes' else None
    coarse_times = T-make_timesteps(T,delta,200)
    fine_times = T-make_timesteps(T,delta,400)
    if not np.array_equal(coarse_times,fine_times[::2]):
        raise ArithmeticError('Fine time grid does not nest the original grid exactly')
    outputs=[]
    started=time.perf_counter()

    def advance(y,tau,tau_next,noise,generator):
        if method=='cindes':
            score=cindes_score(model,y,tau,K,generator,posterior,diagnostic)
        elif method=='dsm':
            score=model(y,torch.full((len(y),1),tau,dtype=y.dtype,device=device))
        elif method=='reference':
            score=posterior.score(y,tau)
        else:
            score=-y
        h=tau-tau_next
        decay=math.exp(-h)
        mean=decay*y+2*(-math.expm1(-h))*(score+y)
        result=mean+math.sqrt(-math.expm1(-2*h))*noise
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError(f'Nonfinite reverse sample at time {tau_next}')
        return result

    for start in range(0,n,batch):
        size=min(batch,n-start)
        gaussian=torch.randn((size,d),generator=reverse,device=device,dtype=torch.float32)
        if initialization=='normal':
            y=gaussian
        else:
            reference=torch.as_tensor(sample_reference(reference_rng,(size,d)),device=device)
            y=math.exp(-T)*reference+math.sqrt(-math.expm1(-2*T))*gaussian
        for j in range(200):
            coarse_noise=torch.randn(y.shape,generator=reverse,device=device,dtype=y.dtype)
            if M==200:
                y=advance(y,coarse_times[j],coarse_times[j+1],coarse_noise,particles)
            else:
                tau,mid,tau_next=fine_times[2*j:2*j+3]
                bridge=torch.randn(y.shape,generator=bridges,device=device,dtype=y.dtype)
                first,second=split_innovation(coarse_noise,bridge,tau-mid,mid-tau_next)
                y=advance(y,tau,mid,first,particles)
                y=advance(y,mid,tau_next,second,fine_particles)
        outputs.append(y.cpu())
    values=torch.cat(outputs).numpy()
    metadata={
        'method':method,'n':n,'d':d,'initialization':initialization,
        'T':T,'delta':delta,'M':M,'K':K if method=='cindes' else None,
        'generation_batch':batch,'generation_seconds':time.perf_counter()-started,
        'reverse_seed':int(seed)+999,'particle_seed':int(particle_seed),
        'reference_initialization_seed':int(reference_seed),
        'bridge_seed':int(bridge_seed),'fine_particle_seed':int(fine_particle_seed),
        'last_step_noise':True,'terminal_correction':False,'support_clipping':False,
        'quadrature':posterior.config.__dict__,
        'coarse_noise_coupled_to_original':True,
        'fine_noise_coupling':'Orthogonal OU bridge; independent added normal stream',
        'fine_particle_coupling':'Even steps share coarse uniforms; odd steps use independent stream',
        'maximum_absolute_log_coordinate':float(np.abs(values).max()),
    }
    if diagnostic is not None:
        metadata['score_diagnostics']=diagnostic.summary(K)
    return values,metadata
