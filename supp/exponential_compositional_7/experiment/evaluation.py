"""Evaluate on the original positive data scale; never used for tuning."""
from __future__ import annotations
import numpy as np
from scipy.stats import expon, norm, wasserstein_distance
from dgp import ExponentialCompositionalDGP

METRICS = ('tv1','tv2','llgap','h_tv','h_w1')

def histograms(y, h, dgp):
    edges = expon.ppf(np.linspace(0,1,21))
    hedge = norm.ppf(np.linspace(0,1,21))
    bins = np.searchsorted(edges[1:-1],y,side='right')
    p1 = np.array([np.bincount(bins[:,j],minlength=20) for j in range(dgp.d)],float)/len(y)
    p2 = np.array([np.bincount(20*bins[:,j]+bins[:,k],minlength=400).reshape(20,20)
                   for j,k in dgp.edges],float)/len(y)
    ph = np.bincount(np.searchsorted(hedge[1:-1],h,side='right'),minlength=20)/len(y)
    return p1,p2,ph

class Evaluator:
    def __init__(self, reference):
        self.dgp = ExponentialCompositionalDGP()
        self.reference = np.asarray(reference,dtype=np.float64)
        self.h = self.dgp.H(self.reference)
        self.logp = self.dgp.log_p_unnorm(self.reference)
        self.p1,self.p2,self.ph = histograms(self.reference,self.h,self.dgp)

    def evaluate(self, y):
        y = np.asarray(y,dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.dgp.d or not np.isfinite(y).all() or not (y > 0).all():
            raise ValueError('Expected finite strictly positive generated observations')
        h = self.dgp.H(y)
        lp = self.dgp.log_p_unnorm(y)
        p1,p2,ph = histograms(y,h,self.dgp)
        return {
            'n_gen':len(y), 'tv1':float(.5*np.abs(p1-self.p1).sum(1).mean()),
            'tv2':float(.5*np.abs(p2-self.p2).sum((1,2)).mean()),
            'llgap':float(self.logp.mean()-lp.mean()),
            'h_tv':float(.5*np.abs(ph-self.ph).sum()),
            'h_w1':float(wasserstein_distance(h,self.h)),
            'mean_H':float(h.mean()), 'sd_H':float(h.std(ddof=1)),
            'mean_log_p_unnorm':float(lp.mean()),
            'llgap_mc_se':float(np.sqrt(lp.var(ddof=1)/len(lp)+self.logp.var(ddof=1)/len(self.logp))),
            'tail3_coordinate':float(np.mean(y>3)),
            'tail5_coordinate':float(np.mean(y>5)),
            'tail8_coordinate':float(np.mean(y>8)),
            'near_zero_coordinate':float(np.mean(y<0.01)),
            'minimum':float(y.min()),'maximum':float(y.max()),
            'mean_coordinate':float(y.mean()),
            'mean_coordinate_second_moment':float(np.square(y).mean()),
            'nonpositive_fraction':float(np.mean(y<=0)),
        }
