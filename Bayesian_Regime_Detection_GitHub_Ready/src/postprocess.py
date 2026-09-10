
import numpy as np
import pandas as pd

REGIMES=["Risk-On","Late-Cycle","Transitional","Risk-Off","Post-Shock"]

def temperature_regularize(prob, temperature=1.6):
    """Reduce over-confident probabilities without claiming calibration."""
    p=np.clip(prob,1e-12,1)
    z=np.log(p)/temperature
    z=np.exp(z-z.max(axis=1,keepdims=True))
    return z/z.sum(axis=1,keepdims=True)

def prediction_set(prob, coverage=0.90):
    sets=[]
    for row in prob:
        order=np.argsort(row)[::-1]
        total=0; chosen=[]
        for j in order:
            chosen.append(REGIMES[j]); total += row[j]
            if total>=coverage: break
        sets.append("{"+", ".join(chosen)+"}")
    return sets
