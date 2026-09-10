
import numpy as np
from scipy.special import logsumexp

class GaussianHMM:
    """Small, dependency-light diagonal-covariance Gaussian HMM."""
    def __init__(self, n_states=5, n_iter=20, tol=1e-4, random_state=42):
        self.n_states=n_states; self.n_iter=n_iter; self.tol=tol
        self.random_state=np.random.default_rng(random_state)

    def _log_emission(self,X):
        d=X.shape[1]
        diff=X[:,None,:]-self.means_[None,:,:]
        return -0.5*(np.sum(diff**2/self.vars_[None,:,:],axis=2)+np.sum(np.log(2*np.pi*self.vars_),axis=1))

    def _forward(self, le):
        n=le.shape[0]; k=self.n_states
        a=np.full((n,k),-np.inf); c=np.zeros(n)
        a[0]=np.log(self.startprob_+1e-300)+le[0]
        c[0]=logsumexp(a[0]); a[0]-=c[0]
        for t in range(1,n):
            a[t]=le[t]+logsumexp(a[t-1][:,None]+np.log(self.transmat_+1e-300),axis=0)
            c[t]=logsumexp(a[t]); a[t]-=c[t]
        return a,c,c.sum()

    def _backward(self,le,c):
        n,k=le.shape
        b=np.zeros((n,k))
        for t in range(n-2,-1,-1):
            b[t]=logsumexp(np.log(self.transmat_+1e-300)+le[t+1][None,:]+b[t+1][None,:],axis=1)-c[t+1]
        return b

    def fit(self,X):
        X=np.asarray(X,float); n,d=X.shape; k=self.n_states
        idx=self.random_state.choice(n,size=k,replace=False)
        self.means_=X[idx].copy()
        global_var=np.var(X,axis=0)+1e-3
        self.vars_=np.tile(global_var,(k,1))
        self.startprob_=np.ones(k)/k
        self.transmat_=np.ones((k,k))*0.02
        np.fill_diagonal(self.transmat_,0.90)
        self.transmat_/=self.transmat_.sum(1,keepdims=True)
        prev=-np.inf
        for _ in range(self.n_iter):
            le=self._log_emission(X); a,c,ll=self._forward(le); b=self._backward(le,c)
            lg=a+b; lg-=logsumexp(lg,axis=1,keepdims=True); gamma=np.exp(lg)
            xi=np.zeros((n-1,k,k))
            logT=np.log(self.transmat_+1e-300)
            for t in range(n-1):
                q=a[t,:,None]+logT+le[t+1,None,:]+b[t+1,None,:]
                q-=logsumexp(q)
                xi[t]=np.exp(q)
            self.startprob_=gamma[0]+1e-3; self.startprob_/=self.startprob_.sum()
            self.transmat_=xi.sum(0)+1e-2; self.transmat_/=self.transmat_.sum(1,keepdims=True)
            w=gamma.sum(0)+1e-12
            self.means_=(gamma.T@X)/w[:,None]
            self.vars_=np.maximum((gamma.T@(X**2))/w[:,None]-self.means_**2,1e-5)
            if abs(ll-prev)<self.tol: break
            prev=ll
        self.loglik_=ll
        return self

    def score(self,X):
        return self._forward(self._log_emission(np.asarray(X,float)))[2]

    def predict_proba(self,X):
        le=self._log_emission(np.asarray(X,float)); a,c,ll=self._forward(le); b=self._backward(le,c)
        lg=a+b; lg-=logsumexp(lg,axis=1,keepdims=True)
        return np.exp(lg)

    def predict(self,X):
        return self.predict_proba(X).argmax(1)
