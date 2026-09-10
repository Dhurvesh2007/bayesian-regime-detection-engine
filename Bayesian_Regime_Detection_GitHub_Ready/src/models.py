
from pathlib import Path
import numpy as np, pandas as pd, json, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from .gaussian_hmm import GaussianHMM

REGIMES=["Risk-On","Late-Cycle","Transitional","Risk-Off","Post-Shock"]
FEATURES=[
"Nifty_LogRet","Nifty_Ret_5D","Nifty_Ret_21D","Nifty_Ret_63D","Nifty_Ret_126D","Nifty_Ret_252D",
"Nifty_Vol_10D","Nifty_Vol_21D","Nifty_Vol_63D","Nifty_Mom_20D","Nifty_Mom_63D","Nifty_Mom_126D","Nifty_Mom_252D",
"Price_to_MA200","MA50_to_MA200","Drawdown","Midcap_vs_Nifty","Smallcap_vs_Nifty","AD_Ratio","AD_Breadth",
"Breadth_50D_Change","Breadth_200D_Change","FII_Net_Cr_Z63","DII_Net_Cr_Z63","FII_DII_Balance_Z63",
"VIX_Z252","VIX_Change_21D","USDINR_Ret_21D","Gilt_Change_21D","Brent_Ret_21D","DXY_Ret_21D","Gold_Ret_21D",
"US10Y_Change_21D","CPI_Change_12M","IIP_Change_12M","WPI_Change_12M","PMI_3M_Avg","GDP_Growth"]

def fit_scaler(df):
    scaler=StandardScaler()
    X=scaler.fit_transform(df[FEATURES])
    return X,scaler

def bic(hmm,n_obs,n_features):
    # Gaussian diagonal emissions + transition matrix + initial distribution.
    p=(hmm.n_states-1)+hmm.n_states*(hmm.n_states-1)+hmm.n_states*n_features*2
    return -2*hmm.loglik_+p*np.log(n_obs)

def state_signature(df,probs):
    tmp=df[["Nifty_LogRet","Nifty_Vol_21D","Price_to_MA200","AD_Breadth","India_VIX"]].copy()
    tmp["FII_Net_Cr"]=df["FII_Net_Cr"] if "FII_Net_Cr" in df else np.nan
    tmp["Drawdown"]=df["Drawdown"]
    tmp["state"]=probs.argmax(1)
    sig=tmp.groupby("state").mean(numeric_only=True)
    return sig

def label_states(df,probs):
    # Regime labels are assigned post-hoc from observable signatures.
    tmp=df.copy(); tmp["state"]=probs.argmax(1)
    rows=[]
    for s,g in tmp.groupby("state"):
        rows.append([s,g.Nifty_LogRet.mean(),g.Nifty_Vol_21D.mean(),g.Price_to_MA200.mean(),
                     g.AD_Breadth.mean(),g.India_VIX.mean(),g.Drawdown.mean(),g.Midcap_vs_Nifty.mean(),
                     g.Smallcap_vs_Nifty.mean(),g.FII_Net_Cr.mean()])
    sig=pd.DataFrame(rows,columns=["state","ret","vol","trend","breadth","vix","dd","midrel","smallrel","fii"]).set_index("state")
    # Risk-Off: most negative return / worst breadth / high vol
    riskoff=(sig["ret"].rank()+sig["breadth"].rank()+(-sig["vix"]).rank()+sig["dd"].rank()).idxmin()
    # Risk-On: strongest trend/return with lower vol
    riskon=(sig["ret"].rank()+sig["trend"].rank()+sig["breadth"].rank()+(-sig["vol"]).rank()).idxmax()
    remaining=[s for s in sig.index if s not in [riskoff,riskon]]
    # Post-Shock: among remaining, strong return + high volatility + relatively bad prior trend/drawdown
    if remaining:
        post=max(remaining,key=lambda s: sig.loc[s,"ret"]+0.8*sig.loc[s,"vol"]-0.6*sig.loc[s,"trend"]+0.5*sig.loc[s,"dd"])
    else: post=None
    remaining=[s for s in remaining if s!=post]
    if remaining:
        # Late-cycle: positive trend but weaker breadth / higher vol; Transitional gets residual.
        late=max(remaining,key=lambda s: sig.loc[s,"trend"]-0.7*sig.loc[s,"breadth"]+0.4*sig.loc[s,"vol"])
    else: late=None
    remaining=[s for s in remaining if s!=late]
    trans=remaining[0] if remaining else None
    mapping={riskon:"Risk-On",riskoff:"Risk-Off",post:"Post-Shock",late:"Late-Cycle",trans:"Transitional"}
    return mapping,sig

def fit_full(df):
    X,scaler=fit_scaler(df)
    results={}
    for k in [3,5,7]:
        h=GaussianHMM(k,n_iter=12,random_state=100+k).fit(X)
        results[k]={"hmm":h,"bic":bic(h,len(df),X.shape[1])}
    h=results[5]["hmm"]
    p=h.predict_proba(X)
    mapping,sig=label_states(df,p)
    canonical=np.zeros((len(df),5))
    for s,lab in mapping.items():
        canonical[:,REGIMES.index(lab)]=p[:,s]
    out=df[["Date"]].copy()
    for j,r in enumerate(REGIMES): out[f"{r}_Probability"]=canonical[:,j]
    out["Dominant_Regime"]=[REGIMES[i] for i in canonical.argmax(1)]
    out["Confidence"]=canonical.max(1)
    out["Entropy"]=-(canonical*np.log(canonical+1e-12)).sum(1)
    # A simple ensemble uncertainty proxy from posterior entropy.
    out["Uncertainty"]=out["Entropy"]/np.log(5)
    # 90% probability-mass set
    sets=[]
    for row in canonical:
        order=np.argsort(row)[::-1]; total=0; names=[]
        for j in order:
            names.append(REGIMES[j]); total+=row[j]
            if total>=.90: break
        sets.append("{"+", ".join(names)+"}")
    out["Conformal_90_Set"]=sets
    meta={"state_mapping":{str(k):v for k,v in mapping.items()},"state_signature":sig.to_dict(),
          "bic":{"3":results[3]["bic"],"5":results[5]["bic"],"7":results[7]["bic"]},
          "features":FEATURES}
    return out,meta,scaler,h,canonical

def walk_forward(df,train_frac=.70):
    cut=int(len(df)*train_frac)
    train=df.iloc[:cut].copy(); test=df.iloc[cut:].copy()
    Xtr,scaler=fit_scaler(train)
    Xt=scaler.transform(test[FEATURES])
    h=GaussianHMM(5,n_iter=15,random_state=77).fit(Xtr)
    ptr=h.predict_proba(Xtr); mapping,_=label_states(train,ptr)
    pt=h.predict_proba(Xt)
    E=np.zeros((len(test),5))
    for s,lab in mapping.items(): E[:,REGIMES.index(lab)]=pt[:,s]
    result=test[["Date"]].copy()
    for j,r in enumerate(REGIMES): result[f"{r}_Probability"]=E[:,j]
    result["Dominant_Regime"]=[REGIMES[i] for i in E.argmax(1)]
    return train,test,result,mapping,scaler,h

def regime_forward_stats(df,result,horizon=21):
    base=df.copy().set_index("Date")
    r21=np.log(base.Nifty_Close).shift(-horizon)-np.log(base.Nifty_Close)
    z=result.copy().set_index("Date")
    joined=z.join(r21.rename("Future_LogReturn"),how="left")
    stats=joined.groupby("Dominant_Regime")["Future_LogReturn"].agg(["count","mean","median","std"])
    return stats.reset_index()

if __name__=="__main__":
    df=pd.read_csv("data/processed/model_features.csv",parse_dates=["Date"])
    # Add required raw columns for labeling
    raw=pd.read_csv("data/processed/master_daily_raw.csv",parse_dates=["Date"])
    df=df.merge(raw[["Date","Nifty_Close","India_VIX","FII_Net_Cr"]],on="Date",how="left")
    out,meta,scaler,h,E=fit_full(df)
    out.to_csv("outputs/full_regime_predictions.csv",index=False)
    joblib.dump({"scaler":scaler,"hmm":h,"meta":meta},"models/regime_model.joblib")
    Path("outputs/model_metadata.json").write_text(json.dumps(meta,indent=2,default=float))
    tr,te,wf,mapping,scaler2,h2=walk_forward(df)
    wf.to_csv("outputs/walk_forward_test_predictions.csv",index=False)
    stats=regime_forward_stats(df,wf,21)
    stats.to_csv("outputs/regime_21d_forward_stats.csv",index=False)
    print(out.tail(1).T)
    print("\nBIC:",meta["bic"])
    print("\nWalk-forward regime stats:")
    print(stats.to_string(index=False))
