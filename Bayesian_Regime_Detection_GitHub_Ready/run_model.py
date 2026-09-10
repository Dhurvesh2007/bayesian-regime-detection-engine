
import pandas as pd, json, joblib
from src.models import fit_full, walk_forward, regime_forward_stats

df=pd.read_csv("data/processed/model_features.csv",parse_dates=["Date"])
raw=pd.read_csv("data/processed/master_daily_raw.csv",parse_dates=["Date"])
df=df.merge(raw[["Date","Nifty_Close","India_VIX","FII_Net_Cr"]],on="Date",how="left")
out,meta,scaler,h,E=fit_full(df)
out.to_csv("outputs/full_regime_predictions.csv",index=False)
joblib.dump({"scaler":scaler,"hmm":h,"meta":meta},"models/regime_model.joblib")
open("outputs/model_metadata.json","w").write(json.dumps(meta,indent=2,default=float))
tr,te,wf,mapping,scaler2,h2=walk_forward(df)
wf.to_csv("outputs/walk_forward_test_predictions.csv",index=False)
regime_forward_stats(df,wf,21).to_csv("outputs/regime_21d_forward_stats.csv",index=False)
print("Model complete.")
print(out.tail(1).T.to_string())
print("BIC:",meta["bic"])
