import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import re
import lightgbm as lgb
import matplotlib.colors as colors
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from scipy.optimize import curve_fit
from routines import parse_ml_events, boostinv

#def rf_observable_with_uncertainty(model, X):
#    return obs_mean, obs_unc


# parsing input argument (reduced or full dataset)
parser = argparse.ArgumentParser(description="Which events do you want?")
parser.add_argument('--data',    '-d', type=str, choices=['reduced', 'full'])
parser.add_argument('--training','-t', type=str, choices=['long', 'lows', 'lo'])
parser.add_argument('--is_lo','-i', type=str, choices=['lo', 'nlo'])
args = parser.parse_args()

string_model = ""
if str(args.training) == 'long':
#    string_model = "trained_RFR_nlo_train_events_801748_basis_ct.joblib"
    string_model = "trained_RFR_nlo_train_events_797759_basis_ct.joblib"
# elif str(args.training) == 'short':
#     string_model = "trained_RFR_nlo_train_events_80158_basis_ct.joblib"
elif str(args.training) == 'lows':
    string_model = "trained_RFR_lo_train_events_793238_basis_ct.joblib"
#   string_model = "trained_RFR_lo_train_events_63896_basis_ct.joblib"
elif str(args.training) == 'lo':
     string_model = "trained_RFR_lo_train_events_63178_basis_ct.joblib"


# initialisation and choice of LHE-ML dataset
if str(args.is_lo) == 'lo':
    print(' You are parsing LOPS events')
else:
    print(' You are parsing NLOPS events')

#sigma_uu = np.array([0.15183819E-01,0.71627436E-05])
sigma_ll = np.array([0.8918E-03, 0.0003E-03])

data_dir = Path("../../events/ML_FILES/UU_NLO")
if str(args.is_lo) == 'lo':
    data_dir = Path("../../events/ML_FILES/UU_LOwS")
nr_lhef = 51 
if str(args.data) == 'reduced':
    nr_lhef = 6
N_lhe = 50000
N_tot = N_lhe * (nr_lhef-1)

    
# access ml unpolarised events and various weights
used_weights = {"UU"} # not used
all_events = []
start = time.time()
for i in range(1, min(10,nr_lhef)):
    filepath = data_dir/f"output_shower_events-000{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
for i in range(10, max(10,nr_lhef)):
    filepath = data_dir/f"output_shower_events-00{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
end1 = time.time()
print("Elapsed (1st step):", end1 - start, "seconds")

# create DataFrame
df = pd.DataFrame(all_events)

# compute observables
df["kin_pt_1"] = (df["x0"]**2+df["x1"]**2)**0.5
df["kin_pt_2"] = (df["x4"]**2+df["x5"]**2)**0.5
df["kin_pt_3"] = (df["x8"]**2+df["x9"]**2)**0.5
df["kin_pt_4"] = (df["x12"]**2+df["x13"]**2)**0.5
df["kin_phi_1"] = np.arctan2(df["x1"],df["x0"])
df["kin_phi_2"] = np.arctan2(df["x5"],df["x4"])
df["kin_phi_3"] = np.arctan2(df["x8"],df["x8"])
df["kin_phi_4"] = np.arctan2(df["x13"],df["x12"])
df["kin_y_1"] = 0.5*np.log((df["x3"]+df["x2"]) / (df["x3"]-df["x2"]))
df["kin_y_2"] = 0.5*np.log((df["x7"]+df["x6"]) / (df["x7"]-df["x6"]))
df["kin_y_3"] = 0.5*np.log((df["x11"]+df["x10"]) / (df["x11"]-df["x10"]))
df["kin_y_4"] = 0.5*np.log((df["x15"]+df["x14"]) / (df["x15"]-df["x14"]))
p1 = df[["x0","x1","x2","x3"]].to_numpy()
p2 = df[["x4","x5","x6","x7"]].to_numpy()
p3 = df[["x8","x9","x10","x11"]].to_numpy()
p4 = df[["x12","x13","x14","x15"]].to_numpy()
p12 = p1 + p2
p34 = p3 + p4
ptot = p1 + p2 + p3 + p4
p12_cm = np.zeros_like(p12)
p1_cm = np.zeros_like(p1)
p34_cm = np.zeros_like(p34)
p3_cm = np.zeros_like(p3)
p1_in_p12_rest = np.zeros_like(p1)
p3_in_p34_rest = np.zeros_like(p3)
cos_theta = np.zeros(len(df))
cos_thetab= np.zeros(len(df))
ptv1 = np.zeros(len(df))
ptv2 = np.zeros(len(df))
ptvv = np.zeros(len(df))
yv1 = np.zeros(len(df))
yv2 = np.zeros(len(df))
phiv1 = np.zeros(len(df))
phiv2 = np.zeros(len(df))
for i in range(len(df)):
    p12_cm[i] = boostinv(p12[i], ptot[i])
    p1_cm[i]  = boostinv(p1[i], ptot[i])
    p34_cm[i] = boostinv(p34[i], ptot[i])
    p3_cm[i]  = boostinv(p3[i], ptot[i])
    p1_in_p12_rest[i] = boostinv(p1_cm[i], p12_cm[i])
    p3_in_p34_rest[i] = boostinv(p3_cm[i], p34_cm[i])
    p12_dir_cm = p12_cm[i, :3]
    p34_dir_cm = p34_cm[i, :3]
    p1_dir_rest = p1_in_p12_rest[i, :3]
    p3_dir_rest = p3_in_p34_rest[i, :3]
    cos_theta[i]  = np.dot(p12_dir_cm, p1_dir_rest) / (np.linalg.norm(p12_dir_cm) * np.linalg.norm(p1_dir_rest))
    cos_thetab[i] = np.dot(p34_dir_cm, p3_dir_rest) / (np.linalg.norm(p34_dir_cm) * np.linalg.norm(p3_dir_rest))
    ptv1[i] = (p12[i,0]**2+p12[i,1]**2)**0.5
    ptv2[i] = (p34[i,0]**2+p34[i,1]**2)**0.5
    ptvv[i] = ((p12[i,0]+p34[i,0])**2+(p12[i,1]+p34[i,1])**2)**0.5
    yv1[i]  = 0.5*np.log(( p12[i,3] + p12[i,2] ) / ( p12[i,3] - p12[i,2] ))
    yv2[i]  = 0.5*np.log(( p34[i,3] + p34[i,2] ) / ( p34[i,3] - p34[i,2] ))
    phiv1[i] = np.arctan2(p12[i,1],p12[i,0])
    phiv2[i] = np.arctan2(p34[i,1],p34[i,0])

df["cos_theta_p1_p12"] = cos_theta
df["cos_theta_p3_p34"] = cos_thetab
df["ptZ1"] = ptv1
df["ptZ2"] = ptv2
df["yZ1"] = yv1
df["yZ2"] = yv2
df["phiZ1"] = phiv1
df["phiZ2"] = phiv2
df["pt4l"] = ptvv

print('size of the whole dataset = ', len(df))
print(df.tail(3))

end2 = time.time()
print("Elapsed (2nd step):", end2 - end1, "seconds. Now load trained model and test...")

# loading and testing model
bundle = joblib.load(string_model)
trained_model  = bundle["model"]
input_features = bundle["features"] # ensure same order as for training
X_test = df[input_features]
y_pred = np.clip(trained_model.predict(X_test),0,1)

# model uncertainty from tree-to-tree fluctuations
preds = np.array([tree.predict(X_test.to_numpy()) for tree in trained_model.estimators_])
obs_mean = preds.mean(axis=1).mean()
obs_unc  = preds.mean(axis=1).std(ddof=1)


z_uu = (df.loc[X_test.index, "UU"]).to_numpy()*1e+03/N_tot
w_pred = np.array([z_uu[i] * y_pred[i] for i in range(0,len(z_uu))])
#w_err  = np.array([z_uu[i]**2 * var_events[i] for i in range(0,len(z_uu))])
sigLLpred = np.array([w_pred.sum(), 0.0]) # 1e+03*w_err.sum()/N_tot])

sigLL_trees = (preds * z_uu).sum(axis=1)
sigLL_mean = sigLL_trees.mean()
sigLL_err = sigLL_trees.std(ddof=1)/np.sqrt(len(sigLL_trees))

print(' total number of test events ............................. ', len(y_pred))
print(' expected LL xsec (polarised simulation) ................. %.4f ' % (1e+3*sigma_ll[0]), ' +- %.4f (MC) fb' % (1e+3*sigma_ll[1]) )
#print(' estimated LL xsec (without uncertainty) ................. %.4f ' % (sigLLpred[0]))
print(' estimated LL xsec (with uncertainty) .................... %.4f ' % (sigLL_mean), '(%.0f )' % (1e+04*sigLL_err))
#print(' estimated LL xsec (pred-rLL reweighting, test) .......... %.4f ' % (1e+3*sigma_uu[0]*(y_pred.sum()/len(y_pred))))


endall = time.time()
print("\n Full elapsed time:", endall - start, "seconds. Done.\n")













