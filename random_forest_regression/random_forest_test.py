import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import re
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.base import clone


 # parsing routine for .ml files
def parse_ml_events(filepath, useful_weights):
    events = []
    with open(filepath) as f:
        in_event = False
        in_rwgt = False
        numbers = []
        weights = {}
        for line in f:
            line = line.strip()
            if line == "<event>": # start event
                in_event = True
                numbers = []
                weights = {}
            elif line == "</event>": # end event
                event = {}
                for i, val in enumerate(numbers):
                    event[f"x{i}"] = val
                event.update(weights)
                events.append(event)
                in_event = False
            elif line == "<rwgt>": # weight tag begin
                in_rwgt = True
            elif line == "</rwgt>": # weight tag end
                in_rwgt = False
            elif in_event and not in_rwgt: # event kinematics
                if line and not line.startswith("<"):
                    numbers.extend(float(x) for x in line.split())
            elif in_rwgt: # selected weights
                match = re.search(r"id='([^']+)'>\s*([0-9E+\-.]+)", line)
                if match:
                    wid, val = match.groups()
                    if wid in useful_weights:
                        weights[wid] = float(val)
    return events


def boostinv(qx, px):
    q      = np.array([qx[3],qx[0],qx[1],qx[2]])
    pboost = np.array([px[3],px[0],px[1],px[2]])
    qprime = np.array([0.0,0.0,0.0,0.0])
    rmboost=(max((pboost[0]**2-pboost[1]**2-pboost[2]**2-pboost[3]**2),0.0))**0.5
    aux=(q[0]*pboost[0]-q[1]*pboost[1]-q[2]*pboost[2]-q[3]*pboost[3])/rmboost
    aaux=(aux+q[0])/(pboost[0]+rmboost)
    qprime[3]=aux
    qprime[0]=q[1]-aaux*pboost[1]
    qprime[1]=q[2]-aaux*pboost[2]
    qprime[2]=q[3]-aaux*pboost[3]
    return qprime




# parsing input argument (LO / NLO QCD)
parser = argparse.ArgumentParser(description="Which events do you want?")
parser.add_argument('--order',  '-p', type=str, choices=['lo', 'nlo'])
parser.add_argument('--data',   '-d', type=str, choices=['reduced', 'full'])
parser.add_argument('--model',  '-m', type=str, choices=['all'])
args = parser.parse_args()

# initialisation and choice of LHE-ML dataset
sigma_uu = np.array([0.11245290E-01, 0.37648619E-05])
sigma_ll = np.array([0.6574E-03, 0.0002E-03])
data_dir = Path("../../events/ML_FILES/UU_LO")
nr_lhef = 26
if str(args.data) == 'reduced':
    nr_lhef = 3
#if str(data_dir).find("NLO") != -1:
if str(args.order) == 'nlo':
    print(' You are parsing NLO QCD LHE events')
    sigma_uu = np.array([0.15183819E-01,0.71627436E-05])
    sigma_ll = np.array([0.8918E-03, 0.0003E-03])
    data_dir = Path("../../events/ML_FILES/UU_NLO")
    nr_lhef = 51 # 51
    if str(args.data) == 'reduced':
        nr_lhef = 6
else:
    print(' You are parsing LO LHE events')
    
# access ml unpolarised events and various weights 
used_weights = {"UU", "LL", "LT", "TL", "TT"}
all_events = []
start = time.time()
for i in range(1, min(10,nr_lhef)):   
    filepath = data_dir/f"pwgevents-000{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
for i in range(10, max(10,nr_lhef)): 
    filepath = data_dir/f"pwgevents-00{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
end1 = time.time()
print("Elapsed (1st step):", end1 - start, "seconds")

# frame events with relevant information (kinematics, rLL, rLT, rTL, rTT)
df = pd.DataFrame(all_events)


# x0 x1 x2 x3 -> px py pz E
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

# compute the cos(theta*_e+/mu+) observables
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

df["rLL"] = df["LL"] / df["UU"]
#df["rLT"] = df["LT"] / df["UU"]
#df["rTL"] = df["TL"] / df["UU"]
#df["rTT"] = df["TT"] / df["UU"]
df = df.drop(columns=["UU", "LL", "LT", "TL", "TT"]) # keep  "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"
print('size of the whole dataset (train + test) = ', len(df))

# label events with basic hit-or-miss 
n_longit = 0
err_longit = 0.0e+00
df["label"] = 0
x = np.random.rand(len(df))
for i in range(0,len(df)):
    err_longit += ((df["rLL"].iloc[i])*(1.0-(df["rLL"].iloc[i])))
    if x[i] < df["rLL"].iloc[i]:
        df.at[i,"label"] = 1
        n_longit += 1
#df = df.drop(columns=["rLT", "rTL", "rTT"])
err_longit = err_longit**0.5/float(len(df))

# test print 
print(df.tail(3))


#X = df[["kin_pt_1", "kin_pt_2", "kin_pt_3", "kin_pt_4", "kin_phi_1", "kin_phi_2", "kin_phi_3", "kin_phi_4", "kin_y_1", "kin_y_2", "kin_y_3", "kin_y_4"]]  # non-redundant features
#X = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "cos_theta_p1_p12"]]  # over-redundant features
#X = df[["ptZ1", "ptZ2", "yZ1", "yZ2", "phiZ1", "phiZ2", "cos_theta_p1_p12", "cos_theta_p3_p34"]]  # few features
#X = df[["cos_theta_p1_p12", "cos_theta_p3_p34"]]  # very few features

X = df[["ptZ1", "ptZ2", "yZ1", "yZ2", "cos_theta_p1_p12", "cos_theta_p3_p34"]]  # few features (including decay angles)
y = df["rLL"]   # target

# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
end2 = time.time()
print("Elapsed (2nd step):", end2 - end1, "seconds. Now start training and testing steps...")


if args.model == 'all':
    
    #############################################################################
    #   # Random-Forest Regressor
    #############################################################################
    # define and train model
    model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=20, random_state=None, n_jobs=40) # parallelise over 40 workers (~12 trees per worker)
    model.fit(X_train, y_train)
    end3 = time.time()
    print("Training step:", end3 - end2, "seconds. Now start testing...")

    # # tree-level bootstrap for model uncertainty
    # preds = np.array([tree.predict(X_test.values) for tree in model.estimators_])
    # B = 100
    # T_boot = []
    # for b in range(B):
    #     idx = np.random.randint(0, preds.shape[0], preds.shape[0])
    #     yb = preds[idx].mean(axis=0)
    #     T_boot.append(yb.mean())
    # T_boot = np.array(T_boot)
    # T_hat = T_boot.mean()
    # sigma_T = np.std(T_boot, ddof=1)

    # testing
    #y_pred = model.predict(X_test) # predict, allowing for out-of-range values
    y_pred = np.clip(model.predict(X_test), 0, 1) # predict, avoiding out-of-range values
    mse = mean_squared_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0,1]
    
    # RFR
    print(' total number of train events ............................ ', len(y_train))
    print(' total number of test events ............................. ', len(y_test))
    print(' expected LL xsec (polarised simulation) ................. %.4f ' % (1e+3*sigma_ll[0]), ' +- %.4f (MC) fb' % (1e+3*sigma_ll[1]) )
    print(' estimated LL xsec (true-rLL reweighting, test) .......... %.4f ' % (1e+3*sigma_uu[0]*(y_test.sum()/len(y_test))), ' +- %.4f (MC) fb' % (1e+3*sigma_uu[1]*(y_test.sum()/len(y_test))) )
    print(' estimated LL xsec (true-rLL resampling, test+train) ..... %.4f ' % (1e+3*sigma_uu[0]*(float(n_longit)/float(len(df)))), ' +- %.4f (binomial) fb' % (1e+3*sigma_uu[0]*err_longit))

    
    print(' estimated LL xsec (pred-rLL reweighting, test) .......... %.4f ' % (1e+3*sigma_uu[0]*(y_pred.sum()/len(y_test))), ' +- %.4f (test residuals) fb' % (1e+3*sigma_uu[0]* (mse/float(len(y_test)))**0.5 ))

    print("\nmse, correlation:", mse, corr, " \n")
    
    end4 = time.time()
    print("Testing step:", end4 - end3, "seconds.")

    #############################################################################
    #   # Two-model approach with Light-GBM MSE-regressor for mean and variance 
    #############################################################################

    print(" \n Light-GBM Regressor + model for train residuals\n")
    reg = lgb.LGBMRegressor(objective="mse", alpha=0.5, n_estimators=500, max_depth=10, learning_rate=0.05, force_row_wise=True, verbose=-1)
    reg.fit(X_train, y_train)    
    y_pred_2 = np.clip(reg.predict(X_test),0,1)
    resid = y_train - reg.predict(X_train)
    var_model = lgb.LGBMRegressor(objective="mse", force_row_wise=True, verbose=-1)
    var_model.fit(X_train, resid**2)
    M_hat = y_pred_2.sum()/float(len(y_pred))
    sigma_M = np.sqrt(np.sum( np.clip(var_model.predict(X_test), 0, None)  ))/float(len(y_pred))
    
    end5 = time.time()

    # LGBMR
    print(' estimated LL xsec from new model (pred-rLL, test) ............ %.4f ' % (1e+3*sigma_uu[0]*M_hat), ' +- %.4f (train-residual regression) fb' % (1e+3*sigma_uu[0]*sigma_M))
    print(" other model testing and training:", end5 - end4, "seconds.")

    # now plotting stuff
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10))
    
    yep = df.loc[X_test.index, "kin_y_1"]
    ax1.set_title("Rapidity of the positron")
    ax1.hist(yep, weights=y_test, bins=40, histtype="step", linewidth=1, color="blue", label="true", density=True)
    ax1.hist(yep, weights=y_pred, bins=40, histtype="step", linewidth=1, color="red", label="pred. (RFR)", density=True)
    ax1.hist(yep, weights=y_pred_2, bins=40, histtype="step", linewidth=1, color="green", label="pred. (LGBM)", density=True)
    ax1.legend()
    ax1.set_xlabel("y$_{\\tt e^+}$")
    ax1.set_ylabel("Normalised distributions")

    #ax1.set_yscale("log")
    
    ax2.set_title("Label distribution")
    ax2.hist(y_test, range=(-0.1, 1.1), bins=40, histtype="step", linewidth=1, color="blue", label="true", density=True)
    ax2.hist(y_pred, range=(-0.1, 1.1), bins=40, histtype="step", linewidth=1, color="red", label="pred. (RFR)", density=True)
    ax2.hist(y_pred_2, range=(-0.1,1.1), bins=40, histtype="step", linewidth=1, color="green", label="pred. (LGBM)", density=True)
    ax2.legend()
    ax2.set_xlabel("r$_{\\tt LL}$")
    ax2.set_ylabel("Normalised distributions")
    #ax2.set_yscale("log")
    
    cth = df.loc[X_test.index, "cos_theta_p1_p12"]
    ax3.set_title("Decay angle of the positron")
    ax3.hist(cth, weights=y_test, bins=40, histtype="step", linewidth=1, color="blue", label="true", density=True)
    ax3.hist(cth, weights=y_pred, bins=40, histtype="step", linewidth=1, color="red", label="pred. (RFR)", density=True)
    ax3.hist(cth, weights=y_pred_2, bins=40, histtype="step", linewidth=1, color="green", label="pred. (LGBM)", density=True)
    ax3.legend()
    ax3.set_xlabel("cos$\\theta^*_{\\tt e^+}$")
    ax3.set_ylabel("Normalised distributions")
    
    plt.tight_layout()
    fig.savefig("test_random_forest_regressor_"+ str(args.order) +"_test_events_" + str(len(y_pred)) + ".pdf")
    
    end6 = time.time()
    print("Plotting step:", end6 - end5, "seconds.")


endall = time.time()
print("\n Full elapsed time:", endall - start, "seconds. Done.\n")
#plt.show()













