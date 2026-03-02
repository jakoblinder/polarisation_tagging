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


# parsing input argument (LO / NLO QCD)
parser = argparse.ArgumentParser(description="Which events do you want?")
parser.add_argument('--order',  '-p', type=str, choices=['lo', 'nlo'])
parser.add_argument('--data',   '-d', type=str, choices=['reduced', 'full'])
parser.add_argument('--model',  '-m', type=str, choices=['all'])
parser.add_argument('--features',  '-f', type=str, choices=['ep', 'ct'])
args = parser.parse_args()

# initialisation and choice of LHE-ML dataset
N_lhe = 100000
sigma_uu = np.array([0.11245290E-01, 0.37648619E-05])
sigma_ll = np.array([0.6574E-03, 0.0002E-03])
data_dir = Path("../../events/ML_FILES/UU_LO")
#data_dir = Path("../../events/ML_FILES/UU_LOwS")
t_app = str("(LO, fiducial)")
nr_lhef = 26
if str(args.data) == 'reduced':
    nr_lhef = 3
#if str(data_dir).find("NLO") != -1:
if str(args.order) == 'nlo':
    print(' You are parsing NLO QCD LHE events')
    N_lhe = 50000
    sigma_uu = np.array([0.15183819E-01,0.71627436E-05])
    sigma_ll = np.array([0.8918E-03, 0.0003E-03])
    data_dir = Path("../../events/ML_FILES/UU_NLO")
    nr_lhef = 51 # 51
    t_app = str("(NLO QCD, LHE, fiducial)")
    if str(args.data) == 'reduced':
        nr_lhef = 6
else:
    print(' You are parsing LO LHE events')

# this is the number to divide sum of event weights to get correct normalisation
N_tot = N_lhe * (nr_lhef-1)

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

df["rLL"] = df["LL"] / df["UU"]
#df["rLT"] = df["LT"] / df["UU"]
#df["rTL"] = df["TL"] / df["UU"]
#df["rTT"] = df["TT"] / df["UU"]
df = df.drop(columns=["LT", "TL", "TT"]) # keep  "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"
print('size of the whole dataset (train + test) = ', len(df))

# label events with basic hit-or-miss
####   n_longit = 0
####   err_longit = 0.0e+00
####   df["label"] = 0
####   x = np.random.rand(len(df))
####   for i in range(0,len(df)):
####       err_longit += ((df["rLL"].iloc[i])*(1.0-(df["rLL"].iloc[i])))
####       if x[i] < df["rLL"].iloc[i]:
####           df.at[i,"label"] = 1
####           n_longit += 1
####   #df = df.drop(columns=["rLT", "rTL", "rTT"])
####   err_longit = err_longit**0.5/float(len(df))
####   df = df.drop(columns=["label"])

# test print
print(df.tail(3))


#X = df[["kin_pt_1", "kin_pt_2", "kin_pt_3", "kin_pt_4", "kin_phi_1", "kin_phi_2", "kin_phi_3", "kin_phi_4", "kin_y_1", "kin_y_2", "kin_y_3", "kin_y_4"]]  # non-redundant features
#X = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "cos_theta_p1_p12"]]  # over-redundant features
##X = df[["cos_theta_p1_p12", "cos_theta_p3_p34"]]  # very few features
#X = df[["ptZ1", "ptZ2", "yZ1", "yZ2", "phiZ1", "phiZ2", "cos_theta_p1_p12", "cos_theta_p3_p34"]]  # few features

#X = df[[]]
#if args.features == 'ct':
#elif args.features == 'ep':

X = df[["ptZ1", "ptZ2", "yZ1", "yZ2", "cos_theta_p1_p12", "cos_theta_p3_p34"]]  # few features (including decay angles)
X2 = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]]  # {px, py, pz, E} basis of inut features, as in NN
y = df["rLL"]   # target

# split into training and testing datasets
r_test = 1.0/3.0 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r_test, random_state=99)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=r_test, random_state=99)

end2 = time.time()
print("Elapsed (2nd step):", end2 - end1, "seconds. Now start training and testing steps...")


if args.model == 'all':

    #############################################################################
    #   # Random-Forest Regressor
    #############################################################################
    # define and train model
    model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=20, random_state=99, n_jobs=40, oob_score=True, bootstrap=True) # parallelise over 40 workers (~12 trees per worker)
    model.fit(X_train, y_train)
    model2 = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=20, random_state=99, n_jobs=40, oob_score=True, bootstrap=True) # parallelise over 40 workers (~12 trees per worker)
    model2.fit(X2_train, y2_train)
    
    print(' now save models ... ')
    joblib.dump({"model": model, "features": X.columns.tolist()},"trained_RFR_"+ str(args.order) +"_train_events_"+ str(len(X_train)) +"_basis_ct.joblib")
    joblib.dump({"model2": model2, "features": X2.columns.tolist()},"trained_RFR_"+ str(args.order) +"_train_events_"+ str(len(X2_train)) +"_basis_ep.joblib")
    print(' ... saved')
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
    y2_pred = np.clip(model2.predict(X2_test), 0, 1) # predict, avoiding out-of-range values

    mse = mean_squared_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0,1]


    resid = y_train - model.oob_prediction_ # resid = y_train - model.predict(X_train) would bias the training residuals
    var_model = lgb.LGBMRegressor(objective="mse", force_row_wise=True, verbose=-1)
    var_model.fit(X_train, resid**2)
    var_events = np.clip(var_model.predict(X_test), 0, None)

    resid2 = y2_train - model2.oob_prediction_ # resid = y_train - model.predict(X_train) would bias the training residuals
    var_model2 = lgb.LGBMRegressor(objective="mse", force_row_wise=True, verbose=-1)
    var_model2.fit(X2_train, resid2**2)
    var_events2 = np.clip(var_model2.predict(X2_test), 0, None)

    z_uu = (df.loc[X_test.index, "UU"]).to_numpy()
    z_uu2 = (df.loc[X2_test.index, "UU"]).to_numpy()
    z_ll = (df.loc[X_test.index, "LL"]).to_numpy()
    w_pred = np.array([z_uu[i] * y_pred[i] for i in range(0,len(z_uu))])
    w_pred2 = np.array([z_uu2[i] * y2_pred[i] for i in range(0,len(z_uu2))])
    w_err  = np.array([z_uu[i]**2 * var_events[i] for i in range(0,len(z_uu))])
    w_err2  = np.array([z_uu2[i]**2 * var_events2[i] for i in range(0,len(z_uu2))])

    
#    print(' sigma(unp) = %.4f' % (1e+03*z_uu.sum()/(N_tot*r_test)))
#    print(' sigma(LL) = %.4f' % (1e+03*z_ll.sum() /(N_tot*r_test)))

    sigLLsim = np.array([(1e+3*sigma_ll[0]),(1e+3*sigma_ll[1])])
    sigLLtrue = np.array([(1e+03*z_ll.sum() /(N_tot*r_test)),(1e+03*sigma_uu[1]*(y_test.sum()/len(y_test)))])
    sigLLpred = np.array([(1e+03*w_pred.sum() /(N_tot*r_test)), (1e+03*np.sqrt(w_err.sum()))/(N_tot*r_test)])
    sigLLpred2 = np.array([(1e+03*w_pred2.sum() /(N_tot*r_test)), (1e+03*np.sqrt(w_err2.sum()))/(N_tot*r_test)])

    # RFR
    print(' total number of train events ............................ ', len(y_train))
    print(' total number of test events ............................. ', len(y_test))
    print(' expected LL xsec (polarised simulation) ................. %.4f ' % (sigLLsim[0]), ' +- %.4f (MC) fb' % (sigLLsim[1]) )
    print(' estimated LL xsec (true-rLL reweighting, test) .......... %.4f ' % (sigLLtrue[0]), ' +- %.4f (MC) fb' % (sigLLtrue[1]) )
    print(' estimated LL xsec (pred-rLL reweighting, test, ct) ...... %.4f ' % (sigLLpred[0]), ' +- %.4f (model) fb' % (sigLLpred[1]) )
    print(' estimated LL xsec (pred-rLL reweighting, test, ep) ...... %.4f ' % (sigLLpred2[0]), ' +- %.4f (model) fb' % (sigLLpred2[1]) )
    print("\nmse, correlation (ct):", mse, corr, " \n")

    #print(' estimated LL xsec (true-rLL resampling, test+train) ..... %.4f ' % (1e+3*sigma_uu[0]*(float(n_longit)/float(len(df)))), ' +- %.4f (binomial) fb' % (1e+3*sigma_uu[0]*err_longit))
        

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=99,
        n_jobs=40
    )
    importances = result.importances_mean
    std = result.importances_std
    indices = np.argsort(importances)[::-1]

    result2 = permutation_importance(
        model2,
        X2_test,
        y2_test,
        n_repeats=10,
        random_state=99,
        n_jobs=40
    )
    importances2 = result2.importances_mean
    std2 = result2.importances_std
    indices2 = np.argsort(importances2)[::-1]


    end4 = time.time()
    print("Testing step:", end4 - end3, "seconds.")

#     #############################################################################
#     #   # Two-model approach with Light-GBM MSE-regressor for mean and variance
#     #############################################################################
#
#     print(" \n Light-GBM Regressor + model for train residuals\n")
#     reg = lgb.LGBMRegressor(objective="mse", alpha=0.5, n_estimators=500, max_depth=10, learning_rate=0.05, force_row_wise=True, verbose=-1)
#     reg.fit(X_train, y_train)
#     y_pred_2 = np.clip(reg.predict(X_test),0,1)
#     M_hat = y_pred_2.sum()/float(len(y_pred))
#
#     resid_2 = y_train - reg.predict(X_train)
#     var_model_2 = lgb.LGBMRegressor(objective="mse", force_row_wise=True, verbose=-1)
#     var_model_2.fit(X_train, resid_2**2)
#     sigma_M = np.sqrt(np.sum( np.clip(var_model_2.predict(X_test), 0, None)  ))/float(len(y_pred))
#
#     # LGBMR
#     print(' estimated LL xsec from new model (pred-rLL, test) ............ %.4f ' % (1e+3*sigma_uu[0]*M_hat), ' +- %.4f (train-residual regression) fb' % (1e+3*sigma_uu[0]*sigma_M))
#     print(" other model testing and training:", end5 - end4, "seconds.")


    end5 = time.time()


    #fig, (ax1, ax3, ax4, ax2) = plt.subplots(4, 1, figsize=(7, 15))
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11.5, 14))
    ax1 = axes[0, 0]
    ax3 = axes[0, 1]
    ax4 = axes[1, 0]
    ax2 = axes[1, 1]
    ax5 = axes[2, 0]
    ax6 = axes[2, 1]



    # z_uu = (df.loc[X_test.index, "UU"]).to_numpy()
    # z_ll = (df.loc[X_test.index, "LL"]).to_numpy()
    # w_pred = np.array([z_uu[i] * y_pred[i] for i in range(0,len(z_uu))])
    # w_err  = np.array([z_uu[i]**2 * var_events[i] for i in range(0,len(z_uu))])

    true_weights = z_ll*1e+03/(N_tot*r_test)
    
    # now plotting stuff
    bins = 40 # for physical observables
    norm_factor = 1e+03*sigma_uu[0]/float(len(y_pred))

    yep = df.loc[X_test.index, "kin_y_1"]
    yep2 = df.loc[X2_test.index, "kin_y_1"]
    ax1.set_title("Positron rapidity "+t_app)
    hist_vals, bin_edges = np.histogram(yep, bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.clip(np.digitize(yep, bin_edges) - 1, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    ax1.hist(
        yep,
        weights=true_weights/bin_width,
#        weights=y_test*(norm_factor/bin_width),
        bins=bins, histtype="step",
        linewidth=1,
        color="blue",
        label="true"
    )

    ax1.hist(
        yep2,
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    

    ax1.fill_between(
        bin_centers,
        (hist_vals - bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        (hist_vals + bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        alpha=0.3,
        color='red',
        label="RFR$_{\\tt ct}$",
        edgecolor='red', facecolor='red',
        step='mid'
    )
    ax1.set_xlim(-2.2,2.2)
    ax1.legend(loc='best',   borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax1.set_xlabel("y$_{\\tt e^+}$")
    ax1.set_ylabel("d$\\sigma/$d$y_{\\tt e^+}$ [fb]")


    cth = df.loc[X_test.index, "cos_theta_p1_p12"]
    cth2 = df.loc[X2_test.index, "cos_theta_p1_p12"]
    ax3.set_title("Positron decay angle "+t_app)
    hist_vals, bin_edges = np.histogram(cth, bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(cth, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    bin_width = bin_edges[1] - bin_edges[0]
    ax3.hist(cth, weights=true_weights/bin_width, bins=bins, histtype="step", linewidth=1, color="blue", label="true")

    ax3.hist(
        cth2,
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    
    ax3.fill_between(
        bin_centers,
        (hist_vals - bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        (hist_vals + bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        alpha=0.3,
        color='red',
        label="RFR$_{\\tt ct}$",
        edgecolor='red',
        facecolor='red',
        step='mid'
    )
#    ax3.hist(cth, weights=y_pred, bins=20, histtype="step", linewidth=1, color="red", label="RFR pred.", density=True)
#    ax3.hist(cth, weights=y_pred_2, bins=20, histtype="step", linewidth=1, color="green", label="pred. (LGBM)", density=True)
    ax3.set_xlim(-0.95,0.95)
    ax3.legend(loc='best',   borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax3.set_xlabel("cos$\\theta^*_{\\tt e^+}$")
    ax3.set_ylabel("d$\\sigma/$dcos$\\theta^*_{\\tt e^+}$ [fb]")





    
    ax2.set_title("$r_{\\tt LL}$ label "+t_app)
    ax2.hist(
        y_test,
        weights=z_uu,
        range=(-0.02, 0.4), bins=40,
        histtype="step",
        linewidth=1,
        color="blue",
        label="true",
        density=True
    )
    ax2.hist(
        y2_pred,
        weights=z_uu,
        range=(-0.02, 0.4), bins=40,
        histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$",
        density=True
    )
    ax2.hist(
        y_pred,
        weights=z_uu,
        range=(-0.02, 0.4), bins=40,
        histtype="step",
        linewidth=1.2,
        color="red",
        label="RFR$_{\\tt ct}$",
        density=True
    )
    ax2.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax2.set_xlabel("r$_{\\tt LL}$")
    ax2.set_ylabel("Normalised distribution")
    
    
    ax2.text(0.6, 0.50, f"$\\sigma$(LL, MC sim)   = {sigLLsim[0]:.4f}({(sigLLsim[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")
    ax2.text(0.6, 0.46, f"$\\sigma$(LL, true rLL) = {sigLLtrue[0]:.4f}({(sigLLtrue[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")
    ax2.text(0.6, 0.42, f"$\\sigma$(LL, RFR-ct) = {sigLLpred[0]:.4f}({(sigLLpred[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")
    ax2.text(0.6, 0.38, f"$\\sigma$(LL, RFR-ep) = {sigLLpred2[0]:.4f}({(sigLLpred2[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")

    ptep = df.loc[X_test.index, "pt4l"]
    ptep2 = df.loc[X2_test.index, "pt4l"]
    ax4.set_title("Four-lepton transverse momentum "+t_app)
    hist_vals, bin_edges = np.histogram(ptep, range=(0.0,250.0), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(ptep, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    ax4.hist(ptep, range=(0.0,250.0),  weights=true_weights/bin_width, bins=bins, histtype="step", linewidth=1, color="blue", label="true")
    ax4.hist(
        ptep2,
        range=(0.0,250.0), 
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    
    ax4.fill_between(
        bin_centers,
        (hist_vals - bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        (hist_vals + bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        alpha=0.3,
        color='red',
        label="RFR$_{\\tt ct}$",
        edgecolor='red',
        facecolor='red',
        step='mid'
    )
    ax4.set_xlim(0.5,249.5)
    ax4.set_ylim(1e-06,1e-01)
    ax4.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax4.set_xlabel("$p_{\\tt T, 4\ell}$ [GeV]")
    ax4.set_ylabel("d$\\sigma/$d$p_{\\tt T, 4\ell}$ [fb/GeV]")
    ax4.set_yscale("log")

    # h_to_fit, bef = np.histogram(ptep, range=(0.0,250.0), bins=100, weights=y_test*(norm_factor/bin_width))
    # x = (bef[:-1] + bef[1:])/2.0 
    # 
    # 
    # # p0 = [6.0, 0.5, 2]  # initial guess (n, T)
    # 
    # params, cov = curve_fit(fitting_pdf, x, h_to_fit, p0=[0.035, 4.0, 2.7, 1.1], bounds=([0.02, 2.6, 2.2, 0.5], [0.05, 4.2, 3.1, 3.5]))
    # 
    # pT_plot = np.linspace(0, 250, 100)
    # fit_curve = fitting_pdf(pT_plot, params[0], params[1], params[2], params[3])
    # ax4.plot(pT_plot, fit_curve, lw=2)

    # print ('fit results for C, n, T, pto = ', params[0], params[1], params[2], params[3])
    
    #dval01, dcov01 = curve_fit(distorted, x_an, np.array(test[2]), sigma=np.array(test[3]),  bounds=bounds, absolute_sigma=True, maxfev=20000, method='trf')
   

    

    
    #pT = ptep.to_numpy()
    #print('shape of pT', pT.size)



    
#    ptep = df.loc[X_test.index, "kin_pt_1"]
#    ax4.set_title("Positron transverse momentum "+t_app)
#    hist_vals, bin_edges = np.histogram(ptep, range=(0.0,200.0), bins=bins, weights=y_pred)
#    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#    bin_indices = np.digitize(ptep, bin_edges) - 1
#    bin_indices = np.clip(bin_indices, 0, bins-1)
#    bin_var = np.zeros(bins)
#    bin_width = bin_edges[1] - bin_edges[0]
#    for i, b in enumerate(bin_indices):
#        bin_var[b] += var_events[i]
#    bin_sigma = np.sqrt(bin_var)
#    ax4.hist(ptep, range=(0.0,200.0),  weights=y_test*(norm_factor/bin_width), bins=bins, histtype="step", linewidth=1, color="blue", label="true")
#    ax4.fill_between(
#        bin_centers,
#        (hist_vals - bin_sigma)*(norm_factor/bin_width),#_density,
#        (hist_vals + bin_sigma)*(norm_factor/bin_width),#_density,
#        alpha=0.3,
#        color='red',
#        label="RFR pred.",
#        edgecolor='red', facecolor='red',
#        step='mid'
#    )
#    ax4.set_xlim(5.0,195.0)
#    ax4.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
#    ax4.set_xlabel("$p_{\\tt T, e^+}$ [GeV]")
#    ax4.set_ylabel("d$\\sigma/$d$p_{\\tt T, e^+}$ [fb/GeV]")
#    ax4.set_yscale("log")



    
    ax5.set_title("Permutation importance for RFR$_{\\tt ct}$ "+t_app)
    ax5.bar(range(len(importances)), importances[indices], yerr=0, color="red", alpha = 0.35) #, yerr=std[indices])
    latex_labels = [
        r"cos$\theta^*_{\mathrm{e}^+}$",
        r"cos$\theta^*_{\mu^+}$",
        r"$p_{\mathrm{T}, e^+e^-}$",
        r"$p_{\mathrm{T}, \mu^+\mu^-}$",
        r"$y_{\mathrm{ e}^+\mathrm{e}^-}$",
        r"$y_{\mu^+\mu^-}$"
        #r"$\phi_{\mathrm{ e}^+\mathrm{e}^-}$",
        #r"$\phi_{\mu^+\mu^-}$"
    ]
    ax5.set_xticks(range(len(importances)),
                   X_test.columns[indices],
                   #latex_labels,
                   rotation=45,
                   ha="right")
    ax5.set_ylabel("Decrease in performance")

    ax6.set_title("Permutation importance for RFR$_{\\tt ep}$ "+t_app)
    ax6.bar(range(len(importances2)), importances2[indices2], yerr=0, color="red", alpha = 0.35) #, yerr=std[indices])
    ax6.set_xticks(range(len(importances2)),
                   X2_test.columns[indices2],
                   rotation=45,
                   ha="right")
    ax6.set_ylabel("Decrease in performance")


# uncomm for scatter plot #     ax6.set_title("True vs pred. $r_{\\tt LL}$ labels "+t_app)
# uncomm for scatter plot #     #dr = y_pred - y_test
# uncomm for scatter plot #     #ax6.hist2d(y_test, y_pred, bins=100, cmap="viridis")
# uncomm for scatter plot #     #ax6.set_colorbar(label="events")
# uncomm for scatter plot # 
# uncomm for scatter plot #     #ax6.scatter(y_test, y_pred, s=5, alpha=0.2, color='red')
# uncomm for scatter plot #     hb = ax6.hexbin(y_test, y_pred, gridsize=40, cmap="viridis", mincnt=1, norm=colors.LogNorm())
# uncomm for scatter plot #     plt.colorbar(hb, ax=ax6, label="events")
# uncomm for scatter plot #     #ax6.axhline(0, color='blue', linestyle='--', linewidth=0.85)
# uncomm for scatter plot #     ax6.set_ylabel("RFR pred.")
# uncomm for scatter plot #     ax6.set_xlabel("true ")
# uncomm for scatter plot #     ax6.set_xlim(-0.1,0.4)
# uncomm for scatter plot #     ax6.set_ylim(-0.1,0.4)
# uncomm for scatter plot #     ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1)

    plt.tight_layout()
    fig.savefig("test_random_forest_regressor_"+ str(args.order) +"_test_events_" + str(len(y_pred)) + "_basis_both.pdf")
    plt.close()


    end6 = time.time()
    print("Plotting step:", end6 - end5, "seconds.")



endall = time.time()
print("\n Full elapsed time:", endall - start, "seconds. Done.\n")
#plt.show()













