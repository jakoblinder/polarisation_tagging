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


parser = argparse.ArgumentParser(description="Which events do you want?")
parser.add_argument('--order',  '-p', type=str, choices=['lo', 'lows', 'nlo', 'nlops'])
parser.add_argument('--data',   '-d', type=str, choices=['reduced', 'full'])
parser.add_argument('--model',  '-m', type=str, choices=['all'])
#parser.add_argument('--features',  '-f', type=str, choices=['ep', 'ct'])
args = parser.parse_args()

N_lhe = 100000
sigma_uu = np.array([0.11245290E-01, 0.37648619E-05])
sigma_ll = np.array([0.6574E-03, 0.0002E-03])
data_dir = Path("../../events/ML_FILES/UU_LO")
t_app = str("(LO, fiducial)")
nr_lhef = 26
if str(args.data) == 'reduced':
    nr_lhef = 3
if str(args.order) == 'lows':
    print(' You are parsing LOwS events')
    N_lhe = 100000
    sigma_uu = np.array([0.11348889e-01, 0.37664237e-05])
    sigma_ll = np.array([0.66608205e-03, 0.50535066e-06])
    data_dir = Path("../../events/ML_FILES/UU_LOwS")
    nr_lhef = 26 # 51
    t_app = str("(NLO QCD, LHE, fiducial)")
    if str(args.data) == 'reduced':
        nr_lhef = 3
elif str(args.order) == 'nlo':
    print(' You are parsing NLO LHE events')
    N_lhe = 50000
    sigma_uu = np.array([15.183819E-03, 0.0071627436E-03])
    sigma_ll = np.array([0.8918E-03, 0.0003E-03])
    data_dir = Path("../../events/ML_FILES/UU_NLO")
    nr_lhef = 51 # 51
    t_app = str("(NLO QCD, LHE, fiducial)")
    if str(args.data) == 'reduced':
        nr_lhef = 6
elif str(args.order) == 'nlops':
    print(' You are parsing NLO+PS events')
    N_lhe = 50000
    sigma_uu = np.array([0.15184142e-01, 0.71627344e-05])
    sigma_ll = np.array([0.8918E-03, 0.0003E-04])
    data_dir = Path("../../events/ML_FILES/UU_NLO")
    nr_lhef = 51 # 51
    t_app = str("(NLOPS fiducial)")
    if str(args.data) == 'reduced':
        nr_lhef = 6
else:
    print(' You are parsing LO LHE events')

N_tot = N_lhe * (nr_lhef-1)

used_weights = {"UU", "LL", "LT", "TL", "TT"}
all_events = []
start = time.time()
for i in range(1, min(10,nr_lhef)):
    filepath = data_dir/f"pwgevents-000{i}.ml"
    if str(args.order) == 'nlops':
        filepath = data_dir/f"output_shower_events-000{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
for i in range(10, max(10,nr_lhef)):
    filepath = data_dir/f"pwgevents-00{i}.ml"
    if str(args.order) == 'nlops':
        filepath = data_dir/f"output_shower_events-00{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
end1 = time.time()
print("Elapsed (1st step):", end1 - start, "seconds")

df = pd.DataFrame(all_events)

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
Mv1 = np.zeros(len(df))
dphiee = np.zeros(len(df))
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
    Mv1[i] = (p12[i,3]**2-p12[i,0]**2-p12[i,1]**2-p12[i,2]**2)**0.5
    ptv1[i] = (p12[i,0]**2+p12[i,1]**2)**0.5
    ptv2[i] = (p34[i,0]**2+p34[i,1]**2)**0.5
    ptvv[i] = ((p12[i,0]+p34[i,0])**2+(p12[i,1]+p34[i,1])**2)**0.5
    yv1[i]  = 0.5*np.log(( p12[i,3] + p12[i,2] ) / ( p12[i,3] - p12[i,2] ))
    yv2[i]  = 0.5*np.log(( p34[i,3] + p34[i,2] ) / ( p34[i,3] - p34[i,2] ))
    phiv1[i] = np.arctan2(p12[i,1],p12[i,0])
    phiv2[i] = np.arctan2(p34[i,1],p34[i,0])
    dphiee[i] = (180.0/np.pi)*min(abs(np.arctan2(p1[i,1],p1[i,0]) - np.arctan2(p2[i,1],p2[i,0])), 2.0*np.pi-abs(np.arctan2(p1[i,1],p1[i,0]) - np.arctan2(p2[i,1],p2[i,0])))

    
df["cos_theta_p1_p12"] = cos_theta
df["cos_theta_p3_p34"] = cos_thetab
df["ptZ1"] = ptv1
df["ptZ2"] = ptv2
df["yZ1"] = yv1
df["yZ2"] = yv2
df["phiZ1"] = phiv1
df["phiZ2"] = phiv2
df["pt4l"] = ptvv
df["MZ1"] = Mv1
df["dphiZ1"] = dphiee  


df["rLL"] = df["LL"] / df["UU"]
df = df.drop(columns=["LT", "TL", "TT"]) 
print('size of the whole dataset (train + test) = ', len(df))

print(df.tail(3))

X = df[["ptZ1", "ptZ2", "yZ1", "yZ2", "cos_theta_p1_p12", "cos_theta_p3_p34"]]
X2 = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]]
y = df["rLL"]

r_test = 1.0/3.0 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r_test, random_state=99)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=r_test, random_state=99)

end2 = time.time()
print("Elapsed (2nd step):", end2 - end1, "seconds. Now start training and testing steps...")


if args.model == 'all':

    #############################################################################
    #   # Random-Forest Regressor
    #############################################################################
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


    end5 = time.time()



    f = open("histograms_rfr_ct_" + args.order+ ".top", "w")
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(11.5, 17))
    ax1 = axes[0, 0]
    ax3 = axes[0, 1]
    ax4 = axes[1, 0]
    ax2 = axes[1, 1]
    ax5 = axes[2, 0]
    ax6 = axes[2, 1]
    ax7 = axes[3, 0]
    ax8 = axes[3, 1]
    ax9 = axes[4, 0]
    ax10= axes[4, 1]

    true_weights = z_ll*1e+03/(N_tot*r_test)
    
    norm_factor = 1e+03*sigma_uu[0]/float(len(y_pred))
    
    bins = 20 # for physical observables
    yep = df.loc[X_test.index, "kin_y_1"]
    yep2 = df.loc[X2_test.index, "kin_y_1"]
    ax1.set_title("Positron rapidity "+t_app)
    hist_vals, bin_edges = np.histogram(yep, range=(-2.5,2.5), bins=bins, weights=w_pred)
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
        range=(-2.5,2.5), bins=bins, histtype="step",
        linewidth=1,
        color="blue",
        label="true"
    )
    ax1.hist(
        yep2,
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        range=(-2.5,2.5), bins=bins, histtype="step",
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
    ax1.set_xlim(-2.37,2.37)
    ax1.legend(loc='best',   borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax1.set_xlabel("y$_{\\tt e^+}$")
    ax1.set_ylabel("d$\\sigma/$d$y_{\\tt e^+}$ [fb]")

    
    print('\n# yep ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)



    bins = 30 # for physical observables
    dphill = df.loc[X_test.index, "dphiZ1"]
    dphill2 = df.loc[X2_test.index, "dphiZ1"]
    ax7.set_title("Positron-electron azimuthal separation "+t_app)
    hist_vals, bin_edges = np.histogram(dphill, range=(0,180), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.clip(np.digitize(dphill, bin_edges) - 1, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    ax7.hist(
        dphill,
        weights=true_weights/bin_width,
        range=(0,180), bins=bins, histtype="step",
        linewidth=1,
        color="blue",
        label="true"
    )
    ax7.hist(
        dphill2,
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        range=(0,180), bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    ax7.fill_between(
        bin_centers,
        (hist_vals - bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        (hist_vals + bin_sigma)*(1e+03/(N_tot*r_test)/bin_width),#_density,
        alpha=0.3,
        color='red',
        label="RFR$_{\\tt ct}$",
        edgecolor='red', facecolor='red',
        step='mid'
    )
    ax7.set_xlim(3,177)
    ax7.legend(loc='best',   borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax7.set_xlabel("$\Delta\phi_{\\tt e^+e^-}$")
    ax7.set_ylabel("d$\\sigma/$d$\Delta\phi_{\\tt e^+e^-}$ [fb]")

    print('\n# dphiee ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)






    
    bins = 40
    cth = df.loc[X_test.index, "cos_theta_p1_p12"]
    cth2 = df.loc[X2_test.index, "cos_theta_p1_p12"]
    ax3.set_title("Positron decay angle "+t_app)
    hist_vals, bin_edges = np.histogram(cth, range=(-1.0,1.0), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(cth, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    bin_width = bin_edges[1] - bin_edges[0]
    ax3.hist(
        cth,
        weights=true_weights/bin_width,
        range=(-1.0,1.0), bins=bins, histtype="step",
        linewidth=1,
        color="blue",
        label="true"
    )
    ax3.hist(
        cth2,
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        range=(-1.0,1.0), bins=bins, histtype="step",
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
    ax3.set_xlim(-0.975,0.975)
    ax3.legend(loc='best',   borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax3.set_xlabel("cos$\\theta^*_{\\tt e^+}$")
    ax3.set_ylabel("d$\\sigma/$dcos$\\theta^*_{\\tt e^+}$ [fb]")

    print('\n# cthep ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)


    
    bins = 102
    rll1 = y_test
    rll2 = y2_test
    ax2.set_title("$r_{\\tt LL}$ label "+t_app)
    hist_vals, bin_edges = np.histogram(rll1, range=(-0.02, 1.0), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(rll1, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    bin_width = bin_edges[1] - bin_edges[0]
    ax2.hist(rll1, weights=true_weights/bin_width, range=(-0.02, 1.0), bins=bins, histtype="step", linewidth=1, color="blue", label="true")
    ax2.hist(
        rll2,
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        range=(-0.02, 1.0),
        bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    ax2.fill_between(
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
    ax2.set_xlim(-0.02, 1.0)
    ax2.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax2.set_xlabel("r$_{\\tt LL}$")
    ax2.set_ylabel("Normalised distribution")
    
    print('\n# rll ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)

    ax2.text(0.6, 0.50, f"$\\sigma$(LL, MC sim)   = {sigLLsim[0]:.4f}({(sigLLsim[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")
    ax2.text(0.6, 0.46, f"$\\sigma$(LL, true rLL) = {sigLLtrue[0]:.4f}({(sigLLtrue[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")
    ax2.text(0.6, 0.42, f"$\\sigma$(LL, RFR-ct) = {sigLLpred[0]:.4f}({(sigLLpred[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")
    ax2.text(0.6, 0.38, f"$\\sigma$(LL, RFR-ep) = {sigLLpred2[0]:.4f}({(sigLLpred2[1]*1e+04):.0f}) fb",transform=ax2.transAxes,ha="center")

    



    
#    # TODO: weight with unpolarised cross section (density = False, divide by binwidth, name of the histo for powheg rll)
#    ax2.hist(
#        y_test,
#        weights=z_uu,
#        range=(-0.02, 1.0), bins=102,
#        histtype="step",
#        linewidth=1,
#        color="blue",
#        label="true",
#        density=True
#    )
#    ax2.hist(
#        y2_pred,
#        weights=z_uu,
#        range=(-0.02, 1.0), bins=100,
#        histtype="step",
#        linewidth=1.2,
#        color="green",
#        label="RFR$_{\\tt ep}$",
#        density=True
#    )
#    ax2.hist(
#        y_pred,
#        weights=z_uu,
#        range=(-0.02, 0.4), bins=40,
#        histtype="step",
#        linewidth=1.2,
#        color="red",
#        label="RFR$_{\\tt ct}$",
#        density=True
#    )

    bins = 60 # for physical observables
    pt4lep = df.loc[X_test.index, "pt4l"]
    pt4lep2 = df.loc[X2_test.index, "pt4l"]
    ax4.set_title("Four-lepton transverse momentum "+t_app)
    hist_vals, bin_edges = np.histogram(pt4lep, range=(0.0,300.0), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(pt4lep, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    ax4.hist(pt4lep, range=(0.0,300.0),  weights=true_weights/bin_width, bins=bins, histtype="step", linewidth=1, color="blue", label="true")
    ax4.hist(
        pt4lep2,
        range=(0.0,300.0), 
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
    ax4.set_xlim(2.5,297.5)
    ax4.set_ylim(1e-06,1e-01)
    ax4.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax4.set_xlabel("$p_{\\tt T, 4\ell}$ [GeV]")
    ax4.set_ylabel("d$\\sigma/$d$p_{\\tt T, 4\ell}$ [fb/GeV]")
    ax4.set_yscale("log")

    print('\n# pt4l ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)




    bins = 40 # for physical observables
    ptep = df.loc[X_test.index, "kin_pt_1"]
    ptep2 = df.loc[X2_test.index, "kin_pt_1"]
    ax8.set_title("Positron transverse momentum "+t_app)
    hist_vals, bin_edges = np.histogram(ptep, range=(0.0,400.0), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(ptep, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    ax8.hist(ptep, range=(0.0,400.0),  weights=true_weights/bin_width, bins=bins, histtype="step", linewidth=1, color="blue", label="true")
    ax8.hist(
        ptep2,
        range=(0.0,400.0), 
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    
    ax8.fill_between(
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
    ax8.set_xlim(5,395)
    ax8.set_ylim(1e-06,1e-01)
    ax8.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax8.set_xlabel("$p_{\\tt T, e^+}$ [GeV]")
    ax8.set_ylabel("d$\\sigma/$d$p_{\\tt T, e^+}$ [fb/GeV]")
    ax8.set_yscale("log")

    print('\n# ptep ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)


    bins = 40 # for physical observables
    mee = df.loc[X_test.index, "MZ1"]
    mee2 = df.loc[X2_test.index, "MZ1"]
    ax9.set_title("Positron-electron invariant mass "+t_app)
    hist_vals, bin_edges = np.histogram(mee, range=(81,101), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(mee, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)
    ax9.hist(mee, range=(81,101),  weights=true_weights/bin_width, bins=bins, histtype="step", linewidth=1, color="blue", label="true")
    ax9.hist(
        mee2,
        range=(81,101), 
        weights=w_pred2*1e+03/(N_tot*r_test)/bin_width,
        bins=bins, histtype="step",
        linewidth=1.2,
        color="green",
        label="RFR$_{\\tt ep}$"
    )
    
    ax9.fill_between(
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
    ax9.set_xlim(82,100)
    #ax9.set_ylim(1e-06,1e-01)
    ax9.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax9.set_xlabel("$M_{\\tt e^+e^-}$ [GeV]")
    ax9.set_ylabel("d$\\sigma/$d$M_{\\tt e^+e^-}$ [fb/GeV]")
    ax9.set_yscale("log")

    print('\n# mepem ', file=f)
    left_edges  = bin_centers - 0.5 * bin_width
    right_edges = bin_centers + 0.5 * bin_width
    norm = 1e+03 / (N_tot * r_test * bin_width)
    values = hist_vals * norm
    errors = bin_sigma * norm
    for i in range(len(values)):
        left  = left_edges[i]
        right = right_edges[i]
        val   = values[i]
        err   = errors[i]
        print(f"{left:.6e} {right:.6e} {val:.6e} {err:.6e}", file=f)


    
    
    ax5.set_title("Permutation importance for RFR$_{\\tt ct}$ "+t_app)
    ax5.bar(range(len(importances)), importances[indices], yerr=0, color="red", alpha = 0.35) #, yerr=std[indices])
    latex_labels = [
        r"cos$\theta^*_{\mathrm{e}^+}$",
        r"cos$\theta^*_{\mu^+}$",
        r"$p_{\mathrm{T}, e^+e^-}$",
        r"$p_{\mathrm{T}, \mu^+\mu^-}$",
        r"$y_{\mathrm{ e}^+\mathrm{e}^-}$",
        r"$y_{\mu^+\mu^-}$"
    ]
    ax5.set_xticks(range(len(importances)),
                   X_test.columns[indices],
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




    #################################### new histos
    bins = 60 # for physical observables
    ptep = df.loc[X_test.index, "pt4l"]
    ptep2 = df.loc[X2_test.index, "pt4l"]
    ax4.set_title("Four-lepton transverse momentum "+t_app)
    hist_vals, bin_edges = np.histogram(ptep, range=(0.0,300.0), bins=bins, weights=w_pred)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(ptep, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    bin_var = np.zeros(bins)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, b in enumerate(bin_indices):
        bin_var[b] += w_err[i]
    bin_sigma = np.sqrt(bin_var)

    ax4.hist(
        ptep,
        range=(0.0,300.0),
        weights=true_weights/bin_width,
        bins=bins, histtype="step", linewidth=1,
        color="blue",
        label="true"
    )
    ax4.hist(
        ptep2,
        range=(0.0,300.0), 
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
    ax4.set_xlim(2.5,297.5)
    ax4.set_ylim(1e-06,1e-01)
    ax4.legend(loc='best', borderpad=0.5, framealpha=0.9, frameon=False, ncol = 1)
    ax4.set_xlabel("$p_{\\tt T, 4\ell}$ [GeV]")
    ax4.set_ylabel("d$\\sigma/$d$p_{\\tt T, 4\ell}$ [fb/GeV]")
    ax4.set_yscale("log")







    
    
    plt.tight_layout()
    fig.savefig("test_random_forest_regressor_"+ str(args.order) +"_test_events_" + str(len(y_pred)) + "_basis_both.pdf")
    plt.close()

    end6 = time.time()
    print("Plotting step:", end6 - end5, "seconds.")


endall = time.time()
print("\n Full elapsed time:", endall - start, "seconds. Done.\n")













