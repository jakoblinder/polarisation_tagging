import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import re
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


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


# parsing input argument (LO / NLO QCD)
parser = argparse.ArgumentParser(description="Which events do you want?")
parser.add_argument('--order',  '-p', type=str, choices=['lo', 'nlo'])
args = parser.parse_args()

# initialisation and choice of LHE-ML dataset
sigma_uu = np.array([0.11245290E-01, 0.37648619E-05])
data_dir = Path("../events/ML_FILES/UU_LO")
#if str(data_dir).find("NLO") != -1:
if str(args.order) == 'nlo':
    print(' You are parsing NLO QCD LHE events')
    sigma_uu = np.array([0.15183819E-01,0.71627436E-05])
    data_dir = Path("../events/ML_FILES/UU_NLO")

else:
    print(' You are parsing LO LHE events')
    
# access ml unpolarised events and various weights 
used_weights = {"UU", "LL", "LT", "TL", "TT"}
all_events = []
start = time.time()
for i in range(1, 10):   
    filepath = data_dir/f"pwgevents-000{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
for i in range(10, 50): 
    filepath = data_dir/f"pwgevents-00{i}.ml"
    print('parsing file ', filepath)
    all_events.extend(parse_ml_events(filepath,used_weights))
end1 = time.time()
print("Elapsed (1st step):", end1 - start, "seconds")

# frame events with relevant information (kinematics, rLL, rLT, rTL, rTT)
df = pd.DataFrame(all_events)
df["kin_pt_1"] = (df["x1"]**2+df["x2"]**2)**0.5
df["kin_pt_2"] = (df["x5"]**2+df["x6"]**2)**0.5
df["kin_pt_3"] = (df["x9"]**2+df["x10"]**2)**0.5
df["kin_pt_4"] = (df["x13"]**2+df["x14"]**2)**0.5
df["kin_phi_1"] = np.arctan2(df["x2"],df["x1"])
df["kin_phi_2"] = np.arctan2(df["x6"],df["x5"])
df["kin_phi_3"] = np.arctan2(df["x10"],df["x9"])
df["kin_phi_4"] = np.arctan2(df["x14"],df["x13"])
df["kin_y_1"] = 0.5*np.log((df["x3"]+df["x0"]) / (df["x3"]-df["x0"]))
df["kin_y_2"] = 0.5*np.log((df["x7"]+df["x4"]) / (df["x7"]-df["x4"]))
df["kin_y_3"] = 0.5*np.log((df["x11"]+df["x8"]) / (df["x11"]-df["x8"]))
df["kin_y_4"] = 0.5*np.log((df["x15"]+df["x12"]) / (df["x15"]-df["x12"]))
df["rLL"] = df["LL"] / df["UU"]
df["rLT"] = df["LT"] / df["UU"]
df["rTL"] = df["TL"] / df["UU"]
df["rTT"] = df["TT"] / df["UU"]
df = df.drop(columns=["UU", "LL", "LT", "TL", "TT", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"])
print('size of the whole dataset (train + test) = ', len(df))

# label events with basic hit-or-miss 
n_longit = 0
df["label"] = 0
x = np.random.rand(len(df))
for i in range(0,len(df)):
    if x[i] < df["rLL"].iloc[i]:
        df.at[i,"label"] = 1
        n_longit += 1
print(' estimated longitudinal cross section (hit-or-miss with rLL sampling, using full dataset) ... %.4f ' % (1e+3*sigma_uu[0]*(float(n_longit)/float(len(df)))))
df = df.drop(columns=["rLT", "rTL", "rTT"])
        
# test print 
print(df.tail(3))


#############################################################################
#   # Random-Forest Regressor
#############################################################################

# split into training and testing datasets
X = df[["kin_pt_1", "kin_pt_2", "kin_pt_3", "kin_pt_4", "kin_phi_1", "kin_phi_2", "kin_phi_3", "kin_phi_4", "kin_y_1", "kin_y_2", "kin_y_3", "kin_y_4"]]  # features
y = df["rLL"]   # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
end2 = time.time()
print("Elapsed (2nd step):", end2 - end1, "seconds. Now start training step...")

# define model 
model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=20, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)
end3 = time.time()
print("Training step:", end3 - end2, "seconds. Now start testing...")

# testing
#y_pred = model.predict(X_test) # predict, allowing for out-of-range values
y_pred = np.clip(model.predict(X_test), 0, 1) # predict, avoiding out-of-range values
mse = mean_squared_error(y_test, y_pred)
print("mean squared error:", mse)
corr = np.corrcoef(y_test, y_pred)[0,1]
print("Correlation:", corr)

print(' total parsed events ............................................. ', len(y_test))
print(' estimated longitudinal cross section (true-rLL reweighting) ..... %.4f ' % (1e+3*sigma_uu[0]*(y_test.sum()/len(y_test))), ' +- %.4f fb' % (1e+3*sigma_uu[1]*(y_test.sum()/len(y_test))) )
print(' estimated longitudinal cross section (pred-rLL reweighting) ..... %.4f ' % (1e+3*sigma_uu[0]*(y_pred.sum()/len(y_pred))), ' +- %.4f fb' % (1e+3*sigma_uu[0]*(y_pred.sum()**0.5/len(y_pred))) )

end4 = time.time()
print("Testing step:", end4 - end3, "seconds. Done.")

# now plotting stuff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.set_title("Rapidity of the positron")
ax1.hist(X_test["kin_y_1"], weights=y_test, bins=40, histtype="step", linewidth=1, color="blue", label="true", density=True)
ax1.hist(X_test["kin_y_1"], weights=y_pred, bins=40, histtype="step", linewidth=1, color="red", label="predicted", density=True)
ax1.legend()
ax1.set_xlabel("y$_{\\tt e^+}$")
ax1.set_ylabel("Normalised distributions")
#ax1.set_yscale("log")

ax2.set_title("Label distribution")
ax2.hist(y_test, range=(-0.1, 1.1), bins=40, histtype="step", linewidth=1, color="blue", label="true", density=True)
ax2.hist(y_pred, range=(-0.1, 1.1), bins=40, histtype="step", linewidth=1, color="red", label="predicted", density=True)
ax2.legend()
ax2.set_xlabel("r$_{\\tt LL}$")
ax2.set_ylabel("Normalised distributions")
#ax2.set_yscale("log")

plt.tight_layout()
fig.savefig("test_random_forest_regressor_"+ str(args.order) +"_test_events_" + str(len(y_pred)) + ".pdf")
#plt.show()



# plt.scatter(y_test, y_pred, s=3, alpha=0.3)
# plt.xlabel("True $r_{\\tt LL}$")
# plt.ylabel("Guessed $r_{\\tt LL}$")
# plt.plot([0,1],[0,1],'r--')
# plt.show()

#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
#  scaler = StandardScaler()
#  X_train = scaler.fit_transform(X_train)
#  X_test = scaler.transform(X_test)

#  features = [c for c in df.columns if c.startswith("kin_pt")]
#  target = "rLL" 
#  Xs = df[features]
#  correlation = Xs.corr(method="spearman")
#  correlation_with_target = df[features + [target]].corr()[target].sort_values()
#  print(correlation_with_target)
#  plt.figure(figsize=(10, 8))
#  plt.imshow(correlation, aspect="auto")
#  plt.colorbar(label="Correlation")
#  plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
#  plt.yticks(range(len(correlation)), correlation.columns)
#  plt.title("Correlation Matrix for Kinematic Features")
#  plt.tight_layout()
#  plt.show()















