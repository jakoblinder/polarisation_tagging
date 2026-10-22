import numpy as np
import pandas as pd
import re

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

# events of one .ml file as a DataFrame (cheap to send back from a worker process when files are parsed in parallel)
def parse_ml_events_frame(filepath, useful_weights):
    return pd.DataFrame(parse_ml_events(filepath, useful_weights))

# boost qx written in some frame to the rest frame of px, four-momenta as (px, py, pz, E) in the last axis:
# a single four-momentum or arrays of shape (n, 4) for n events at once
def boostinv(qx, px):
    rmboost = np.maximum(px[...,3]**2-px[...,0]**2-px[...,1]**2-px[...,2]**2, 0.0)**0.5
    aux  = (qx[...,3]*px[...,3]-qx[...,0]*px[...,0]-qx[...,1]*px[...,1]-qx[...,2]*px[...,2])/rmboost
    aaux = (aux+qx[...,3])/(px[...,3]+rmboost)
    return np.stack([qx[...,0]-aaux*px[...,0], qx[...,1]-aaux*px[...,1], qx[...,2]-aaux*px[...,2], aux], axis=-1)

# test fitting function
def fitting_pdf(pT, C, n, T, pto):
    return C * pT * (1.0 + pT / (n * T))**(-n) * np.exp(-pto**2/pT**2)

