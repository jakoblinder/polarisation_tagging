import numpy as np
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

# boost qx written in some frame to the rest frame of px
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

# test fitting function
def fitting_pdf(pT, C, n, T, pto):
    return C * pT * (1.0 + pT / (n * T))**(-n) * np.exp(-pto**2/pT**2)

