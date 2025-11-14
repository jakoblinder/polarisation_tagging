# Utility package to scale target values

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

def boost_into_four_lepton_cm_frame(features):
    f_sum = np.array([
        features[0]+features[4]+features[8]+features[12],
        features[1]+features[5]+features[9]+features[13],
        features[2]+features[6]+features[10]+features[14],
        features[3]+features[7]+features[11]+features[15]
    ])
    features[0:4] = torch.Tensor(boostinv(np.array(features[0:4]),f_sum))    
    features[4:8] = torch.Tensor(boostinv(np.array(features[4:8]),f_sum))    
    features[8:12] = torch.Tensor(boostinv(np.array(features[8:12]),f_sum))   
    features[12:16] = torch.Tensor(boostinv(np.array(features[12:16]),f_sum))   
    return features

def scale_target(x):
    # Utility function to scale target values
    return x * 1000  # Scale target by 1000
