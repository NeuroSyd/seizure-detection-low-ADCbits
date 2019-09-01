import numpy as np

def get_onset_indices(latencies):
    indices = [0]
    for i in range(1,len(latencies)):
        if (latencies[i]==0) and (latencies[i-1]>0):
            indices.append(i)
    indices.append(len(latencies))
    return indices

def group_seizure(X, y, latencies):
    Xg = []
    yg = []
    onset_indices = get_onset_indices(latencies)
    print ('onset_indices', onset_indices)
    print (len(X), len(y))
    for i in range(len(onset_indices)-1):
        Xg.append(
            np.concatenate(X[onset_indices[i]:onset_indices[i+1]], axis=0)
        )
        yg.append(
            np.array(y[onset_indices[i]:onset_indices[i+1]])
        )
    return Xg, yg
