# Rocking the million

import numpy as np
import h5py

def label(file='data.h5'):
    """ 
    Label grains based on rocking curve minima. 
    Returns
    -------
    labels : 2D numpy array
        Label matrix indicating grain assignments.
    """
    
    # load the data
    with h5py.File(file, 'r') as f:
        rc = f['rocking_curves'][:]
        
    # define label matrix
    labels = np.ones(rc.shape, dtype=np.uint8)

    #divide in chunks the data
    chunk = 10000
    for start in range(0, rc.shape[0], chunk):
        end = min(start + chunk, rc.shape[0])
        
        rc_chunk = rc[start:end, :]

        # remove noise
        rc_chunk -= np.median(rc_chunk)
        rc_chunk[rc_chunk<0] = 0

        # find minimum
        minima =  (rc_chunk[:,1:-1] < rc_chunk[:,:-2]) & (rc_chunk[:,1:-1] < rc_chunk[:,2:])
        [x,y] =  np.where(minima)

        # modify labels matrix
        for n in range(0,len(x)):
            labels[start+x[n], (y[n]+1):] += 1

    return labels

if __name__ == "__main__":
    labels = label()