"""a first attempt at creating a task structure for the
experiments done by Nick Steinmetz"""


from _future_ import division
from pycog import tasktools
import numpy as np
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Sequence-related definitions
#-----------------------------------------------------------------------------------------

# Sequence definitions
sequences = {
    1: [4] + [2,0],
    2: [4] + [0,2],
    3: [4] + [8,6],
    4: [4] + [6,8],
    }
   
# Possible targets from each position
#
#   0 1 2
#   3 4 5
#   6 7 8
#


# Options for Cue
options = {
	1: [0],
	2: [2],
	3: [6],
	4: [8],
}

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 12
N    = 200
Nout = 2

# For addressing inputs
DOTS     = range(9)
SEQUENCE = range(9, 9+nseq)

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant
tau = 50

#-----------------------------------------------------------------------------------------
# Noise
#--------------------------------------------------------------------------------------

var_rec = 0.01**2

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

# Screen coordinates
x0, y0 = -0.5, -0.5
dx, dy = +0.5, +0.5

# Convert dots to screen coordinates
def target_position(k):
    j = 2 - k//3
    i = k % 3

    return x0+i*dx, y0+j*dy

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------------
    print(params)
    if params['name'] in ['gradient', 'test']:
        seq = params.get('seq', rng.choice(sequences.keys()))
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % nseq
        if b == 0:
            generate_trial.seqs = rng.permutation(nseq)

        seq = generate_trial.seqs[b] + 1
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    iti          = 100
    targetsonly  = 400
    targetcue    = 1000
    cuedisappear = 250
    decision     = 800
    T        = iti + targetsonly + targetcue + cuedisappear + decision

    epochs = {
        'iti':         (0, iti),
        'targetsonly': (iti, iti + targetsonly),
        'targetcue':   (iti + targetsonly, iti + targetsonly + targetscue),
        'cuedisappear':(iti + targetsonly + targetscue, iti + targetsonly + targetcue + cuedisappear),
        'decision':    (iti + targetsonly + targetcue + cuedisappear, iti + targetsonly + targetcue + cuedisappear + decision),
        }
    epochs['T'] = T

    #---------------------------------------------------------------------------------
    # Trial info
    #---------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {'seq': seq}

    #---------------------------------------------------------------------------------
    # Inputs
    #---------------------------------------------------------------------------------

    # Input matrix
    X = np.zeros((len(t), Nin)

    # Which sequence?
    X[:,SEQUENCE[seq-1]] = 1

    # Sequence
    sequence = sequences[seq]

    # Possible cue direction
    X[e['targetcue'],sequence[0]] = 1
    for I, J in zip(e['targetcue']],
                    [[sequence[0]] + options[sequence[0]],
                     []]):
        for j in J:
            X[I,j] = 1

    # Inputs
    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.zeros((len(t), Nout)) # Mask matrix

        # Gaze direction
        Y[e['decision'],:] = target_position(sequence[3])
        

        # We don't constrain the intertrial interval
        M[e['iti']+e['targetsonly']+e['targetcue']+e['cuedisappear'],:] = 1

        # Output and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #---------------------------------------------------------------------------------

    return trial

min_error = 0.05

mode         = 'continuous'
n_validation = 100*nseq
}

