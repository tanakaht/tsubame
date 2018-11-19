#!/bin/sh
#$ -cwd
#$ -e logs -o logs
#$ -l f_node=1 -l h_rt=0:60:0
#$ -M ken_wakita@me.com
#$ -m beas
#$ -N eigen

# Submit a batch job and exit
if [ -z "$JOB_NAME" -a "$1" == submit ]; then
  qsub -g tga-smartnova $0
  exit 0
fi

# Start of the batch job.  Firstly, arrange the computation environment.

. /etc/profile.d/modules.sh
module load python-extension/2.7

# Start python
python << %_End_of_Python

# import multiprocessing
# import cupy

from __future__ import nested_scopes
from __future__ import generators
from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from __future__ import unicode_literals

import time
import numpy as np
import scipy.linalg

t_start = time.time()
def time():
  return time.time() - t_start

dataset = 'enron'
print('{} - Starting an Eigen batch job for {}'.format(time()))

D = np.load('distance/{}.npy'.format(dataset))
N, _ = D.shape
print('{} - Dimension: {}'.format(N, time())

J = np.eye(N) - np.ones((N, N)) / N # Double centering
B = -J.dot(D * D).dot(J) / 2.0
print('Double centering finished')

n_eigvals = min(N, 500)
eigvals, eigvecs = scipy.linalg.eigh(B, eigvals=(N - n_eigvals, N-1))

print('{} = Saving'.format(time()))
np.save('eigs/{}-eigvals.npy'.format(dataset), eigvals)
np.save('eigs/{}-eigvecs.npy'.format(dataset), eigvecs)
print('{} - Saving done.  Bye'.format(time()))

%_End_of_Python

date "Batch Job ($0) finishing"; date