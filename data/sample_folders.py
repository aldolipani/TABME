import os
import sys
from glob import glob

import numpy as np

if __name__ == '__main__':
    path_list = sys.argv[1]
    lam = int(sys.argv[2])
    samples = int(sys.argv[3])

    names = []
    for name in glob(f'{path_list}/*'):
        name = name.split('/')[-1]
        names.append(name)

    for i, n in enumerate(np.random.poisson(lam, samples)):
        for name in np.random.choice(names, n):
            print(name, i + 1)
