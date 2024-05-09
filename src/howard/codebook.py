import numpy as np
from hypervector import PhasorHV

def buildCodebook(numBasis, maxRange, resolution, standDev):
    codebookRange = np.arange(-maxRange, maxRange + resolution, resolution)
    numHV = len(codebookRange)
    codebook = np.empty((numHV, numBasis), dtype=object)
    basesSet = np.empty(numBasis, dtype=object)
    for i in range(numBasis):
        basesSet[i] = PhasorHV(dimension=10000, standDev=standDev, distribution='normal')
    for i in range(numBasis):
        for j in range(numHV):
            codebook[j, i] = basesSet[i].encode(codebookRange[j])
    return codebook, codebookRange, basesSet