from codebook import buildCodebook
from hypervector import PhasorHV
from resnet import ResonatorNetwork
import numpy as np


# Main execution script
numBasis = 3
maxRange = 10
resolution = 0.5
standDev = 100 * np.pi

codebook, codebookRange, basesSet = buildCodebook(numBasis, maxRange, resolution, standDev)

PURPLE = PhasorHV.bind(basesSet[0].encode(6.2), PhasorHV.bind(basesSet[1].encode(-6.2), basesSet[2].encode(5.3)))
BLUE = PhasorHV.bind(basesSet[0].encode(0), PhasorHV.bind(basesSet[1].encode(-10), basesSet[2].encode(5)))
ORANGE = PhasorHV.bind(basesSet[0].encode(6.7), PhasorHV.bind(basesSet[1].encode(5.7), basesSet[2].encode(10)))

X = PhasorHV.bind(PhasorHV.unbind(ORANGE, PURPLE), BLUE)

RN = ResonatorNetwork(codebook, X)
idxFactorEst, iterationNum = RN.findFactors()

print("Estimated Factors Indices:", idxFactorEst)
print("Number of Iterations:", iterationNum)
print(codebook[idxFactorEst])