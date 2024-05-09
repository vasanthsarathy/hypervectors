import numpy as np
from hypervector import PhasorHV

class ResonatorNetwork:
    """
    Resonator network class. Currently works with only three equivalent-size code books.
    """

    def __init__(self, codeBook, boundHV, confidenceThreshold=0.5):
        self.codeBook = codeBook
        self.boundHV = boundHV
        self.confidenceThreshold = confidenceThreshold

    def findFactors(self):
        """
        Identify factors by resonating with the codebook.
        """
        done = False
        iterationNum = 0
        codebook = self.codeBook
        S = self.boundHV
        confidenceThresh = self.confidenceThreshold

        # Initialize factors with random binding of one factor from each book
        x_hat = S.superimpose(codebook[:, 0])
        y_hat = S.superimpose(codebook[:, 1])
        z_hat = S.superimpose(codebook[:, 2])

        # Update function using lambda
        factorUpdate = lambda estimate, weight, dimension: PhasorHV(dimension=dimension, samples=estimate.samples * weight)

        while not done:
            iterationNum += 1

            # Unbind to get temporary factor estimates
            x_hat_temp = S.unbind(S, S.bind(y_hat, z_hat))
            y_hat_temp = S.unbind(S, S.bind(x_hat, z_hat))
            z_hat_temp = S.unbind(S, S.bind(x_hat, y_hat))

            # Project current noisy estimate to full codebook
            x_hat_temp2 = np.array([S.similarity(x, x_hat_temp) for x in codebook[:, 0]])
            y_hat_temp2 = np.array([S.similarity(y, y_hat_temp) for y in codebook[:, 1]])
            z_hat_temp2 = np.array([S.similarity(z, z_hat_temp) for z in codebook[:, 2]])

            x_hat_temp2 = np.abs(x_hat_temp2)
            y_hat_temp2 = np.abs(y_hat_temp2)
            z_hat_temp2 = np.abs(z_hat_temp2)

            # Check confidence levels
            xConfidence = np.max(x_hat_temp2)
            yConfidence = np.max(y_hat_temp2)
            zConfidence = np.max(z_hat_temp2)
            done = (xConfidence > confidenceThresh and yConfidence > confidenceThresh and zConfidence > confidenceThresh)

            # Randomly break ties for the best resonating HV
            idxFactorEst = [np.random.choice(np.where(x_hat_temp2 == xConfidence)[0]),
                            np.random.choice(np.where(y_hat_temp2 == yConfidence)[0]),
                            np.random.choice(np.where(z_hat_temp2 == zConfidence)[0])]

            # Weighted mean of the codebook for the official estimate update
            dims = np.array([len(x.samples) for x in codebook[:, 0]])
            x_hat_temp3 = np.array([factorUpdate(a, b, d) for a, b, d in zip(codebook[:, 0], x_hat_temp2, dims)])
            y_hat_temp3 = np.array([factorUpdate(a, b, d) for a, b, d in zip(codebook[:, 1], y_hat_temp2, dims)])
            z_hat_temp3 = np.array([factorUpdate(a, b, d) for a, b, d in zip(codebook[:, 2], z_hat_temp2, dims)])

            x_hat = S.superimpose(x_hat_temp3)
            y_hat = S.superimpose(y_hat_temp3)
            z_hat = S.superimpose(z_hat_temp3)

        return idxFactorEst, iterationNum
