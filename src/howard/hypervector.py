from abc import ABC, abstractmethod
import numpy as np

class Hypervector(ABC):
    """
    Abstract hypervector class. Constructor builds a generic hypervector
    of all zeros of the specified dimension.
    """

    # @property
    # @abstractmethod
    # def dimension(self):
    #     pass

    # @property
    # @abstractmethod
    # def samples(self):
    #     pass

    def __init__(self):
        # Constructor - might be implemented in non-abstract subclasses
        pass

    def superimpose(self, vectors):
        """
        Create a superposition of an array of PhasorHV objects and return an object of
        class PhasorHV. Assumes all HV have the same dimension.
        This function should be generic for Boolean and Binary HV as well.
        Normalizes the output according to the angle of the sample sums.
        """
        N = len(vectors)
        D = vectors[0].dimension
        x = np.reshape([v.samples for v in vectors], (N, D))
        superpos = np.sum(x, axis=0)

        if isinstance(vectors[0], PhasorHV):
            result = PhasorHV(dimension=D, samples=superpos)
            result.normalize()
        elif isinstance(vectors[0], BinaryHV):
            processedSuperpos = np.sign(superpos + np.random.rand(D) * 0.1)  # add small random number to change any zeros
            result = BinaryHV(dimension=D, samples=processedSuperpos)

        return result

# Example usage: Subclasses would need to be defined for PhasorHV and BinaryHV.

class PhasorHV(Hypervector):
    """
    Class for circular hypervectors.
    """

    def __init__(self, dimension=1000, samples=None, meanBias=0, standDev=np.pi/2, distribution='uniform'):
        self.dimension = dimension
        if samples is None:
            if distribution == 'uniform':
                phi = 2 * np.pi * np.random.rand(self.dimension) - np.pi  # Uniform distribution [-pi, pi)
            elif distribution == 'normal':
                phi = standDev * np.random.randn(self.dimension) + meanBias  # Normal distribution with mean and std deviation
            self.samples = np.exp(1j * phi)  # Create phasors from angles
        else:
            self.samples = np.array(samples)

    def normalize(self):
        """
        Normalize each sample to unit magnitude, retaining only the angle.
        """
        self.samples = self.samples / np.abs(self.samples)
        return self

    @staticmethod
    def bind(v1, v2):
        """
        Bind two phasor vectors by element-wise multiplication of their samples.
        """
        bound_samples = v1.samples * v2.samples
        return PhasorHV(dimension=v1.dimension, samples=bound_samples)

    @staticmethod
    def unbind(v1, v2):
        """
        Unbind two phasor vectors using the complex conjugate of the second vector.
        """
        bound_samples = v1.samples * np.conj(v2.samples)
        return PhasorHV(dimension=v1.dimension, samples=bound_samples)

    @staticmethod
    def similarity(v1, v2):
        """
        Calculate the similarity as the real part of the normalized dot product of two phasor vectors.
        """
        result = np.real(np.vdot(v1.samples, v2.samples)) / v1.dimension
        return result

    def inverse(self):
        """
        Compute the inverse of a phasor vector by adding pi to each angle.
        """
        inv_samples = self.samples * np.exp(1j * np.pi)
        return PhasorHV(dimension=self.dimension, samples=inv_samples)

    def encode(self, state):
        """
        Encode a value into the hypervector by raising each sample to the power of the state.
        """
        encoded_samples = self.samples ** state
        return PhasorHV(dimension=self.dimension, samples=encoded_samples)

