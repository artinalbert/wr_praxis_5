import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    # Initialize matrix with size n x n
    F = np.zeros((n, n), dtype='complex128')

    # Principal term for DFT matrix
    w = np.exp(-2j * np.pi / n)

    # Fill matrix with values
    for j in range(n):
        for k in range(n):
            F[j, k] = w ** (j * k)

    # Normalize DFT matrix
    F /= np.sqrt(n)

    return F

def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    # Calculate the conjugate transpose of the matrix
    conj_transpose = matrix.conj().T

    # Perform matrix multiplication with its conjugate transpose
    product = np.dot(matrix, conj_transpose)

    # Check if the product is close to the identity matrix
    identity = np.eye(matrix.shape[0])
    return np.allclose(product, identity)


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix
    sigs = []
    fsigs = []

    # Create DFT matrix
    F = np.zeros((n, n), dtype='complex128')
    w = np.exp(-2j * np.pi / n)
    for j in range(n):
        for k in range(n):
            F[j, k] = w ** (j * k)
    F /= np.sqrt(n)

    # Create delta impulse signals and their Fourier transforms
    for i in range(n):
        # Create a delta impulse signal
        signal = np.zeros(n)
        signal[i] = 1
        sigs.append(signal)

        # Perform Fourier transform using the DFT matrix
        f_signal = np.dot(F, signal)
        fsigs.append(f_signal)

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # TODO: implement shuffling by reversing index bits
    n = data.size
    assert n > 0 and ((n & (n - 1)) == 0), "The size of data must be a power of two"

    # Function to reverse the bits of a number up to log2(n)
    def reverse_bits(num, log2_n):
        result = 0
        for _ in range(log2_n):
            result = (result << 1) | (num & 1)
            num >>= 1
        return result

    # Calculate the bit-reversed indices
    log2_n = int(np.log2(n))
    reversed_indices = [reverse_bits(i, log2_n) for i in range(n)]

    # Shuffle the array according to bit-reversed indices
    shuffled_data = np.empty_like(data)
    for i in range(n):
        shuffled_data[i] = data[reversed_indices[i]]

    return shuffled_data



def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # TODO: first step of FFT: shuffle data


    # TODO: second step, recursively merge transforms

    # TODO: normalize fft signal

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency

    # Define the amplitude (volume of the tone)
    A = 1.0

    # Generate time values
    t = np.linspace(0, 1, num_samples, endpoint=True)

    # Generate sine wave
    data = A * np.sin(2 * np.pi * f * t)

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """

    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # TODO: compute Fourier transform of input data

    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.

    # TODO: compute inverse transform and extract real component
  # Compute the Fourier transform of the input data
    adata_ft = np.fft.fft(adata)

    # Translate bandlimit from Hz to data index according to sampling rate and data size
    # The Nyquist limit is half the sampling rate
    nyquist_limit = sampling_rate // 2
    total_samples = adata.size

    # Calculate the frequency for each index in the Fourier transform
    frequencies = np.fft.fftfreq(total_samples, d=1/sampling_rate)

    # Zero out high frequencies above bandlimit
    # Ensure symmetry: if you zero an index i, also zero the index -i
    for i, freq in enumerate(frequencies):
        if abs(freq) > bandlimit:
            adata_ft[i] = 0
            adata_ft[-i] = 0

    # Compute the inverse Fourier transform
    adata_filtered = np.fft.ifft(adata_ft)

    # Extract the real component
    adata_filtered = adata_filtered.real

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
