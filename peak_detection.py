from __future__ import print_function
import numpy as np
import logging
from matplotlib import pyplot as plt


def detect(input, dft_window_length, dft_step, conv_length):
    """ input: normalized audio file array -1.0~1.0
    """
    result = []
    sz = input.size
    spectrum = np.empty([sz // dft_step + 1, dft_window_length],dtype=np.complex64)
    counter = 0
    # compute spectrum
    for i in range(dft_window_length, sz, dft_step):
        frame = input[i : i + dft_window_length]
        frame_trans = np.fft.fft(frame, dft_window_length)
        spectrum[counter,:] = frame_trans
        counter = counter + 1
    spectrum = np.absolute(spectrum)
    # get convolution kernel
    kernel = np.empty(conv_length, dtype=float)
    spectrum_conv = np.empty(spectrum.shape, dtype=float)
    stepvalue = 1.0
    for i in range(conv_length):
        kernel[i] = stepvalue
        stepvalue = stepvalue * 0.95
    for i in range(dft_window_length):
        col = spectrum[:,i]
        col_conv = np.convolve(col,kernel,'same')
        # todo: add mel-scale frequency
        spectrum_conv[:,i] = col_conv
    # differentiate
    spectrum_diff = np.diff(spectrum_conv, axis=0)
    result = np.sum(spectrum_diff,axis=1)
    # dct
    return result, spectrum_conv
def mfcc(filters, fbank):

    pass
    
def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def plot_result(result):
    """ plot the result
    """
    result = result[:,0:2000].T
    rmax = np.amax(result)
    rmin = np.amin(result)
    print(rmax)
    print(rmin)
    plt.imshow(result, cmap='gray',vmin=rmin, vmax=rmax)
    plt.show()


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False
def test_main():
    import soundfile as sf
    data, samplerate = sf.read('i7-965-clipped.wav') 
    result, spec = detect(data,2**10,2**8,2**0)
    #valid_imshow_data(spec)
    plot_result(spec)
    plt.figure(1)
    plt.plot(result)
    plt.show()