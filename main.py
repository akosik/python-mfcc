"""
A class for computing MFCC features on 1-d arrays
"""
from math import ceil
import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile

def window(seq, window_size=2, shift_size=1):
    "Sliding window generator w/ opt step size"
    num_windows = ((len(seq)-window_size)//shift_size)+1
    for i in range(0, num_windows*shift_size, shift_size):
        yield seq[i:i+window_size]

class MFCCComputer():
    """
    Computes MFCC features over an array
    """
    def __init__(self, n_mfcc=13, frame_length_ms=25, frame_shift_ms=10, sample_rate=16000,
                 n_fft=512, n_filt=26, fbank_lb=0, fbank_ub=None, channel=0):
        self.workspace = None
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length_ms / 1000 * self.sample_rate)
        self.frame_shift = int(frame_shift_ms / 1000 * self.sample_rate)
        self.n_fft = n_fft
        self.n_filt = n_filt
        self.fbank_lb = fbank_lb
        self.fbank_ub = fbank_ub if fbank_ub else self.sample_rate / 2
        self.channel = channel

    def compute(self, wav):
        "Computes MFCC feats on signal w/ specified class params"
        wav = wav[:, self.channel].astype(np.float64)
        self.workspace = self._pad_signal(wav)
        self.workspace = self._frame(self.workspace)
        self.workspace = self._calc_periodogram(self.workspace)
        self.workspace = self._apply_mel_filterbanks(self.workspace)
        self.workspace = np.log(np.where(self.workspace == 0, np.finfo(float).eps,
                                         self.workspace))
        self.workspace = dct(self.workspace, norm='ortho')
        return self.workspace[:, :self.n_mfcc]

    def _pad_signal(self, signal):
        n_frames = 1 + int(ceil((len(signal) - self.frame_length) / self.frame_shift))
        w_last_frame = int((n_frames - 1) * self.frame_shift + self.frame_length)
        padding = np.zeros((w_last_frame - len(signal),))
        return np.concatenate((signal, padding))

    def _frame(self, signal):
        assert (len(signal) - self.frame_length) % self.frame_shift == 0, \
            f"Padding is wrong {len(signal) % self.frame_shift}"
        return np.array(list(window(signal, self.frame_length, self.frame_shift)))

    def _calc_periodogram(self, framed_signal):
        # multiply by window function, here: hamming window
        framed_signal *= np.tile(self._get_hamming_window(self.frame_length),
                                 (framed_signal.shape[0], 1))
        return self._compute_power_spectrum(framed_signal)

    def _compute_power_spectrum(self, framed_signal):
        magnitude_spectrum = np.absolute(np.fft.rfft(framed_signal, self.n_fft))
        return 1.0 / self.n_fft * np.square(magnitude_spectrum)

    def _apply_mel_filterbanks(self, framed_signal):
        return np.dot(framed_signal, self._calc_mel_fbanks().T)

    def _calc_mel_fbanks(self):
        low = self._hertz2mel(self.fbank_lb)
        high = self._hertz2mel(self.fbank_ub)
        # fbanks are triangular filters linearly spaced in mels
        # get points defining each triangular filter
        filter_peaks = np.linspace(low, high, self.n_filt+2)

        # get fft bin corresponding to mel
        bins = np.floor((self.n_fft+1) * self._mel2hertz(filter_peaks) / self.sample_rate)

        fbank = np.zeros([self.n_filt, (self.n_fft // 2) + 1])

        # first filter starts at i=0, peaks at i=1, closes at i=2
        # second filter starts at i=1, peaks at i=2, closes at i=3
        # etc.
        for i in range(self.n_filt):
            # left half of filter
            for j in range(int(bins[i]), int(bins[i+1])):
                fbank[i, j] = (j - bins[i]) / (bins[i+1]-bins[i])
            # right half of filter
            for j in range(int(bins[i+1]), int(bins[i+2])):
                fbank[i, j] = (bins[i+2]-j) / (bins[i+2]-bins[i+1])
        return fbank

    @staticmethod
    def _get_hamming_window(n):
        return np.array([0.54 - 0.46*np.cos(2*np.pi*i/(n-1)) for i in range(n)])

    @staticmethod
    def _hertz2mel(hz):
        return 1125.0 * np.log(1+hz/700.)

    @staticmethod
    def _mel2hertz(mel):
        return 700*(np.exp(mel/1125.0)-1)


if __name__ == "__main__":
    c = MFCCComputer()
    _, data = wavfile.read('./test.wav')
    mfcc = c.compute(data)
    print(mfcc)
