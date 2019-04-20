from scipy.fftpack import fft
from scipy.signal import fftconvolve, kaiser
import numpy as np


class ImpulseResponseEstimator(object):
    """
    Gives probe impulse, gets recording and calculate impulse response.
    Method from paper: "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
    Angelo Farina
    """

    def __init__(self, duration=5.0, low=20.0, high=20000.0, fs=44100.0):
        self.fs = fs

        # Total length in samples
        self.T = fs*duration
        self.w1 = low / self.fs * 2*np.pi
        self.w2 = high / self.fs * 2*np.pi

        # Generate test signal
        self.test_signal = self.generate_test_signal()

        # This is what the value of K will be at the end (in dB):
        kend = 10**((-6*np.log2(self.w2/self.w1))/20)
        # dB to rational number.
        k = np.log(kend)/self.T

        # Making reverse probe impulse so that convolution will just
        # calculate dot product. Weighting it with exponent to acheive
        # 6 dB per octave amplitude decrease.
        c = np.array(list(map(lambda t: np.exp(float(t)*k), range(int(self.T)))))
        self.inverse_filter = np.flip(self.test_signal) * c

        # Now we have to normalize energy of result of dot product.
        # This is "naive" method but it just works.
        frp = fft(fftconvolve(self.inverse_filter, self.test_signal))
        self.inverse_filter /= np.abs(frp[round(frp.shape[0]/4)])

    def generate_test_signal(self):
        """Generates test signal."""
        w1 = self.w1
        w2 = self.w2
        T = self.T

        # page 5
        def lin_freq(t):
            return w1*t + (w2-w1)/T * t*t / 2

        # page 6
        def log_freq(t):
            K = T * w1 / np.log(w2/w1)
            L = T / np.log(w2/w1)
            return K * (np.exp(t/L)-1.0)

        freqs = log_freq(range(int(T)))
        impulse = np.sin(freqs)

        # Apply exponential signal on the beginning and the end of the probe signal.
        window_len = 2500
        win = kaiser(window_len, 16)
        win_start = win[:int(win.shape[0]/2)]
        impulse[:win_start.shape[0]] *= win_start
        window_len = 2500
        win = kaiser(window_len, 14)
        win_end = win[int(win.shape[0]/2):]
        impulse[-win_end.shape[0]:] *= win_end

        return impulse

    def estimate(self, recording):
        """Estimates impulse response"""
        ir = fftconvolve(recording, self.inverse_filter, mode='full')
        ir = ir[self.test_signal.shape[0]:self.test_signal.shape[0]*2+1]
        return ir
