from scipy import signal, fftpack
import peakutils, datetime
import matplotlib.pyplot as plt
import numpy as np
# from numba import jit


class Filtering:

    def iirfilt(self, bandtype, data, Fs, Wp, Ws=[], order=3, analog_val=False, automatic=1, Rp=3, As=60, filttype='butter',
                showresponse=0):
        '''Designs butterworth filter:
        Data is the data that you want filtered
        Fs is the sampling frequency (in Hz)
        Ws and Wp are stop and pass frequencies respectively (in Hz)

        Passband (Wp) : This is the frequency range which we desire to let the signal through with minimal attenuation.
        Stopband (Ws) : This is the frequency range which the signal should be attenuated.

        Digital: Ws is the normalized stop frequency where 1 is the nyquist freq (pi radians/sample in digital)
                 Wp is the normalized pass frequency

        Analog: Ws is the stop frequency in (rads/sec)
                Wp is the pass frequency in (rads/sec)

        Analog is false as default, automatic being one has Python select the order for you. pass_atten is the minimal attenuation
        the pass band, stop_atten is the minimal attenuation in the stop band. Fs is the sample frequency of the signal in Hz.

        Rp = 0.1      # passband maximum loss (gpass)
        As = 60 stoppand min attenuation (gstop)


        filttype : str, optional
            The type of IIR filter to design:
            Butterworth : ‘butter’
            Chebyshev I : ‘cheby1’
            Chebyshev II : ‘cheby2’
            Cauer/elliptic: ‘ellip’
            Bessel/Thomson: ‘bessel’

        bandtype : {‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’}, optional
        '''

        cutoff = Wp
        if Ws != []:
            cutoff2 = Ws

        stop_amp = 1.5
        stop_amp2 = 1.4

        if not analog_val:  # need to convert the Ws and Wp to have the units of pi radians/sample
            # this is for digital filters
            if bandtype in ['low', 'high']:
                Wp = Wp / (Fs / 2)  # converting to fraction of nyquist frequency

                Ws = Wp * stop_amp

            elif bandtype == 'band':
                Wp = Wp / (Fs / 2)  # converting to fraction of nyquist frequency
                Wp2 = Wp / stop_amp2

                Ws = Ws / (Fs / 2)  # converting to fraction of nyquist frequency
                Ws2 = Ws * stop_amp2

        else:  # need to convert the Ws and Wp to have the units of radians/sec
            # this is for analog filters
            if bandtype in ['low', 'high']:
                Wp = 2 * np.pi * Wp

                Ws = Wp * stop_amp

            elif bandtype == 'band':
                Wp = 2 * np.pi * Wp
                Wp2 = Wp / stop_amp2

                Ws = 2 * np.pi * Ws
                Ws2 = Ws * stop_amp2

        if automatic == 1:
            if bandtype in ['low', 'high']:
                b, a = signal.iirdesign(wp=Wp, ws=Ws, gpass=Rp, gstop=As, analog=analog_val, ftype=filttype)
            elif bandtype == 'band':
                b, a = signal.iirdesign(wp=[Wp, Ws], ws=[Wp2, Ws2], gpass=Rp, gstop=As, analog=analog_val, ftype=filttype)
        else:
            if bandtype in ['low', 'high']:
                if filttype == 'cheby1' or 'cheby2' or 'ellip':
                    b, a = signal.iirfilter(order, Wp, rp=Rp, rs=As, btype=bandtype, analog=analog_val, ftype=filttype)
                else:
                    b, a = signal.iirfilter(order, Wp, btype=bandtype, analog=analog_val, ftype=filttype)
            elif bandtype == 'band':
                if filttype == 'cheby1' or 'cheby2' or 'ellip':
                    b, a = signal.iirfilter(order, [Wp, Ws], rp=Rp, rs=As, btype=bandtype, analog=analog_val,
                                            ftype=filttype)
                else:
                    b, a = signal.iirfilter(order, [Wp, Ws], btype=bandtype, analog=analog_val, ftype=filttype)

        if data != []:
            if len(data.shape) > 1:
                #print('Filtering multidimensional array!')
                filtered_data = np.zeros((data.shape[0], data.shape[1]))
                filtered_data = signal.filtfilt(b, a, data, axis=1)
                #for channel_num in range(0, data.shape[0]):
                #    # filtered_data[channel_num,:] = signal.lfilter(b, a, data[channel_num,:])
                #    filtered_data[channel_num, :] = signal.filtfilt(b, a, data[channel_num, :])
            else:
                # filtered_data = signal.lfilter(b, a, data)
                filtered_data = signal.filtfilt(b, a, data)

        if showresponse == 1:  # set to 1 if you want to visualize the frequency response of the filter
            if filttype == 'butter':
                FType = 'Butterworth'
            elif filttype == 'cheby1':
                FType = 'Chebyshev I'
            elif filttype == 'cheby2':
                FType = 'Chebyshev II'
            elif filttype == 'ellip':
                FType = 'Cauer/Elliptic'
            elif filttype == 'bessel':
                FType = 'Bessel/Thomson'

            if analog_val:
                mode = 'Analog'
            else:
                mode = 'Digital'

            if not analog_val:
                w, h = signal.freqz(b, a, worN=8000)  # returns the requency response h, and the normalized angular
                # frequencies w in radians/sample
                # w (radians/sample) * Fs (samples/sec) * (1 cycle/2pi*radians) = Hz
                f = Fs * w / (2 * np.pi)  # Hz
            else:
                w, h = signal.freqs(b, a, worN=8000)  # returns the requency response h,
                # and the angular frequencies w in radians/sec
                # w (radians/sec) * (1 cycle/2pi*radians) = Hz
                f = w / (2 * np.pi)  # Hz

            plt.figure(figsize=(10, 5))
            #plt.subplot(211)
            plt.semilogx(f, np.abs(h), 'b')
            plt.xscale('log')

            if 'cutoff2' in locals():
                plt.title('%s Bandpass Filter Frequency Response (Order = %s, Wp=%s (Hz), Ws =%s (Hz))'
                          % (FType, order, cutoff, cutoff2))
            else:
                plt.title('%s Lowpass Filter Frequency Response (Order = %s, Wp=%s (Hz))'
                          % (FType, order, cutoff))

            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Gain [V/V]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(cutoff, color='green')
            if 'cutoff2' in locals():
                plt.axvline(cutoff2, color='green')
                # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko') # cutoff frequency
            plt.show()
        if data != []:
            return filtered_data

    def notch_filt(self, data, Fs, band=10, freq=60, ripple=1, order=2, filter_type='butter', analog_filt=False, showresponse=0):
        '''# Required input defintions are as follows;
        # time:   Time between samples
        # band:   The bandwidth around the centerline freqency that you wish to filter
        # freq:   The centerline frequency to be filtered
        # ripple: The maximum passband ripple that is allowed in db
        # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
        #         IIR filters are best suited for high values of order.  This algorithm
        #         is hard coded to FIR filters
        # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
        # data:         the data to be filtered'''

        cutoff = freq
        nyq = Fs / 2.0
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = signal.iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                                analog=analog_filt, ftype=filter_type)
        if data != []:
            if len(data.shape) > 1:  # lfilter is one dimensional so we need to perform for loop on multi-dimensional array
                filtered_data = np.zeros((data.shape[0], data.shape[1]))
                filtered_data = signal.filtfilt(b, a, data, axis=1)
                #for channel_num in range(0, data.shape[0]):
                    # filtered_data[channel_num,:] = signal.lfilter(b, a, data[channel_num,:])
                 #   filtered_data[channel_num, :] = signal.filtfilt(b, a, data[channel_num, :])
            else:
                # filtered_data = signal.lfilter(b, a, data)
                filtered_data = signal.filtfilt(b, a, data)

        if showresponse == 1:
            if filter_type == 'butter':
                FType = 'Butterworth'
            elif filter_type == 'cheby1':
                FType = 'Chebyshev I'
            elif filter_type == 'cheby2':
                FType = 'Chebyshev II'
            elif filter_type == 'ellip':
                FType = 'Cauer/Elliptic'
            elif filter_type == 'bessel':
                FType = 'Bessel/Thomson'

            if analog_filt == 1:
                mode = 'Analog'
            else:
                mode = 'Digital'

            if analog_filt == False:
                w, h = signal.freqz(b, a, worN=8000)  # returns the requency response h, and the normalized angular
                # frequencies w in radians/sample
                # w (radians/sample) * Fs (samples/sec) * (1 cycle/2pi*radians) = Hz
                f = Fs * w / (2 * np.pi)  # Hz
            else:
                w, h = signal.freqs(b, a, worN=8000)  # returns the requency response h, and the angular frequencies
                # w in radians/sec
                # w (radians/sec) * (1 cycle/2pi*radians) = Hz
                f = w / (2 * np.pi)  # Hz

            plt.figure(figsize=(20, 15))
            plt.subplot(211)
            plt.semilogx(f, np.abs(h), 'b')
            plt.xscale('log')
            plt.title('%s Filter Frequency Response (%s)' % (FType, mode))
            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Gain [V/V]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(cutoff, color='green')

        return filtered_data

    def dcblock(self, data, fc, fs=None, analog_val=False, showresponse=0):

        """This method will return the filter coefficients for a DC Blocker Filter"""

        if fs is None:
            Fc = fc

        else:
            Fc = 2 * fc / fs

        p = (np.sqrt(3) - 2 * np.sin(np.pi * Fc)) / (np.sin(np.pi * Fc) +
                                                     np.sqrt(3) * np.cos(np.pi * Fc));

        b = np.array([1, -1])
        a = np.array([1, -p])

        if len(data) != 0:
            if len(data.shape) > 1:

                # filtered_data = np.zeros((data.shape[0], data.shape[1]))
                filtered_data = signal.filtfilt(b, a, data, axis=1)

            else:
                # filtered_data = signal.lfilter(b, a, data)
                filtered_data = signal.filtfilt(b, a, data)

        if showresponse == 1:  # set to 1 if you want to visualize the frequency response of the filter
            self.PlotResponse(a, b, fc, fs, analog_val)

        if len(data) != 0:
            return filtered_data

    def PlotResponse(self, a, b, fc, fs, analog_val):

        if analog_val:
            mode = 'Analog'
        else:
            mode = 'Digital'

        if not analog_val:
            w, h = signal.freqz(b, a, worN=8000)  # returns the requency response h, and the normalized angular
            # frequencies w in radians/sample
            # w (radians/sample) * Fs (samples/sec) * (1 cycle/2pi*radians) = Hz
            f = fs * w / (2 * np.pi)  # Hz
        else:
            w, h = signal.freqs(b, a, worN=8000)  # returns the requency response h,
            # and the angular frequencies w in radians/sec
            # w (radians/sec) * (1 cycle/2pi*radians) = Hz
            f = w / (2 * np.pi)  # Hz

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        # plt.subplot(211)
        # ax.plot(f, np.abs(h), 'b')
        # ax.set_xlim([0,1])
        ax.semilogx(f, np.abs(h), 'b')
        ax.set_xscale('log')
        # ax.set_xlim([-1000, ax.get_xlim()[1]])

        plt.title("DC Block Filter")

        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel('Gain [V/V]')
        ax.margins(0, 0.1)
        ax.grid(which='both', axis='both')
        ax.axvline(fc, color='green')
        plt.show()


class Thresholding():
    #@jit
    def amp_threshold(self, active_chans, data, Fs, perc=0.1, min_dist_val=10):
        '''Finds the spikes in the data using a thresholding technique.

        -perc: percent of maximum value to consider as threshold, takes on values from 0-1, 0 = 0% and 1 = 100% of max value

        -min_dist_val - Minimum distance between each detected peak. The peak with the highest amplitude
        is preferred to satisfy this constraint.

        if data is a multi-dimensional array the result will be a dictionary where the keys are the channel numbers from 0 to N,
        and the items are the spike times (spike_times) or the spike indices (spike_index)
        '''


        if len(data.shape) > 1:
            n_samples = data.shape[1]
        else:
            n_samples = len(data)

        total_time = total_time = n_samples / Fs
        time = np.arange(0, total_time, total_time/n_samples)

        if len(data.shape) > 1:  # multidimensional array
            spike_index = {}  # creating a dictionary to hold all the indices for the channels
            spike_times = {}
            for channel_num in range(0, data.shape[0]):
                spike_index[active_chans[channel_num]] = peakutils.peak.indexes(data[channel_num, :], thres=perc,
                                                                  min_dist=min_dist_val)
                spike_times[active_chans[channel_num]] = time[spike_index[active_chans[channel_num]]]
        else:
            spike_index = peakutils.peak.indexes(data, thres=perc, min_dist=min_dist_val)
            spike_times = time[spike_index]

        return spike_index, spike_times

    #@jit
    def auto_threshold(self, active_chans, data, Fs, multiple=4, min_dist_val=10):
        '''Auto thresholding technique incorporated by:
        Quian Quiroga in 2014 - Unsupervised Spike Detection and Sorting with Wavelets and
        Superparamagnetic Clustering

        Thr = 4*sigma, sigma = median(abs(x)/0.6745)

        if user thinks 4 times the standard deviation (sigma) is not enough they can change the multiple variable
        '''

        import numpy as np

        if len(data.shape) > 1:
            n_samples = data.shape[1]
        else:
            n_samples = len(data)

        total_time = total_time = n_samples / Fs
        time = np.arange(0, total_time, total_time / n_samples)

        if len(data.shape) > 1:  # multidimensional array
            sigma_n = multiple * np.median(np.abs(data) / 0.6745, axis=1)  # calculates the median of the rows
            # print(sigma_n.shape)  # debugging purposes
            max_vals = np.ndarray.max(data, axis=1)

            spike_index = {}  # creating a dictionary to hold all the indices for the channels
            spike_times = {}
            for channel_num in range(0, data.shape[0]):
                # converting the standard deviation value to a threshold percentage

                if int(max_vals[channel_num]) == 0:
                    pass

                else:
                    perc = sigma_n[channel_num] / max_vals[channel_num]

                if perc > 1:
                    #  this means the threshold is set higher than the maximum value
                    print('[%s %s]: The threshold value for channel: %s is larger than the max value, setting threshold to %s percent of the max!'
                          % (str(datetime.datetime.now().date()), str(datetime.datetime.now().time())[:8], channel_num, self.settings['ThreshPerc']))

                    perc = (float(self.settings['ThreshPerc'])/100)*max_vals[channel_num]

                else:
                    pass

                spike_index[active_chans[channel_num]] = peakutils.peak.indexes(data[channel_num, :], thres=perc,
                                                                  min_dist=min_dist_val)
                spike_times[active_chans[channel_num]] = time[spike_index[active_chans[channel_num]]]
        else:  # for arrays
            sigma_n = multiple * np.median(np.abs(data) / 0.6745)  # calculates the median of the rows
            # print(sigma_n.shape)  # debugging purposes
            max_vals = np.amax(data)
            perc = sigma_n / max_vals
            spike_index = peakutils.peak.indexes(data, thres=perc, min_dist=min_dist_val)
            spike_times = time[spike_index]

        return spike_index, spike_times


class Plotting():

    def fft_plot(self, Fs, Y):
        '''Takes the Sample Frequency: Fs(Hz), the numer of samples, N, and the data values (Y),
        and performs a Fast Fourier Transformation to observe the signal in the frequency domain'''

        N = len(Y)
        k = np.arange(N)

        T = N / Fs

        frq = k / T  # two sides frequency range
        frq = frq[range(np.int(N / 2))]  # one side frequency range

        FFT = fftpack.fft(Y)  # fft computing and normalization
        fig, ax = plt.subplots(1)
        ax.plot(frq, 2.0 / N * np.abs(FFT[0:int(N / 2)]), 'b')

        ax.set_title('FFT Plot')
        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel('Amplitude')
        ax.margins(0, 0.1)
        ax.grid()

        return fig, ax


def FastFourier(Fs, Y):
    '''Takes the Sample Frequency: Fs(Hz), the numer of samples, N, and the data values (Y),
    and performs a Fast Fourier Transformation to observe the signal in the frequency domain'''

    N = len(Y)
    k = np.arange(N)

    T = N / Fs

    frq = k / T  # two sides frequency range
    frq = frq[range(np.int(N / 2))]  # one side frequency range

    FFT = fftpack.fft(Y)  # fft computing and normalization

    FFT_norm = 2.0 / N * np.abs(FFT[0:int(N / 2)])

    return frq, FFT_norm