from typing import Union, Dict

import abc
import numpy as np
import scipy.signal as scisig
import scipy.interpolate as interp


def design_kaiser_lowpass(
        sampling_freq: float,
        stop_db: float,
        cutoff_freq: float,
        width: float,
) -> (np.ndarray, np.ndarray):
    """
    Design a low-pass filter using the Kaiser window method.

    :param sampling_freq: (float) The original sampling frequency.
    :param stop_db: (float) The required DB level at the stop-band.
    :param cutoff_freq: (float) The required cutoff frequency.
    :param width: (float) The width of the stop-band.

    :return: (np.ndarray, np.ndarray) The taps and beta parameters for
    SciPy's Kaiser filter.
    """

    stop_db = np.abs(stop_db)

    # Convert to normalized frequencies
    nyq = 0.5 * sampling_freq
    cutoff = cutoff_freq / nyq
    width = width / nyq

    # Design the parameters for the Kaiser window FIR filter.
    N, beta = scisig.kaiserord(stop_db, width)
    N |= 1  # Ensure a Type I FIR filter.

    taps = scisig.firwin(N, cutoff, window=('kaiser', beta), scale=False)

    return taps, beta


class ScalarNormalizer:
    """
    Utility class for performing scalar normalization
    """

    def __init__(self, scale: Union[float, int]):
        self.scale = scale

    def __call__(self, value: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        return self.scale * value


class PhysioProcessor(abc.ABC):
    """
    A general purpose processor for most of PhysioNet databases.
    """

    def __init__(
            self,
            output_signal: str = 'ECG',
            trim_n_seconds: int = 0,
            detrend: bool = True,
            scale: str = None,
            downsample_params: dict = None,
            align_ecg: bool = False,
            n_beats_in_aligned_grid: int = 60,
            n_points_per_beat_in_grid: int = 81,
            overlapping_beats: int = 0,
            estimate_spectrum: bool = False,
    ):
        """
        Constructor for the 'PhysioProcessor', where it is possible to specify the
        required output signals modalities, pre-processing operations to be applied
        and final outputs structure.

        :param output_signal: (str) Either 'ECG' or 'RR'. Specifies whether the
        outputted signal should be in the form of ECG or RR-intervals.
        :param trim_n_seconds: (int) Number of seconds to trim for the beginning and
        ending of recordings, in order to discard the most commonly noisy segments in
        recordings.
        :param detrend: (bool) Whether to detrend the signal, detrending is done by
        removing a fitted 2nd order polynomial to the entire signal.
        :param scale: (str) Specify how to scale the signal. Possible options are:
            'z': Z normalization (normalization to mean 0 and std 1)
            'min_max': Normalize the minimal value to be 0 and the maximal value to be 1
            'tanh': Apply the modified TanH normalization (for further details see
            for example: https://alfurka.github.io/2018-11-10-preprocessing-for-nn/)
        If left as None doesn't apply any scaling.
        :param downsample_params: (dict) If specified then downsamples the original
        raw signal to the specified frequency using the given parameters
        and a lowpass Kaiser window. If left as None, doesn't perform downsampling.
        If not None, the downsample_params should be of the following structure:
        {
        'down_frequency': (float)  The frequency to downsample to, in Hz.
        'width': (float) width of the stop-band, in Hz.
        'ripple_db': (float) The desired attenuation in the stop band, in dB.
        }
        :param align_ecg: (bool) Whether to align the ECG samples around each R peak
        and produced aligned outputs. If set to True then seperates the ECG signal
        to separated, aligned beats
        :param n_beats_in_aligned_grid: (int) Used if 'align_ecg' is set to True.
        Specifies the number of beats in each sample.
        :param n_points_per_beat_in_grid: (int) Used if 'align_ecg' is set to True.
        Specifies the number points to resample each aligned beat.
        :param overlapping_beats: (int) The number of overlapping beats between
         consecutive samples.
        :param estimate_spectrum: (bool) Whether to also produce spectral estimation
        for each temporal signal. If 'align_ecg' is set to True, the spectral
        estimations are done on a beat resolution.
        """

        assert output_signal in ("ECG", "RR", "HRV_dep"), \
            f"'output_signal' cannot be {output_signal}, must be either 'ECG' " \
            f"for raw ecg signal, or 'RR' for beat intervals signal"

        assert scale in (None, 'z', 'min_max', 'tanh'), \
            f"{scale} is not a supported scaling type. " \
            f"The currently supported types are:" \
            f"\n'z': Z Normalization,\n'scale': Scale Normalization," \
            f"\n'tanh': Modified TanH normalization,"

        if downsample_params is not None:
            valid_keys = ["down_frequency", "width", "ripple_db"]
            assert (sorted(valid_keys) ==
                    sorted(list(downsample_params.keys()))), \
                f"'downsample_params' has invalid keys: " \
                f"{[k for k in downsample_params.keys() if k not in valid_keys]}."

        # Setup
        self._output_signal = output_signal
        self._n = trim_n_seconds
        self._de_trend = detrend
        self._scale = scale
        self._downsample_params = downsample_params
        self._align_ecg = align_ecg
        self._n_beats_in_aligned_grid = n_beats_in_aligned_grid
        self._n_points_per_beat_in_grid = n_points_per_beat_in_grid
        self._overlapping_beats = overlapping_beats
        self._spectrum = estimate_spectrum

        # For placeholder for the computed scaling parameters (if required)
        self.scaling_parameters = None

    @staticmethod
    def _downsample_signal(
            taps: np.ndarray,
            x_axis: np.ndarray,
            new_x_axis: np.ndarray,
            signal: np.ndarray,
    ) -> np.ndarray:
        """
        A utility method for downsampling a signal using low-pass filtering and Cubic
        Spline interpolatoin

        :param taps: (np.ndarray) The downsampling filter's denominator
        :param x_axis: (np.ndarray) The x-axis over which to downsample.
        :param new_x_axis: (np.ndarray) The x-axis on which to downsample.
        :param signal: (np.ndarray) The signal to downsample.

        :return: (np.ndarray) The downsampled signal.
        """

        # Perform a low-pass filtering of the down-sampled signal
        ds_sig = scisig.filtfilt(b=taps, a=1.0, x=signal, axis=0)

        # Down-sample in time domain using interpolation
        cs_interp = interp.CubicSpline(x=x_axis, y=ds_sig, axis=0, bc_type='not-a-knot',
                                       extrapolate=False)
        ds_sig = cs_interp(new_x_axis)

        return ds_sig

    @staticmethod
    def _z_norm(signal: np.ndarray) -> (np.ndarray, dict):
        """
        A utility method for performing the Z-normalization.

        :param signal: (np.ndarray) The signal to normalize.

        :return: (np.ndarray, dict) The normalized signal, and the normalization
        parameters, i.e. the original mean and std , keyed as
        'mu' and 'sigma'.
        """

        mu = np.mean(signal, axis=0, keepdims=True)
        sigma = np.std(signal, axis=0, keepdims=True)

        norm_signal = (signal - mu) / sigma

        scaling_parameters = {
            'mu': mu,
            'sigma': sigma,
        }

        return norm_signal, scaling_parameters

    @staticmethod
    def _min_max_norm(signal: np.ndarray) -> (np.ndarray, dict):
        """
        A utility method for performing the min-max normalizatoin.

        :param signal: (np.ndarray) The signal to normalize.

        :return: (np.ndarray, dict) The normalized signal, and the normalization
        parameters, i.e. the original minimal and maximal values, keyed as
        'min' and 'max'.
        """

        min_val = np.min(signal, axis=0, keepdims=True)
        norm_signal = signal - min_val

        max_val = np.max(norm_signal, axis=0, keepdims=True)
        norm_signal = norm_signal / max_val

        scaling_parameters = {
            'min': min_val,
            'max': max_val,
        }

        return norm_signal, scaling_parameters

    @staticmethod
    def _tanh_norm(signal: np.ndarray) -> (np.ndarray, dict):
        """
        A utility method for performing the modified TanH normalizatoin.

        :param signal: (np.ndarray) The signal to normalize.

        :return: (np.ndarray, dict) The normalized signal, and the normalization
        parameters, i.e. the original mean and std , keyed as
        'mu' and 'sigma'.
        """

        mu = np.mean(signal, axis=0, keepdims=True)
        sigma = np.std(signal, axis=0, keepdims=True)

        norm_signal = (0.01 * ((signal - mu) / sigma))
        norm_signal = 0.5 * (np.tanh(norm_signal) + 1)

        scaling_parameters = {
            'mu': mu,
            'sigma': sigma,
        }

        return norm_signal, scaling_parameters

    @staticmethod
    def _detrend(signal: np.ndarray) -> np.ndarray:
        """
        A utility method for detrending the signal

        :param signal: (np.ndarray) The signal to detrend.

        :return: (np.ndarray) The de-trended signal
        """

        detrended_signal = scisig.detrend(signal, axis=0, type='linear')

        return detrended_signal

    @staticmethod
    def _trim_n_seconds(
            signal: np.ndarray = None,
            frequency: float = 1.0,
            n: int = 0,
            qrs: np.ndarray = None,
            rhythms: np.ndarray = None,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        A utility method for removing n seconds from the beginning and ending of
         a signal. If also given QRS and rhythms annotations, re-align the annotaions
         to the trimmed signal.

        :param signal: (np.ndarray) The signal to trim
        :param frequency: (float) The signal's frequency.
        :param n: (int) How many seconds to trim
        :param qrs: Optional (np.ndarray) QRS annotations
        :param rhythms: Optional (np.ndarray) Rhythm annotations.

        :return: (np.ndarray, np.ndarray, np.ndarray) The trimmed signal,
        the re-aligned QRS annotations and the re-aligned rhythms annotations.
        """

        assert (signal is not None or qrs is not None), \
            "At least one of 'signal' or 'qrs' " \
            "must be a np.ndarray and not a None object."

        trim_n_seconds = int(n)
        if trim_n_seconds > 0:
            k = frequency * trim_n_seconds

            if signal is not None:
                original_length = signal.shape[0]
                signal = signal[k:-k]

            else:
                original_length = qrs[-1]

            if qrs is not None:
                start_inds = np.where(qrs > k)[0]
                end_inds = np.where(
                    qrs > (original_length - k)
                )[0]

                if len(start_inds) and len(end_inds):
                    qrs = qrs[start_inds[0]:end_inds[0]] - k

                elif len(start_inds):
                    qrs = qrs[(start_inds[0]):] - k

                elif len(end_inds):
                    qrs = qrs[:end_inds[0]]

            else:
                qrs = np.array([])

            if rhythms is not None and len(qrs):
                if len(start_inds) and len(end_inds):
                    rhythms = rhythms[start_inds[0]:end_inds[0]]

                elif len(start_inds):
                    rhythms = rhythms[start_inds[0]:]

                elif len(end_inds):
                    rhythms = rhythms[:end_inds[0]]

            elif rhythms is not None:
                time = np.arange(original_length)
                start_inds = np.where((time / frequency) > (k / frequency))[0]
                end_inds = np.where((time / frequency) >
                                    ((original_length - k) / frequency))[0]

                if len(start_inds) and len(end_inds):
                    rhythms = rhythms[start_inds[0]:end_inds[0]]

                elif len(start_inds):
                    rhythms = rhythms[start_inds[0]:]

                elif len(end_inds):
                    rhythms = rhythms[:end_inds[0]]

            else:
                rhythms = np.array([])

        return signal, qrs, rhythms

    @staticmethod
    def _estimate_spectrum(signal: np.ndarray, frequency: float = 1.0,
                           aligned: bool = True) -> np.ndarray:
        """
        A utility method for estimating a signal's spectrum using Welch's Power
        spectrum estimator.

        :param signal: (np.ndarray) The signal of which to estimate the spectrum.
        :param frequency: (float) The signal sampling frequency.
        :param aligned: (bool) Whether the signal is aligned on a beat-by-beat level,
        hence the spectrum should also be estimated separately per each beat.

        :return: (np.ndarray) The estimated power spectrum.
        """

        if aligned:
            pxx = np.concatenate([
                np.expand_dims(fftshift(welch(signal[window], fs=frequency,
                                              nperseg=signal.shape[0],
                                              return_onesided=False,
                                              scaling='spectrum',
                                              axis=0)[1], axes=0), 0)
                for window in range(signal.shape[0])
            ], 0)

        else:
            pxx = fftshift(welch(signal, fs=frequency, nperseg=signal.shape[0],
                                 return_onesided=False,
                                 scaling='spectrum', axis=0)[1], axes=0)

        return pxx

    def _realign_beats(
            self,
            record: np.ndarray,
            qrs: np.ndarray,
            labels: np.ndarray = None,
    ) -> (np.ndarray, Union[np.ndarray, None]):
        """
        A utility method for realigning beats according to their R-Peaks.

        :param record: (np.ndarray) The recording to align.
        :param qrs: (np.ndarray) QRS annotation to align the beats by.
        :param labels: (np.ndarray) Rhythm annotations to align according to the beats.

        :return: (np.ndarray) The aligned beats.
        """

        assert qrs is not None, "Cannot use the ECG beat alignment without " \
                                "supplying a QRS vector."

        assert self._overlapping_beats < self._n_beats_in_aligned_grid, \
            f"Cannot have {self._overlapping_beats} overlapping beats in a grid " \
            f"with a total of {self._n_beats_in_aligned_grid} beats."

        half_len = self._n_beats_in_aligned_grid // 2

        # Handle the first & last beats in case they are too close to the
        # start / end of the recording
        misplaced_beats_start = np.where(qrs <= half_len)[0]
        if len(misplaced_beats_start):
            qrs = qrs[(misplaced_beats_start[-1] + 1):]

        misplaced_beats_end = np.where(qrs >= (record.shape[0] - half_len))[0]
        if len(misplaced_beats_end):
            qrs = qrs[:misplaced_beats_end[0]]

        # Compute the number of aligned windows in the record
        if self._overlapping_beats:
            n_windows = ((qrs.shape[0] - self._n_beats_in_aligned_grid) //
                         (self._n_beats_in_aligned_grid - self._overlapping_beats))

        else:
            n_windows = qrs.shape[0] // self._n_beats_in_aligned_grid

        # Placeholders to fill in the aligned beats windows
        # and their respective labels
        if len(record.shape) > 1:
            n_leads = record.shape[1]

        else:
            n_leads = 1

        aligned_record = np.zeros((n_windows, self._n_beats_in_aligned_grid,
                                   self._n_points_per_beat_in_grid, n_leads))
        if labels is not None:
            aligned_labels = np.zeros((n_windows, self._n_beats_in_aligned_grid))

        else:
            aligned_labels = None

        # Re-sample & align beats
        for window in range(n_windows):
            beats_start_ind = window * (self._n_beats_in_aligned_grid -
                                        self._overlapping_beats)
            beats_end_ind = beats_start_ind + self._n_beats_in_aligned_grid

            beats_inds = qrs[beats_start_ind:beats_end_ind]
            beats = np.concatenate(
                [
                    np.expand_dims(record[(beat - half_len):(beat + half_len + 1)], 0)
                    for beat in beats_inds
                ],
                0)

            # Sample labels as well if applicable
            if labels is not None:
                label = labels[beats_start_ind:beats_end_ind]
                aligned_labels[window] = label

            aligned_record[window] = beats

        return aligned_record, aligned_labels

    def _downsample(
            self,
            record: np.ndarray,
            qrs: np.ndarray,
            labels: np.ndarray = None,
            frequency: float = 1.0,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Utility method which defines the downsampling filter and performs the
        downsampling process via the _downsample_signal method.

        :param record: (np.ndarray) The record to downsample.
        :param qrs: (np.ndarray) QRS annotation to align after downsampling.
        :param labels: (np.ndarray) Rhythm annotation to align after downsampling.
        :param frequency: (float) The sampling frequency of the recording.

        :return: (np.ndarray, np.ndarray, np.ndarray) The downsampled signal,
        the aligned QRS annotations, the aligned rhythm annotations.
        """

        assert frequency > self._downsample_params['down_frequency'], \
            f"Cannot downsample from {frequency} to " \
            f"{self._downsample_params['down_frequency']}. " \
            f"The downsampled frequency must be < than the original frequency."

        # Generate the original & downsampled x-axes
        n_points = record.shape[0]
        downsampling_ratio = frequency // self._downsample_params['down_frequency']
        downsampled_points = n_points // downsampling_ratio
        x_axis = np.arange(n_points)
        new_x_axis = np.linspace(start=0, stop=(n_points - 1),
                                 num=downsampled_points,
                                 endpoint=True)

        # Design the low-pass filter to be used
        # The Nyquist rate of the signal.
        nyq_rate = frequency / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate
        width = (((frequency - self._downsample_params['down_frequency']) //
                  self._downsample_params['width']) /
                 nyq_rate)

        # The desired attenuation in the stop band, in dB.
        ripple_db = self._downsample_params['ripple_db']

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = scisig.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = self._downsample_params['down_frequency'] / 2.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = scisig.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        # Downsample the signal
        downsampled_record = self._downsample_signal(
            taps=taps, x_axis=x_axis, new_x_axis=new_x_axis, signal=record)

        # Re-align the QRS indices if applicable
        if qrs is not None:
            new_qrs = qrs / downsampling_ratio
            new_qrs = np.round(new_qrs).astype(np.int)

            # Remove any repeating indices due to the rounding operation
            new_qrs = np.unique(new_qrs)

            # Reomve any QRS index which is out of bound
            valid_inds = np.where(new_qrs < downsampled_record.shape[0])[0]
            new_qrs = new_qrs[valid_inds]

        else:
            new_qrs = np.array([])

        # Re-align the labels if applicable
        if labels is not None and qrs is not None:
            new_labels = labels[valid_inds]

        else:
            new_labels = np.array([])

        return downsampled_record, new_qrs, new_labels

    def process_record(
            self,
            record: np.ndarray = None,
            qrs: np.ndarray = None,
            labels: np.ndarray = None,
            frequency: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        The method with the main logic in class, which applies all of the
        required pre-processing steps.

        :param record: (np.ndarray) The record to process
        :param qrs: (np.ndarray) QRS annotations
        :param labels: (np.ndarray) Rhythm annotations
        :param frequency: (float) The record's sampling frequency

        :return: (dict) Containing four keys, 'signal', 'spectrum', 'rhythms', 'qrs',
        containing the processed signal, the estimated power spectrum the
        rhythm annotations and the original QRS indices, each of which is a NumPy array.
        """

        print(f"\n{'-' * 25}Processing record{'-' * 25}\n")

        assert (self._output_signal != 'RR' or qrs is not None), \
            "If output_signal is RR then qrs cannot be None and must be provided."

        # Remove edges if applicable
        if self._n:
            print(f"Trimming the first & last {self._n} seconds")

            record, qrs, labels = self._trim_n_seconds(
                signal=record,
                frequency=frequency,
                n=self._n,
                qrs=qrs,
                rhythms=labels,
            )

        # If the required signal is RR intervals then switch to it now
        if self._output_signal == 'RR':
            record = np.diff((qrs / frequency))

            # Adjust the beat rhythm annotations accordingly,
            # by shifting them 1 to the right, just the like the np.diff operator does.
            labels = labels[1:]

        # Estimate the signal's spectrum if required
        if self._spectrum and not self._align_ecg:
            print(f"Estimating the signal's spectrum")
            spectrum = self._estimate_spectrum(signal=record, frequency=frequency,
                                               aligned=True)

        else:
            spectrum = np.ndarray([])

        # Downsample the ECG signal if required
        if self._downsample_params is not None and self._output_signal == 'ECG':
            signal, qrs, labels = self._downsample(
                record=record,
                qrs=qrs,
                labels=labels,
                frequency=frequency,
            )

        # If we don't do down-sampling, we would still like to work on a copy of
        # the record, in case we will need the original record to
        # estimate the spectrum on aligned ECG windows
        else:
            signal = np.copy(record)

        # Detrend the relevant signal if required
        if self._de_trend:
            print("Detrending the signal")
            signal = self._detrend(signal)

        # Re-scale the relevant signal if required
        if self._scale is not None:
            print("Re-Scaling the signal")

            if self._scale == 'z':
                signal, self.scaling_parameters = self._z_norm(signal=signal)

            elif self._scale == 'scale':
                signal, self.scaling_parameters = self._min_max_norm(signal=signal)

            elif self._scale == 'tanh':
                signal, self.scaling_parameters = self._tanh_norm(signal=signal)

        # Align the ECG signals on a constant grid if required
        if self._align_ecg and self._output_signal == 'ECG':
            print(f"Aligning the ECG beats on a constant grid")

            # If spectrum estimation is required then do it on a per-window basis
            if self._spectrum:
                signal, _ = self._realign_beats(
                    record=record,
                    qrs=qrs,
                    labels=labels,
                )
                spectrum = self._estimate_spectrum(signal=signal,
                                                   frequency=frequency,
                                                   aligned=True)

            else:
                signal, labels = self._realign_beats(
                    record=signal,
                    qrs=qrs,
                    labels=labels,
                )
                spectrum = np.array([])

        processed_record = {
            'signal': signal,
            'spectrum': spectrum,
            'rhythms': labels,
            'qrs': qrs,
        }

        return processed_record

    def __call__(
            self,
            record: np.ndarray = None,
            qrs: np.ndarray = None,
            labels: np.ndarray = None,
            frequency: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        return self.process_record(
            record=record,
            qrs=qrs,
            labels=labels,
            frequency=frequency,
        )
