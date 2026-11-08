"""
MIT-BIH data: filters, AAMI classes, record splits, the two dataset classes
(hand-crafted features and raw beat windows), RR intervals and NSTDB noise.
"""
import os

import numpy as np
import scipy.signal as sp
import torch
import wfdb
from scipy.stats import skew, kurtosis
from torch.utils.data import Dataset

# Specify cutoff in Hertz
lpf_cutoff = 0.5 
hpf_cutoff = 20
# 2. Funktion för Bandpassfilter (Högpass + Lågpass kombinerat)
def butter_bandpass_filter(data, cutoff_low, cutoff_high, fs, order=4):
    nyq = 0.5 * fs
    # Normalisera båda brytfrekvenserna som en lista [low, high]
    normal_cutoff = [cutoff_low / nyq, cutoff_high / nyq]

    # btype='bandpass' skapar både högpass och lågpass samtidigt
    sos = sp.butter(order, normal_cutoff, btype="bandpass", analog=False, output="sos")

    # Filtrera med nollfasförskjutning
    filtered_data = sp.sosfiltfilt(sos, data)
    return filtered_data


def notch_filter(data, notch_freq,fs, quality_factor=30.0):
    # Calculate filter coefficients
    b, a = sp.iirnotch(notch_freq, quality_factor, fs)

    # 3. Apply the filter using zero-phase filtering (filtfilt) to prevent time shifts
    filtered_signal = sp.filtfilt(b, a, data)
    return filtered_signal

# Mapping MIT-BIH symbols to standard 5 AAMI classes
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal / Non-ectopic
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular Ectopic
    'V': 2, 'E': 2,                          # Ventricular Ectopic
    'F': 3#,                                  # Fusion Beat
    #'/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced / Artifacts
}
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']#, 'Unknown (Q)']

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mit-bih-arrhythmia-database-1.0.0") + os.sep
noise_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mit-bih-noise-stress-test-database-1.0.0") + os.sep

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
    print("WARNING: pywavelets not installed (`pip install PyWavelets`). "
          "Wavelet-energy features will be filled with zeros.")

# AAMI-standard symbols used to build per-record beat templates, independent
# of however AAMI_MAPPING happens to be indexed.
DEFAULT_NORMAL_SYMBOLS = ('N', 'L', 'R', 'e', 'j')
DEFAULT_V_SYMBOLS = ('V', 'E')

# Annotation symbols that mark real heart activity. RR intervals are measured
# between these only, since rhythm, noise and comment annotations are not beats.
RR_BEAT_SYMBOLS = ('N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e',
                   'j', 'n', 'E', '/', 'f', 'Q', '?', '!')


class MITBIHFeatureDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.
    Extracts hand-crafted feature vectors (RR-interval, morphological,
    statistical, spectral, wavelet, dual-template-similarity, and
    optional second-lead features) for each annotated beat.

    IMPORTANT: Fit feature templates and normalization stats on the training
    split only, then pass them into val/test dataset constructors:

        train_ds = MITBIHFeatureDataset(train_records)
        val_ds   = MITBIHFeatureDataset(val_records,
                                         templates=train_ds.templates,
                                         feature_mean=train_ds.feature_mean,
                                         feature_std=train_ds.feature_std)
    """
    def __init__(self, record_list, window_size=256, channel=0, channel2=1,
                 offset=0,
                 normal_symbols=DEFAULT_NORMAL_SYMBOLS,
                 v_symbols=DEFAULT_V_SYMBOLS,
                 wavelet='db4', wavelet_level=3,
                 normalize=True, feature_mean=None, feature_std=None,
                 templates=None):
        """
        Args:
            record_list (list): List of record IDs, e.g., ['100', '101', '102'].
            window_size (int): Window length (samples) used to compute
                morphological/statistical features around each R-peak.
            channel (int): Primary ECG lead channel (0 is typically Lead II).
            channel2 (int or None): Secondary ECG lead channel used for a
                small set of cross-lead features. Set to None to disable.
                Records with fewer than `channel2 + 1` signals fall back to
                zero-filled second-lead features automatically.
            offset (int): Sample offset applied to the window center.
            normal_symbols (tuple): Annotation symbols treated as "Normal"
                when fitting the normal-beat template.
            v_symbols (tuple): Annotation symbols treated as "Ventricular"
                when fitting the ventricular-beat template.
            wavelet (str): PyWavelets wavelet name for morphology features.
            wavelet_level (int): Decomposition level for wavelet features.
            normalize (bool): Whether to z-score normalize features.
            feature_mean, feature_std (np.ndarray or None): If provided,
                these stats are used to transform features instead of
                fitting new ones from this dataset (use train split's
                stats for val/test datasets).
            templates (dict or None): Pre-fitted feature templates. Pass
                templates from the training dataset to val/test datasets.
        """
        self.window_size = window_size
        self.channel = channel
        self.channel2 = channel2
        self.offset = offset
        self.normal_symbols = set(normal_symbols)
        self.v_symbols = set(v_symbols)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.templates = templates
        self.features = []
        self.labels = []
        self.feature_names = None

        self._load_data(record_list)

        self.raw_features = self.features.copy()
        self.feature_mean = None
        self.feature_std = None
        if normalize and len(self.features) > 0:
            if feature_mean is not None and feature_std is not None:
                self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
                self.feature_std = np.asarray(feature_std, dtype=np.float32)
            else:
                self.feature_mean = self.features.mean(axis=0)
                self.feature_std = self.features.std(axis=0)
            safe_std = self.feature_std.copy()
            safe_std[safe_std == 0] = 1.0
            self.features = (self.features - self.feature_mean) / safe_std

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self, record_list):
        half_window = self.window_size // 2
        record_beats_by_record = []
        normal_segments, v_segments = [], []
        normal_segments2 = []
        all_segments = []
        excluded_beats = 0

        for record_name in record_list:
            record = wfdb.rdrecord(directory + record_name)
            annotation = wfdb.rdann(directory + record_name, 'atr')
            fs = record.fs

            signal = record.p_signal[:, self.channel]
            signal = butter_bandpass_filter(signal, lpf_cutoff, hpf_cutoff, fs)
            signal = notch_filter(signal, 50, fs)

            has_ch2 = (self.channel2 is not None
                       and record.p_signal.shape[1] > self.channel2)
            if has_ch2:
                signal2 = record.p_signal[:, self.channel2]
                signal2 = butter_bandpass_filter(signal2, lpf_cutoff, hpf_cutoff, fs)
                signal2 = notch_filter(signal2, 50, fs)
            else:
                signal2 = None

            samples = annotation.sample
            symbols = annotation.symbol
            n_annotations = len(samples)
            beat_samples = samples[np.isin(symbols, RR_BEAT_SYMBOLS)]
            n_beats = len(beat_samples)

            # ---- Pass 1: extract segments + RR features for this record ----
            record_beats = []
            for i in range(n_annotations):
                sample_idx = samples[i]
                symbol = symbols[i]

                if symbol not in AAMI_MAPPING:
                    continue

                start_idx = sample_idx - half_window + self.offset
                end_idx = sample_idx + half_window + self.offset
                if start_idx < 0 or end_idx >= len(signal):
                    excluded_beats += 1
                    continue

                segment = signal[start_idx:end_idx]
                segment2 = signal2[start_idx:end_idx] if has_ch2 else None

                beat_pos = np.searchsorted(beat_samples, sample_idx)
                pre_rr = (sample_idx - beat_samples[beat_pos - 1]) / fs if beat_pos > 0 else np.nan
                post_rr = (beat_samples[beat_pos + 1] - sample_idx) / fs if beat_pos < n_beats - 1 else np.nan

                local_window = beat_samples[max(0, beat_pos - 5):min(n_beats, beat_pos + 6)]
                local_rr_avg = np.mean(np.diff(local_window)) / fs if len(local_window) > 1 else np.nan

                if np.isnan(pre_rr):
                    pre_rr = post_rr if not np.isnan(post_rr) else local_rr_avg
                if np.isnan(post_rr):
                    post_rr = pre_rr if not np.isnan(pre_rr) else local_rr_avg
                if np.isnan(local_rr_avg):
                    local_rr_avg = np.nanmean([pre_rr, post_rr])

                rr_ratio = pre_rr / post_rr if post_rr != 0 else 0.0
                compensatory_pause_ratio = (post_rr - pre_rr) / pre_rr if pre_rr != 0 else 0.0

                record_beats.append(dict(
                    segment=segment, segment2=segment2, symbol=symbol, fs=fs,
                    pre_rr=pre_rr, post_rr=post_rr,
                    local_rr_avg=local_rr_avg, rr_ratio=rr_ratio,
                    compensatory_pause_ratio=compensatory_pause_ratio,
                ))

                if self.templates is None:
                    if symbol in self.normal_symbols:
                        normal_segments.append(segment)
                        if has_ch2:
                            normal_segments2.append(segment2)
                    elif symbol in self.v_symbols:
                        v_segments.append(segment)

            if len(record_beats) == 0:
                continue

            if self.templates is None:
                all_segments.extend(b['segment'] for b in record_beats)
            record_beats_by_record.append((record_beats, fs))

        if self.templates is None and all_segments:
            fallback_template = np.mean(np.stack(all_segments), axis=0)
            self.templates = {
                'normal': (np.mean(np.stack(normal_segments), axis=0)
                           if normal_segments else fallback_template),
                'ventricular': (np.mean(np.stack(v_segments), axis=0)
                                if v_segments else fallback_template),
                'normal_ch2': (np.mean(np.stack(normal_segments2), axis=0)
                               if normal_segments2 else None),
            }

        if self.templates is not None:
            template_n = self.templates['normal']
            template_v = self.templates['ventricular']
            template_n2 = self.templates['normal_ch2']
        else:
            template_n = template_v = template_n2 = None

        for record_beats, fs in record_beats_by_record:
            if template_n is None:
                continue
            template_n_qrs_width = self._compute_qrs_width(template_n, fs)
            for b in record_beats:
                feature_vector, names = self._extract_features(
                    b['segment'], b['fs'], template_n, template_v,
                    template_n_qrs_width,
                    b['pre_rr'], b['post_rr'], b['local_rr_avg'],
                    b['rr_ratio'], b['compensatory_pause_ratio'],
                    segment2=b['segment2'], template_n2=template_n2,
                )
                if self.feature_names is None:
                    self.feature_names = names

                self.features.append(feature_vector)
                self.labels.append(AAMI_MAPPING[b['symbol']])

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Excluded beats: {excluded_beats}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_qrs_width(segment, fs):
        """QRS width (seconds) via half-amplitude threshold, searched
        outward from the R-peak in each direction (avoids picking up a
        neighboring beat elsewhere in the window)."""
        r_peak_idx = np.argmax(segment)
        r_peak_amp = segment[r_peak_idx]
        threshold = 0.5 * np.abs(r_peak_amp)

        left = r_peak_idx
        while left > 0 and np.abs(segment[left]) >= threshold:
            left -= 1
        right = r_peak_idx
        while right < len(segment) - 1 and np.abs(segment[right]) >= threshold:
            right += 1
        return (right - left) / fs

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_features(self, segment, fs, template_n, template_v,
                           template_n_qrs_width, pre_rr, post_rr,
                           local_rr_avg, rr_ratio, compensatory_pause_ratio,
                           segment2=None, template_n2=None):
        """
        Compute a hand-crafted feature vector for a single beat segment.

        Returns:
            (np.ndarray, list[str]): feature values and their names.
        """
        names = []
        values = []

        # --- Statistical / amplitude features -----------------------------
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        min_val = np.min(segment)
        max_val = np.max(segment)
        median_val = np.median(segment)
        ptp_val = max_val - min_val
        energy = np.sum(segment ** 2)
        skew_val = skew(segment) if std_val > 0 else 0.0
        kurt_val = kurtosis(segment) if std_val > 0 else 0.0

        names += ['mean', 'std', 'min', 'max', 'median', 'peak_to_peak',
                  'energy', 'skewness', 'kurtosis']
        values += [mean_val, std_val, min_val, max_val, median_val,
                   ptp_val, energy, skew_val, kurt_val]

        # --- Hjorth parameters + zero-crossing complexity ------------------
        d1 = np.diff(segment)
        d2 = np.diff(d1)
        var0, var1, var2 = np.var(segment), np.var(d1), np.var(d2)
        hjorth_mobility = np.sqrt(var1 / var0) if var0 > 0 else 0.0
        mobility_d1 = np.sqrt(var2 / var1) if var1 > 0 else 0.0
        hjorth_complexity = mobility_d1 / hjorth_mobility if hjorth_mobility > 0 else 0.0
        # Sign changes in the first derivative - crude "notchiness" measure.
        # Fusion/Ventricular beats tend to have more inflection points than
        # a clean Normal beat.
        zero_crossings = np.sum(np.diff(np.sign(d1)) != 0)

        names += ['hjorth_mobility', 'hjorth_complexity', 'zero_crossings']
        values += [hjorth_mobility, hjorth_complexity, zero_crossings]

        # --- Morphological features -----------------------------------
        r_peak_idx = np.argmax(segment)
        r_peak_amp = segment[r_peak_idx]
        qrs_width = self._compute_qrs_width(segment, fs)
        # QRS width relative to THIS PATIENT's own normal QRS width - "wide"
        # is only meaningful relative to a patient's own baseline.
        qrs_width_delta = qrs_width - template_n_qrs_width

        max_slope = np.max(d1) if len(d1) else 0.0
        min_slope = np.min(d1) if len(d1) else 0.0

        names += ['r_peak_amplitude', 'r_peak_position', 'qrs_width',
                  'qrs_width_delta', 'max_slope', 'min_slope']
        values += [r_peak_amp, r_peak_idx / len(segment), qrs_width,
                   qrs_width_delta, max_slope, min_slope]

        # --- Dual-template-similarity features ------------------------
        # Fusion beats are, by definition, a morphological blend of a
        # Normal beat and a Ventricular ectopic beat, so where a beat sits
        # BETWEEN the two patient-specific templates is more informative
        # for separating F than distance-from-Normal alone.
        def _corr(a, b):
            if np.std(a) > 0 and np.std(b) > 0:
                return np.corrcoef(a, b)[0, 1]
            return 0.0

        template_n_correlation = _corr(segment, template_n)
        template_n_distance = np.linalg.norm(segment - template_n) / len(segment)
        template_v_correlation = _corr(segment, template_v)
        template_v_distance = np.linalg.norm(segment - template_v) / len(segment)

        blend_score = template_v_correlation - template_n_correlation
        denom = template_n_distance + template_v_distance + 1e-8
        blend_distance_ratio = template_n_distance / denom  # ~0 -> close to N, ~1 -> close to V

        names += ['template_n_correlation', 'template_n_distance',
                   'template_v_correlation', 'template_v_distance',
                   'blend_score', 'blend_distance_ratio']
        values += [template_n_correlation, template_n_distance,
                    template_v_correlation, template_v_distance,
                    blend_score, blend_distance_ratio]

        # --- Spectral features -----------------------------------------
        fft_vals = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / fs)
        total_power = np.sum(fft_vals ** 2) + 1e-12

        dominant_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) else 0.0

        band_edges = [(0, 5), (5, 15), (15, 30), (30, fs / 2)]
        band_powers = []
        for lo, hi in band_edges:
            mask = (freqs >= lo) & (freqs < hi)
            band_power = np.sum(fft_vals[mask] ** 2) / total_power
            band_powers.append(band_power)

        names += ['dominant_freq'] + [f'band_power_{lo}_{hi}' for lo, hi in band_edges]
        values += [dominant_freq] + band_powers

        # --- Wavelet-energy features -------------------------------------
        if _HAS_PYWT:
            coeffs = pywt.wavedec(segment, self.wavelet, level=self.wavelet_level)
            energies = np.array([np.sum(c ** 2) for c in coeffs])
            total_wavelet_energy = np.sum(energies) + 1e-12
            wavelet_energy_ratios = energies / total_wavelet_energy
        else:
            wavelet_energy_ratios = np.zeros(self.wavelet_level + 1)

        names += [f'wavelet_energy_{i}' for i in range(len(wavelet_energy_ratios))]
        values += list(wavelet_energy_ratios)

        # --- RR-interval features --------------------------------------
        names += ['pre_rr', 'post_rr', 'local_rr_avg', 'rr_ratio',
                   'compensatory_pause_ratio']
        values += [pre_rr, post_rr, local_rr_avg, rr_ratio,
                   compensatory_pause_ratio]

        # --- Second-lead features ---------------------------------------
        # A beat that's ambiguous on one lead is often much clearer on the
        # other; this is especially useful for the Fusion/Normal boundary.
        # Zero-filled (with a validity flag) when a second lead isn't
        # available for this record, so the feature vector stays fixed-size.
        if segment2 is not None:
            ch2_valid = 1.0
            ch2_std = np.std(segment2)
            ch2_energy = np.sum(segment2 ** 2)
            ch2_r_peak_amp = segment2[np.argmax(segment2)]
            ch2_qrs_width = self._compute_qrs_width(segment2, fs)
            ch2_template_correlation = (
                _corr(segment2, template_n2) if template_n2 is not None else 0.0
            )
        else:
            ch2_valid = 0.0
            ch2_std = ch2_energy = ch2_r_peak_amp = 0.0
            ch2_qrs_width = ch2_template_correlation = 0.0

        names += ['ch2_valid', 'ch2_std', 'ch2_energy', 'ch2_r_peak_amplitude',
                  'ch2_qrs_width', 'ch2_template_correlation']
        values += [ch2_valid, ch2_std, ch2_energy, ch2_r_peak_amp,
                   ch2_qrs_width, ch2_template_correlation]

        return np.array(values, dtype=np.float32), names

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y




class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.
    Extracts 1D ECG beat segments centered on annotated R-peaks.
    """
    def __init__(self, record_list, window_size=256, channel=0, offset=0, preprocessing=True):
        """
        Args:
            record_list (list): List of record IDs, e.g., ['100', '101', '102'].
            window_size (int): Total signal length per window.
            channel (int): ECG lead channel (0 is typically Lead II).
        """
        self.window_size = window_size
        self.channel = channel
        self.beats = []
        self.labels = []
        self.offset = offset
        self.signals = []
        self.preprocessing= preprocessing
        self._load_data(record_list, preprocessing)

    def _load_data(self, record_list, preprocessing):
        half_window = self.window_size // 2
        
        for record_name in record_list:
            # Load raw signal and annotations
            record = wfdb.rdrecord(directory+record_name)
            annotation = wfdb.rdann(directory+record_name, 'atr')
            
            signal = record.p_signal[:, self.channel]
            if preprocessing:
                signal = butter_bandpass_filter(signal,lpf_cutoff,hpf_cutoff,record.fs)
                signal = notch_filter(signal, 50, record.fs)

            last_sample_idx=-self.window_size
            
            self.signals.append(signal)
            
            # Extract valid beats
            for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in AAMI_MAPPING:
                    #print("WARNING: INVALID BEAT: ["+ str(sample_idx)+"] "+ symbol)
                    continue
                if last_sample_idx + self.window_size > sample_idx:
                    print("Too close beats")
                start_idx = sample_idx - half_window + self.offset
                end_idx = sample_idx + half_window + self.offset
                last_sample_idx = sample_idx
                # Boundary check
                if start_idx >= 0 and end_idx < len(signal):
                    segment = signal[start_idx:end_idx]

                    
                    # Apply Z-score normalization to each window individually
                    if preprocessing:
                        std = np.std(segment)
                        if std > 0:
                            segment = (segment - np.mean(segment)) / std
                        else:
                            segment = segment - np.mean(segment)
                    
                    self.beats.append(segment)
                    self.labels.append(AAMI_MAPPING[symbol])
                    
        # Convert lists to NumPy arrays
        self.beats = np.array(self.beats, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        # Shape output for PyTorch 1D Convolution: [Channels, Sequence_Length]
        x = torch.tensor(self.beats[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    

# Record splits: DS1 (train + validation) and DS2 (test)
train_val_records = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
test_records   = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
train_records = ['101', '106', '108', '109', '114', '115', '116', '118', '119', '122', '124', '203', '207', '208', '209', '215', '220', '230']
val_records = ['112', '201', '205', '223']


# ------------------------------------------------------------------
# CNN + RR input: beat windows, RR intervals and noise
# ------------------------------------------------------------------
PEAK_TOLERANCE = 0.15   # sekunder, en detekterad topp så nära slaget räknas som slaget självt


def kept_beats(annotation, signal_length, window_size, offset=0):
    # Samma slag som MITBIHDataset använder
    half_window = window_size // 2
    starts = annotation.sample - half_window + offset
    ends = annotation.sample + half_window + offset
    keep = np.isin(annotation.symbol, list(AAMI_MAPPING)) & (starts >= 0) & (ends < signal_length)
    labels = [AAMI_MAPPING[symbol] for symbol in np.array(annotation.symbol)[keep]]
    return annotation.sample[keep], np.array(labels)


def annotated_peaks(annotation):
    return annotation.sample[np.isin(annotation.symbol, RR_BEAT_SYMBOLS)]


def rr_intervals(beat_samples, peaks, fs):
    # Tid mellan slagets egen topp och närmaste topp före och efter, i sekunder
    tolerance = PEAK_TOLERANCE * fs
    peaks = np.sort(np.asarray(peaks, dtype=float))
    padded = np.concatenate([[np.nan], peaks, [np.nan]])

    # Slagets egen topp är närmaste topp inom toleransen, annars används annoteringen.
    # Detektorn markerar toppen lite senare än annoteringen, så RR mäts mellan toppar
    i = np.searchsorted(peaks, beat_samples)
    distance_left = np.nan_to_num(beat_samples - padded[i], nan=np.inf)
    distance_right = np.nan_to_num(padded[i + 1] - beat_samples, nan=np.inf)
    nearest = np.where(distance_left <= distance_right, padded[i], padded[i + 1])
    own = np.where(np.minimum(distance_left, distance_right) <= tolerance, nearest, beat_samples)

    before = np.searchsorted(peaks, own - tolerance, side="left")
    after = np.searchsorted(peaks, own + tolerance, side="right") + 1
    pre_rr = (own - padded[before]) / fs
    post_rr = (padded[after] - own) / fs

    # Saknas en granne används den andra, som i MITBIHFeatureDataset
    pre_rr = np.where(np.isnan(pre_rr), post_rr, pre_rr)
    post_rr = np.where(np.isnan(post_rr), pre_rr, post_rr)
    return np.stack([pre_rr, post_rr], axis=1)


def found_by_detector(beat_samples, peaks, fs):
    tolerance = PEAK_TOLERANCE * fs
    peaks = np.sort(np.asarray(peaks, dtype=float))
    return (np.searchsorted(peaks, beat_samples + tolerance, side="right")
            - np.searchsorted(peaks, beat_samples - tolerance, side="left")) > 0


def add_noise(signal, noise, snr_db):
    # Skalar bruset så att signal/brus-förhållandet blir snr_db
    noise = noise[:len(signal)]
    scale = np.sqrt(np.var(signal) / (np.var(noise) * 10 ** (snr_db / 10)))
    return signal + scale * noise


def filter_signal(signal, fs):
    # Samma filter som MITBIHDataset
    filtered = butter_bandpass_filter(signal, lpf_cutoff, hpf_cutoff, fs)
    return notch_filter(filtered, 50, fs)


def beat_windows(filtered, beat_samples, window_size, offset=0):
    # Samma z-normalisering per fönster som MITBIHDataset
    half_window = window_size // 2
    windows = np.stack([filtered[s - half_window + offset:s + half_window + offset] for s in beat_samples])
    std = windows.std(axis=1, keepdims=True)
    windows = (windows - windows.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)
    return windows.astype(np.float32)


def train_rr_stats(window_size, offset=0):
    # RR normaliseras med träningsdatans medelvärde och standardavvikelse, som i feature-cachen
    train_rr = []
    for record_name in train_records:
        annotation = wfdb.rdann(directory + record_name, 'atr')
        header = wfdb.rdheader(directory + record_name)
        beat_samples, _ = kept_beats(annotation, header.sig_len, window_size, offset)
        train_rr.append(rr_intervals(beat_samples, annotated_peaks(annotation), header.fs))
    train_rr = np.concatenate(train_rr)
    return train_rr.mean(axis=0), train_rr.std(axis=0)
