import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wfdb
import torchshow as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import copy
from collections import Counter
import scipy.signal as sp
from scipy.stats import skew, kurtosis
from ecgdetectors import Detectors
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
# Specify cutoff in Hertz
lpf_cutoff = 0.5 
hpf_cutoff = 20
detectors = Detectors(360)
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

directory = "../mit-bih-arrhythmia-database-1.0.0/"

def compute_class_weights(labels, num_classes=5):
    """
    Computes balanced class weights inversely proportional to class frequencies.
    
    Formula: weight_i = total_samples / (num_classes * count_i)
    """
    # Count occurrences of each class index
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    # Prevent division by zero if a class has 0 samples
    weights = []
    for count in class_counts:
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)
        
    weights = np.array(weights, dtype=np.float32)
    
    # Optional: Normalize weights so their mean equals 1.0
    weights = weights / np.mean(weights)
    
    return torch.tensor(weights, dtype=torch.float32)

def format_seconds(seconds: float) -> str:
    """
    Convert a float number of seconds into a human-readable string.
    Automatically scales from milliseconds up to days.
    """
    abs_seconds = abs(seconds)
    sign = "-" if seconds < 0 else ""

    if abs_seconds < 1e-3:
        return f"{sign}{abs_seconds * 1e6:.3f} µs"
    elif abs_seconds < 1:
        return f"{sign}{abs_seconds * 1e3:.3f} ms"
    elif abs_seconds < 60:
        return f"{sign}{abs_seconds:.3f} s"
    elif abs_seconds < 3600:
        return f"{sign}{abs_seconds / 60:.3f} min"
    elif abs_seconds < 86400:
        return f"{sign}{abs_seconds / 3600:.3f} h"
    else:
        return f"{sign}{abs_seconds / 86400:.3f} days"




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


class MITBIHFeatureDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.
    Extracts hand-crafted feature vectors (RR-interval, morphological,
    statistical, spectral, wavelet, dual-template-similarity, and
    optional second-lead features) for each annotated beat.

    IMPORTANT - normalization:
    Feature z-score stats (mean/std) are fit on THIS dataset's data unless
    `feature_mean` / `feature_std` are explicitly passed in, in which case
    those stats are used instead (and not re-fit). Always fit stats on the
    TRAINING split only, then pass them into val/test dataset constructors:

        train_ds = MITBIHFeatureDataset(train_records)
        val_ds   = MITBIHFeatureDataset(val_records,
                                         feature_mean=train_ds.feature_mean,
                                         feature_std=train_ds.feature_std)
    """
    def __init__(self, record_list, window_size=256, channel=0, channel2=1,
                 offset=0,
                 normal_symbols=DEFAULT_NORMAL_SYMBOLS,
                 v_symbols=DEFAULT_V_SYMBOLS,
                 wavelet='db4', wavelet_level=3,
                 normalize=True, feature_mean=None, feature_std=None):
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
                when building the per-record normal-beat template.
            v_symbols (tuple): Annotation symbols treated as "Ventricular"
                when building the per-record ventricular-beat template.
            wavelet (str): PyWavelets wavelet name for morphology features.
            wavelet_level (int): Decomposition level for wavelet features.
            normalize (bool): Whether to z-score normalize features.
            feature_mean, feature_std (np.ndarray or None): If provided,
                these stats are used to transform features instead of
                fitting new ones from this dataset (use train split's
                stats for val/test datasets).
        """
        self.window_size = window_size
        self.channel = channel
        self.channel2 = channel2
        self.offset = offset
        self.normal_symbols = set(normal_symbols)
        self.v_symbols = set(v_symbols)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.features = []
        self.labels = []
        self.feature_names = None

        self._load_data(record_list)

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
            n_beats = len(samples)

            # ---- Pass 1: extract segments + RR features for this record ----
            record_beats = []
            normal_segments, v_segments = [], []
            normal_segments2 = []

            for i in range(n_beats):
                sample_idx = samples[i]
                symbol = symbols[i]

                if symbol not in AAMI_MAPPING:
                    continue

                start_idx = sample_idx - half_window + self.offset
                end_idx = sample_idx + half_window + self.offset
                if start_idx < 0 or end_idx >= len(signal):
                    continue

                segment = signal[start_idx:end_idx]
                segment2 = signal2[start_idx:end_idx] if has_ch2 else None

                pre_rr = (sample_idx - samples[i - 1]) / fs if i > 0 else np.nan
                post_rr = (samples[i + 1] - sample_idx) / fs if i < n_beats - 1 else np.nan

                local_window = samples[max(0, i - 5):min(n_beats, i + 6)]
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

                if symbol in self.normal_symbols:
                    normal_segments.append(segment)
                    if has_ch2:
                        normal_segments2.append(segment2)
                elif symbol in self.v_symbols:
                    v_segments.append(segment)

            if len(record_beats) == 0:
                continue

            all_segments = [b['segment'] for b in record_beats]

            # Per-record templates. Each falls back to the record's overall
            # mean beat if that symbol class wasn't observed in this record.
            template_n = (np.mean(np.stack(normal_segments), axis=0)
                          if normal_segments else np.mean(np.stack(all_segments), axis=0))
            template_v = (np.mean(np.stack(v_segments), axis=0)
                          if v_segments else np.mean(np.stack(all_segments), axis=0))
            template_n2 = (np.mean(np.stack(normal_segments2), axis=0)
                           if has_ch2 and normal_segments2 else None)

            template_n_qrs_width = self._compute_qrs_width(template_n, fs)

            # ---- Pass 2: finalize feature vectors using the templates ----
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
    def __init__(self, record_list, window_size=256, channel=0, offset=0):
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
        
        self._load_data(record_list)

    def _load_data(self, record_list):
        half_window = self.window_size // 2
        
        for record_name in record_list:
            # Load raw signal and annotations
            record = wfdb.rdrecord(directory+record_name)
            annotation = wfdb.rdann(directory+record_name, 'atr')
            
            signal = record.p_signal[:, self.channel]
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
    
class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: [Batch, 1, 256] -> [Batch, 32, 128]
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 2: [Batch, 32, 128] -> [Batch, 64, 64]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 3: [Batch, 64, 64] -> [Batch, 128, 32]
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BeatClassifierMLP(nn.Module):
    """
    Simple feed-forward classifier for hand-crafted ECG beat features.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def select_features(train_dataset, n_features_to_select=None, direction='forward',
                     cv=3, scoring='f1_macro', estimator=None, n_jobs=-1, verbose=True):
    """
    Fit sklearn's SequentialFeatureSelector on a (already feature-extracted
    and normalized) training dataset and return the selected feature subset.

    A lightweight sklearn estimator is used purely as a proxy scorer to
    rank feature subsets during selection - the actual downstream model can
    still be the PyTorch MLP; only the subset of engineered features it
    trains on changes. Logistic regression is used by default because it's
    fast enough to fit hundreds of times (forward/backward selection fits
    one model per candidate feature, per step, per CV fold) while still
    being sensitive to which features carry discriminative signal.

    IMPORTANT: fit this on the TRAIN split only, same reasoning as
    normalization stats - selecting features using val/test data leaks
    information about those splits into the "feature engineering" step.

    Args:
        train_dataset: a MITBIHFeatureDataset (train split), already built
            (and ideally already normalized, since it's about to be fed to
            LogisticRegression).
        n_features_to_select (int, float, 'auto', or None): passed through
            to SequentialFeatureSelector. None defaults to half of the
            available features.
        direction ('forward' or 'backward'): forward starts from 0 features
            and adds the best one each step; backward starts from all
            features and removes the worst. Forward is cheaper when
            n_features_to_select is well below the total feature count
            (the common case here).
        cv (int): cross-validation folds used to score each candidate subset.
        scoring (str): sklearn scoring string. 'f1_macro' is used by default
            since it weights all four AAMI classes equally, matching what
            we actually care about (not just overall/Normal-dominated
            accuracy).
        estimator: sklearn-compatible estimator used as the proxy scorer.
            Defaults to class-balanced multinomial LogisticRegression.
        n_jobs (int): parallelism for SequentialFeatureSelector.

    Returns:
        selected_indices (np.ndarray[int]): column indices into
            train_dataset.features that were selected.
        selected_names (list[str] or None): corresponding feature names,
            if train_dataset.feature_names is available.
    """
    X = train_dataset.features
    y = train_dataset.labels

    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, class_weight='balanced')

    if n_features_to_select is None:
        n_features_to_select = max(1, X.shape[1] // 2)

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        direction=direction,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    )
    selector.fit(X, y)

    selected_mask = selector.get_support()
    selected_indices = np.where(selected_mask)[0]
    selected_names = (
        [train_dataset.feature_names[i] for i in selected_indices]
        if train_dataset.feature_names is not None else None
    )

    if verbose:
        print(f"Selected {len(selected_indices)}/{X.shape[1]} features "
              f"(direction={direction}, scoring={scoring}, cv={cv}):")
        if selected_names is not None:
            for name in selected_names:
                print(f"  - {name}")

    return selected_indices, selected_names


def apply_feature_selection(dataset, selected_indices):
    """
    Subset a MITBIHFeatureDataset's feature matrix (and associated
    feature_names / feature_mean / feature_std, if present) to the given
    column indices, in place. Use this to apply the SAME selection (fit on
    train) to val/test datasets.
    """
    dataset.features = dataset.features[:, selected_indices]
    if dataset.feature_names is not None:
        dataset.feature_names = [dataset.feature_names[i] for i in selected_indices]
    if dataset.feature_mean is not None:
        dataset.feature_mean = dataset.feature_mean[selected_indices]
    if dataset.feature_std is not None:
        dataset.feature_std = dataset.feature_std[selected_indices]
    return dataset


class ECG_model_nature(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes=6,
        filter_length=64,
        kernel_size=16,
        drop_rate=0.2
    ):
        super(ECG1DCNN, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.filter_length = filter_length
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate

        # ============================================================
        # Block 1: First convolutional residual block
        # ============================================================
        self.first_block = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(filter_length),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(filter_length),
            nn.ReLU(),

            nn.Dropout(drop_rate),

            nn.Conv1d(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
        )

        # ============================================================
        # Block 2: Main residual blocks
        # ============================================================
        self.main_blocks = nn.Sequential(
            # Block 0: 64 -> 64, stride 2
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 1: 64 -> 64, stride 1
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 2: 64 -> 64, stride 2
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 3: 64 -> 64, stride 1
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 4: 64 -> 128, stride 2
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 5: 128 -> 128, stride 1
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 6: 128 -> 128, stride 2
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 7: 128 -> 128, stride 1
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 8: 128 -> 256, stride 2
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 9: 256 -> 256, stride 1
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 10: 256 -> 256, stride 2
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 11: 256 -> 256, stride 1
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 12: 256 -> 512, stride 2
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 8,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 13: 512 -> 512, stride 1
            ResidualBlock(
                in_channels=filter_length * 8,
                out_channels=filter_length * 8,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 14: 512 -> 512, stride 2
            ResidualBlock(
                in_channels=filter_length * 8,
                out_channels=filter_length * 8,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),
        )

        # ============================================================
        # Block 3: Output block
        # ============================================================
        self.output_block = nn.Sequential(
            nn.BatchNorm1d(filter_length * 8),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate final flattened size
        final_length = input_size

        # 8 blocks have stride=2
        for _ in range(8):
            final_length = final_length // 2

        self.classifier = nn.Linear(
            filter_length * 8 * final_length,
            num_classes
        )

        self._initialize_weights()

    def forward(self, x):

        # Keras: (batch, length, channels)
        # PyTorch: (batch, channels, length)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        # ------------------------------------------------------------
        # Block 1
        # ------------------------------------------------------------
        shortcut = x
        x = self.first_block(x)

        # pool_size=1, stride=1 is an identity operation
        x = shortcut + x

        # ------------------------------------------------------------
        # Block 2
        # ------------------------------------------------------------
        x = self.main_blocks(x)

        # ------------------------------------------------------------
        # Block 3
        # ------------------------------------------------------------
        x = self.output_block(x)

        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):
    """
    One residual block used inside main_blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        drop_rate
    ):
        super(ResidualBlock, self).__init__()

        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(drop_rate)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Shortcut
        self.pool = nn.MaxPool1d(
            kernel_size=stride,
            stride=stride
        )

        if in_channels != out_channels:
            self.channel_pad = nn.ZeroPad2d(
                (0, 0, 0, out_channels - in_channels)
            )
        else:
            self.channel_pad = nn.Identity()

    def forward(self, x):

        # Shortcut
        shortcut = self.pool(x)

        if isinstance(self.channel_pad, nn.ZeroPad2d):
            shortcut = self.channel_pad(shortcut)

        # Main branch
        out = self.bn1(x)
        out = F.relu(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)

        # Residual addition
        return shortcut + out

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc

def count_parameters(model):
    print("Modules", "Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(f"{name}: {params}")
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params
def f1_calc(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get class predictions (highest probability/logit index)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return f1_score(all_targets, all_preds,average='macro')
    
    
def evaluate_and_plot_cm(model, dataloader, device, name):
    """
    Evaluates the model on the test set, prints a classification report,
    and displays an sklearn confusion matrix heatmap.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get class predictions (highest probability/logit index)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Print Detailed Metrics (Precision, Recall, F1-score)
    print(f"\n--- Classification Report {name} dataset ---")
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=CLASS_NAMES, 
        digits=4, 
        zero_division=0
    ))

    # 2. Compute Confusion Matrix via scikit-learn
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(CLASS_NAMES)), normalize='true')

    # 3. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f'MIT-BIH {name} Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return cm

def log_plot_balance(dataset):
    counts = Counter(dataset.labels)
    #print(counts[0])
    counts = [counts[i] for i in range(len(CLASS_NAMES))]
    #print(counts[0])
    count_sum = sum(counts)
    for i in range(len(counts)):
        counts[i]=counts[i]*100/count_sum
    print(counts)
    # Skapa histogrammet
    plt.bar(CLASS_NAMES,counts, color='skyblue', edgecolor='black', log=True)
    # Lägg till titlar och etiketter
    plt.title('Frequency of labels')
    plt.xlabel('Labels')
    plt.ylabel('Procent')
    plt.show()

# 1. Define train and validation record splits
train_val_records = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
test_records   = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# best 80% split
#train_records = ['101', '106', '108', '109', '114', '115', '116', '118', '119', '122', '124', '203', '207', '208', '209', '215', '220', '230']
#val_records =['112', '201', '205', '223']


# NEW best 80% split
train_records = ['101', '106', '108', '109', '114', '115', '116', '118', '119', '122', '124', '203', '207', '208', '209', '215', '220', '230']
val_records = ['112', '201', '205', '223']

# best 90% split
#train_records = ['101', '106', '108', '109', '112', '114', '115', '118', '119', '122', '124', '201', '203', '205', '208', '209', '215', '220', '223', '230']
#val_records = ['116', '207']

# NEW best 90% split
#train_records = ['101', '106', '108', '109', '112', '114', '115', '118', '119', '122', '124', '201', '203', '205', '208', '209', '215', '220', '223', '230']
#val_records = ['116', '207']