import pandas as pd
import numpy as np

def _signal_features(window_data_df, channels, fs):
    features = pd.DataFrame()

    # Compute signal entropy
    feat_df_signal_entropy = _signal_entropy(window_data_df, channels)

    # Compute RMS
    feat_df_signal_rms = _signal_rms(window_data_df, channels)

    # Compute range
    feat_df_signal_range = _signal_range(window_data_df, channels)

    # Compute Dominant Frequency
    sampling_rate = fs
    frequncy_cutoff = 12.0
    feat_df_dom_freq = _dominant_frequency(window_data_df, sampling_rate, frequncy_cutoff, channels)

    # Compute mean cross rate
    feat_df_mean_cross_rate = _mean_cross_rate(window_data_df, channels)

    features = features.join(feat_df_signal_entropy, how='outer')
    features = features.join(feat_df_signal_rms, how='outer')
    features = features.join(feat_df_signal_range, how='outer')
    features = features.join(feat_df_dom_freq, how='outer')
    features = features.join(feat_df_mean_cross_rate, how='outer')

    return features

def _signal_entropy(signal_df, channels):
    signal_entropy_df = pd.DataFrame()

    for channel in channels:
        data_norm = signal_df[channel]/np.std(signal_df[channel])
        h, d = _histogram(data_norm)

        lowerbound = d[0]
        upperbound = d[1]
        ncell = int(d[2])

        estimate = 0
        sigma = 0
        count = 0

        for n in range(ncell):
            if h[n] != 0:
                logf = np.log(h[n])
            else:
                logf = 0
            count = count + h[n]
            estimate = estimate - h[n] * logf
            sigma = sigma + h[n] * logf ** 2

        nbias = -(float(ncell) - 1) / (2 * count)

        estimate = estimate / count
        estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell) - nbias

        # Scale the entropy estimate to stretch the range
        estimate = np.exp(estimate ** 2) - np.exp(0) - 1

        signal_entropy_df[channel + '_signal_entropy'] = [estimate]

    return signal_entropy_df

def _signal_rms(signal_df, channels):
    rms_df = pd.DataFrame()

    for channel in channels:
        rms_df[channel + '_rms'] = [np.std(signal_df[channel] - signal_df[channel].mean())]

    return rms_df

def _signal_range(signal_df, channels):
    range_df = pd.DataFrame()

    for channel in channels:
        range_df[channel + '_range'] = [signal_df[channel].max(skipna=True) - signal_df[channel].min(skipna=True)]

    return range_df

def _dominant_frequency(signal_df, sampling_rate, cutoff, channels):
    from scipy import stats

    dominant_freq_df = pd.DataFrame()
    for channel in channels:
        signal_x = signal_df[channel]

        padfactor = 1
        dim = signal_x.shape
        nfft = 2 ** ((dim[0] * padfactor).bit_length())

        freq_hat = np.fft.fftfreq(nfft) * sampling_rate
        freq = freq_hat[0:nfft // 2]

        idx1 = freq <= cutoff
        idx_cutoff = np.argwhere(idx1)
        freq = freq[idx_cutoff]

        sp_hat = np.fft.fft(signal_x, nfft)
        sp = sp_hat[0:nfft // 2] * np.conjugate(sp_hat[0:nfft // 2])
        sp = sp[idx_cutoff]
        sp_norm = sp / sum(sp)

        max_freq = freq[sp_norm.argmax()][0]
        max_freq_val = sp_norm.max().real

        idx2 = (freq > max_freq - 0.5) * (freq < max_freq + 0.5)
        idx_freq_range = np.where(idx2)[0]
        dom_freq_ratio = sp_norm[idx_freq_range].real.sum()

        # Calculate spectral flatness
        spectral_flatness = 10.0*np.log10(stats.mstats.gmean(sp_norm)/np.mean(sp_norm))

        # Estimate spectral entropy
        spectral_entropy_estimate = 0
        for isess in range(len(sp_norm)):
            if sp_norm[isess] != 0:
                logps = np.log2(sp_norm[isess])
            else:
                logps = 0
            spectral_entropy_estimate = spectral_entropy_estimate - logps * sp_norm[isess]

        spectral_entropy_estimate = spectral_entropy_estimate / np.log2(len(sp_norm))
        # spectral_entropy_estimate = (spectral_entropy_estimate - 0.5) / (1.5 - spectral_entropy_estimate)

        dominant_freq_df[channel + '_dom_freq_value'] = [max_freq]
        dominant_freq_df[channel + '_dom_freq_magnitude'] = [max_freq_val]
        dominant_freq_df[channel + '_dom_freq_ratio'] = [dom_freq_ratio]
        dominant_freq_df[channel + '_spectral_flatness'] = [spectral_flatness[0].real]
        dominant_freq_df[channel + '_spectral_entropy'] = [spectral_entropy_estimate[0].real]

    return dominant_freq_df

def _mean_cross_rate(signal_df, channels):
    '''
    Compute mean cross rate of sensor signals.

    :param signal_df: dataframe housing desired sensor signals
    :param channels: channels of signal to measure mean cross rate
    :return: dataframe housing calculated mean cross rate for each signal channel
    '''
    mean_cross_rate_df = pd.DataFrame()
    signal_df_mean = signal_df[channels] - signal_df[channels].mean()

    for channel in channels:
        MCR = 0

        for i in range(len(signal_df_mean) - 1):
            if np.sign(signal_df_mean.loc[i, channel]) != np.sign(signal_df_mean.loc[i + 1, channel]):
                MCR += 1

        MCR = float(MCR) / len(signal_df_mean)

        mean_cross_rate_df[channel + '_mean_cross_rate'] = [MCR]

    return mean_cross_rate_df

def _histogram(signal_x):
    descriptor = np.zeros(3)

    ncell = np.ceil(np.sqrt(len(signal_x)))

    max_val = np.nanmax(signal_x.values)
    min_val = np.nanmin(signal_x.values)

    delta = (max_val - min_val) / (len(signal_x) - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ncell

    h = np.histogram(signal_x, ncell.astype(int), range=(min_val, max_val))

    return h[0], descriptor