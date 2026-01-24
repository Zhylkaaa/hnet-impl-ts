"""Augmentation methods taken from https://github.com/Nokia-Bell-Labs/ecg-foundation-model/tree/main"""

import numpy as np
import random


CHANNEL_MAPPING = {
    "I": 0,
    "II": 1,
    "III": 2,
    "AVR": 3,
    "AVL": 4,
    "AVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
    "X": 12,
    "Y": 13,
    "Z": 14,
}


def _match_shape(output_signal, input_signal):
    return np.reshape(output_signal, input_signal.shape)


def identity(ecg_signal, noise_data_mat=None, sampling_rate=None, channel_name=None):
    """
    Identity function for ECG signal, no augmentation applied.
    """
    return ecg_signal.copy()


def load_noise_from_mat(noise_type, ecg_length, noise_rms, noise_data_mat):
    """
    Load noise from a .mat file and adjust it to the desired RMS value.

    Args:
        noise_type (int): 0 - none, 1 - motion artefact, 2 - electrode movement, 3 - baseline wander, 4 - mixture
        ecg_length (int): Length of the ECG signal to generate noise for
        noise_rms (float): Desired RMS value for the noise
        noise_data_mat (dict): Loaded .mat data containing noise

    Returns:
        np.ndarray: Noise array of shape (n_leads, ecg_length)
    """
    noise_map = {
        1: "motion_artefacts",
        2: "electrode_movement",
        3: "baseline_wander",
        4: "mixture_of_noises",
    }
    n_leads = 15

    if noise_type == 0:
        return np.zeros((n_leads, ecg_length))

    noise_key = noise_map.get(noise_type)
    if noise_key is None or noise_key not in noise_data_mat:
        raise ValueError("Invalid noise_type or missing key in .mat data")

    noise_data = noise_data_mat[noise_key]  # shape: (15, noise_length)
    noise_length = noise_data.shape[1]

    if ecg_length < noise_length:
        noise_start = random.randint(0, noise_length - ecg_length - 1)
        multilead_noise = noise_data[:, noise_start : noise_start + ecg_length]
    else:
        half_length = noise_length // 2
        cycles = int(np.ceil(ecg_length / half_length))
        base_noise = noise_data[:, :half_length]
        multilead_noise = np.tile(base_noise, (1, cycles))[:, :ecg_length]

    # Adjust to desired RMS
    for i in range(multilead_noise.shape[0]):
        std = np.std(multilead_noise[i])
        if std > 0:
            multilead_noise[i] = noise_rms * (multilead_noise[i] / std)
        else:
            multilead_noise[i] = 0

    return multilead_noise


def add_motion_with_mat(
    ecg_signal,
    noise_data_mat,
    sampling_rate=None,
    channel_name=None,
    noise_type=1,
    noise_rms=0.1,
):
    noise = load_noise_from_mat(noise_type, ecg_signal.shape[-1], noise_rms, noise_data_mat)
    if channel_name == "ALL":
        return ecg_signal + noise[:ecg_signal.shape[0], :]
    else:
        channel_id = CHANNEL_MAPPING.get(channel_name, 0)
        return ecg_signal + noise[channel_id, :]


def add_electrode_with_mat(
    ecg_signal,
    noise_data_mat,
    sampling_rate=None,
    channel_name=None,
    noise_type=2,
    noise_rms=0.1,
):
    noise = load_noise_from_mat(noise_type, ecg_signal.shape[-1], noise_rms, noise_data_mat)
    if channel_name == "ALL":
        return ecg_signal + noise[:ecg_signal.shape[0], :]
    else:
        channel_id = CHANNEL_MAPPING.get(channel_name, 0)
        return ecg_signal + noise[channel_id, :]


def add_baseline_with_mat(
    ecg_signal,
    noise_data_mat,
    sampling_rate=None,
    channel_name=None,
    noise_type=3,
    noise_rms=0.1,
):
    noise = load_noise_from_mat(noise_type, ecg_signal.shape[-1], noise_rms, noise_data_mat)
    if channel_name == "ALL":
        return ecg_signal + noise[:ecg_signal.shape[0], :]
    else:
        channel_id = CHANNEL_MAPPING.get(channel_name, 0)
        return ecg_signal + noise[channel_id, :]


def add_noise_with_mat(
    ecg_signal,
    noise_data_mat,
    sampling_rate=None,
    channel_name=None,
    noise_type=4,
    noise_rms=0.1,
):
    noise = load_noise_from_mat(noise_type, ecg_signal.shape[-1], noise_rms, noise_data_mat)
    if channel_name == "ALL":
        return ecg_signal + noise[:ecg_signal.shape[0], :]
    else:
        channel_id = CHANNEL_MAPPING.get(channel_name, 0)
        return ecg_signal + noise[channel_id, :]


def zero_masking(ecg_signal, mask_ratio=0.1):
    """
    Apply zero masking to the ECG signal.

    Args:
        ecg_signal (np.ndarray): The original ECG signal.
        mask_ratio (float): The ratio of the signal to be masked with zeros.

    Returns:
        np.ndarray: The ECG signal with zero masking applied.
    """
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ecg_signal must be a numpy array.")

    if ecg_signal.ndim != 1:
        n_samples = ecg_signal.shape[-1]
    else:
        n_samples = len(ecg_signal)

    mask_length = int(n_samples * mask_ratio)
    start_idx = random.randint(0, n_samples - mask_length)
    masked_signal = ecg_signal.copy()

    if ecg_signal.ndim != 1:
        masked_signal[:, start_idx : start_idx + mask_length] = 0
    else:
        masked_signal[start_idx : start_idx + mask_length] = 0

    return masked_signal


############### ECG-Specific Augmentations ###############
def ecg_positive_augmentation(
    ecg_signal, channel_name, sampling_rate, noise_mat, p=0.2
):
    """
    Apply a random ECG-specific positive augmentation.

    Some augmentations only make sense with specific channels or frequencies.

    Usage:
        plot_rpqst(ecg_signal, sampling_rate)
        plot_ecg_augmentation(ecg_signal, sampling_rate, add_motion_with_mat)
    """
    augmentations = [
        identity,  # No augmentation
        add_motion_with_mat,
        add_electrode_with_mat,
        add_baseline_with_mat,
        add_noise_with_mat,
    ] # 20% probability not doing augmentation here

    aug_func = random.choice(augmentations)
    aug_signal = aug_func(ecg_signal, noise_mat, sampling_rate, channel_name)

    if p > 0 and random.random() < p:
        aug_signal = zero_masking(aug_signal, mask_ratio=random.uniform(0.1, 0.3))

    return _match_shape(aug_signal, ecg_signal)