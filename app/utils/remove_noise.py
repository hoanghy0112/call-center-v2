from scipy.signal import butter, lfilter

import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=6):
    """
    Create a Butterworth bandpass filter.

    Parameters:
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        fs (int): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        b, a: Filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    Apply a Butterworth bandpass filter to the data.

    Parameters:
        data (np.ndarray): Input signal as a float array.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        fs (int): Sampling frequency.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y


def remove_noise_from_bytearray(
    audio_bytes, fs=44100, lowcut=300.0, highcut=3000.0, order=6
):
    """
    Remove noise from an audio buffer represented as a bytearray.

    Parameters:
        audio_bytes (bytearray): Input audio as bytearray (16-bit PCM).
        fs (int): Sampling frequency (default is 44100 Hz).
        lowcut (float): Low cutoff frequency for bandpass filter.
        highcut (float): High cutoff frequency for bandpass filter.
        order (int): Filter order.

    Returns:
        bytearray: Filtered audio as a bytearray.
    """
    # Convert bytearray to a NumPy array of int16
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    # Normalize to float values in range [-1, 1]
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Apply the bandpass filter
    filtered_float = butter_bandpass_filter(audio_float, lowcut, highcut, fs, order)

    # Convert back to int16 format
    filtered_int16 = np.int16(filtered_float * 32767)

    # Convert the NumPy array back to bytearray
    filtered_bytes = bytearray(filtered_int16.tobytes())
    return filtered_bytes
