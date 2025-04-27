# Standard Library
import sys
from typing import Optional, Tuple, Literal, List
from functools import lru_cache

# Anaconda Libraries
import numpy as np
from numpy import ndarray
import scipy as sp
import matplotlib.pyplot as plt

# Third Party Libraries
from pydub import AudioSegment
from pydub.utils import mediainfo

cp = None # Placeholder for cupy, will be imported if available
cufft = None # Placeholder for cupy.fft, will be imported if available

MAX_AMPLITUDE_AT_16_BIT = 32767 # The maximum amplitude for a 16-bit audio signal
number = int | float

def load_sound_file(file: str) -> Optional[AudioSegment]:
    """Method for loading an audio file and ensuring that the correct format is used to decode the file.

    Args:
        file (str): The path to the file to be loaded
    
    Returns:
       Optional[AudioSegment]: The audio file as an AudioSegment object, or None if an error occurs while loading the file.
    """
    try:
        return AudioSegment.from_file(file, format=mediainfo(file)['format_name'])
    except: # If the file cannot be loaded, print an error message and return None
        print(f"Error loading file {file}")
        print(f'Mediainfo: {mediainfo(file)}')
        print(f'Format: {mediainfo(file)["format_name"]}')
        return None


def window_sound(sound: AudioSegment, length: number=5, overlap: number=2.5) -> List[AudioSegment]:
    """Window an audio signal into smaller segments of a specified length with a specified overlap.
    The segments are returned as a list of AudioSegment objects.

    Args:
        sound (AudioSegment): The audio signal to be windowed.
        length (number, optional): The length of the segments in seconds. Defaults to 5.
        overlap (number, optional): The overlap between segments in seconds. Defaults to 2.5.
    
    Returns:
        List[AudioSegment]: A list of the windowed audio segments.
    """
    length = int(length * 1000) # Convert the length to milliseconds
    overlap = int(overlap * 1000) # Convert the overlap to milliseconds
    sounds = [] # Initialize an empty list to store the windowed segments
    while True:
        if len(sound) < length:
            break # If the remaining sound is shorter than the segment length, break the loop
        sounds.append(sound[:length]) # Append the first segment to the list
        if len(sound) < length + overlap: # If the remaining sound is shorter than the segment length plus the overlap
            sound = sound[-length:] # Keep the last segment
        sound = sound[length - overlap:] # Move the sound forward by the segment length minus the overlap
    return sounds # Return the list of windowed segments


def signal_to_audio(signal: ndarray, sample_rate: int=44100, denormalize: bool=False) -> AudioSegment:
    """Convert a signal array to an audio segment object.
    Will automatically convert cupy arrays to int16 numpy arrays if necessary.

    Args:
        signal (cp.ndarray | np.ndarray): The signal array to be converted.
        sample_rate (int, optional): The sample rate of the signal. Defaults to 44100.
        denormalize (bool, optional): Whether to denormalize the signal to the range of int16. Defaults to False.
    
    Returns:
        AudioSegment: The audio segment object created from the signal array.
    """
    if cp and isinstance(signal, cp.ndarray): # If the signal is a cupy array, convert it to a numpy array
        signal = cp.asnumpy(signal)
    if denormalize: # If the signal should be denormalized, scale it to the range of int16
        signal = signal * MAX_AMPLITUDE_AT_16_BIT
    signal = signal.astype(np.int16) # Convert the signal to a 16-bit integer array
    return AudioSegment(signal.tobytes(), frame_rate=sample_rate, sample_width=signal.dtype.itemsize, channels=1) # Create an audio segment object from the signal array


def normalize_signal(signal: ndarray, max_amplitude: Optional[number]=None) -> ndarray:
    """Normalizes the amplitude of an audio signal to the range [-1,1].
    Can perform normalization on either the CPU or GPU, depending on the input signal type.
    
    Args:
        signal (cp.ndarray | np.ndarray): The audio signal to normalize.
        max_amplitude (int | float, optional): The maximum amplitude value in the signal. Defaults to None.
    
    Returns:
        cp.ndarray|np.ndarray: The normalized audio signal. Matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if cp and isinstance(signal, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array
    if max_amplitude is None: # If the maximum amplitude is not provided find the maximum amplitude in the signal
        max_amplitude = math.max(math.abs(signal))
    return signal / max_amplitude # Normalize the signal to the range [-1,1]


def normalize_decibels(signal: ndarray) -> ndarray:
    """Normalize the decibel values of a signal to the range [0,1] from -120 to 0 db.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.
    
    Args:
        signal (cp.ndarray | np.ndarray): The signal to normalize.
    
    Returns:
        cp.ndarray|np.ndarray: The normalized signal. Return type matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if cp and isinstance(signal, cp.ndarray):
        math = cp # Use CuPy for GPU operations if the signal is a cupy array

    signal = math.clip(signal, -120, 0) # Clip the signal to the range [-120,0]
    signal = signal + 120 # Shift the signal to the range [0,120]
    signal = signal / 120 # Normalize the signal to the range [0,1]
    return signal


def magnitude_to_db(signal: ndarray|number, epsilon: float=sys.float_info.epsilon) -> ndarray|number:
    """Calculate the magnitude of an audio signal in decibels.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.
    
    Args:
        signal (cp.ndarray | np.ndarray | float | int): The audio signal to calculate the magnitude of. Will take either an array or a single value.
        epsilon (float, optional): A small value to prevent `log(0)`. Defaults to `sys.float_info.epsilon`.
        
    Returns:
        cp.ndarray|np.ndarray|float|int: The magnitude of the audio signal in decibels. Return type matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if cp and isinstance(signal, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array
    return 20 * math.log10(math.abs(signal + epsilon)) # Calculate the magnitude in decibels


def fourier_transform(signal: ndarray, sample_rate: int, cache: bool=False) -> Tuple[ndarray, ndarray]:
    """Calculate the Fourier Transform of a real valued signal.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.
    
    Args:
        signal (cp.ndarray | np.ndarray): The signal to calculate the Fourier Transform of.
        sample_rate (int): The sample rate of the signal.
        cache (bool, optional): If True, the FFT plan is cached for faster computation. Defaults to False.
    
    Returns
    --------
     - **cp.ndarray|np.ndarray:** The frequencies of the Fourier Transform. Matches the input signal type.
     - **cp.ndarray|np.ndarray:** The frequency-signal of the Fourier Transform in decibels. Matches the input signal type.
    """    
    fft = sp.fft # Use scipy for CPU operations by default (faster than numpy at performing FFT)
    if cp and isinstance(signal, cp.ndarray):
        fft = cufft # Use CuPy for GPU operations if the signal is a cupy array
    else:
        if cache: # If the cache is requested but the signal is not a cupy array, print a warning
            print('Warning: Cache is only available for GPU operations.')
    
    def rfft(signal, sample_rate): # Helper function to calculate the Fourier Transform
        n = signal.size # Get the number of samples in the signal
        frequencies = fft.rfftfreq(n, 1/sample_rate) # Calculate the frequencies
        signal = fft.rfft(signal, overwrite_x=True) # Calculate the Fourier Transform of the signal
        signal = signal / (n//2) # Normalize the signal
        return frequencies, signal # Return the frequencies and magnitudes of the Fourier Transform
    
    if cp and isinstance(signal, cp.ndarray) and (cache is False): # If the operation is on the GPU and cache is not requested
        with fft.get_fft_plan(signal, value_type='R2C'): # Create the FFT plan
            return rfft(signal, sample_rate) # Calculate the Fourier Transform using the plan
    else:
        return rfft(signal, sample_rate) # Calculate the Fourier Transform


def group_frequencies(frequencies: ndarray,
                      freq_signal: ndarray,
                      freq_bins: Optional[ndarray]=None,
                      num_groups: int=100,
                      ) -> Tuple[ndarray, ndarray]:
    """Group the frequencies of a Fourier Transform into a smaller number of groups by calculating the RMS of each group.
    Each group is spaced evenly along the Bark Scale.
    Can perform the grouping on either the CPU or GPU, depending on the input signal type.

    Args:
        frequencies (cp.ndarray | np.ndarray): The frequencies of the Fourier Transform.
        freq_signal (cp.ndarray | np.ndarray): The resulting complex signal from the Fourier Transform.
        freq_bins (cp.ndarray | np.ndarray, optional): The frequency bins to group the frequencies into. Defaults to None.
        num_groups (int, optional): The number of groups to create. Defaults to 100.
      
    Returns
    -------
    - **cp.ndarray|np.ndarray**: The grouped frequencies. Matches the input signal type.
    - **cp.ndarray|np.ndarray**: The magnitude value for the bin. Matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if cp and isinstance(frequencies, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array

    if freq_bins is None: # If the frequency bins are not provided
        freq_bins = get_bark_spaced_frequencies(num_groups, math.__name__) # Get the Bark spaced frequencies
    bin_indicies = math.digitize(frequencies, freq_bins) - 1 # Group the frequencies into bins
    reduced_freq_signal = math.zeros(len(freq_bins)) # Initialize an array to store the maximum magnitude in each bin
    for i in range(len(freq_bins)): # Iterate through each bin
        bin_values = freq_signal[bin_indicies == i] # Get the values in the bin FIRST
        if bin_values.size == 0: # THEN check if there are no values in the bin
            reduced_freq_signal[i] = 0 # Assign a default value (e.g., 0) or handle as needed
            continue # Skip to the next bin
        # reduced_freq_signal[i] = math.sqrt(math.mean(math.abs(bin_values) ** 2)) # Calculate the RMS of the bin
        reduced_freq_signal[i] = math.max(math.abs(bin_values)) # Calculate the maximum magnitude in the bin

    return freq_bins, reduced_freq_signal # Return the grouped frequencies and maximum magnitudes


def split_around_silence(sounds: AudioSegment, size: number=5, threshold_rms: int=3) -> List[AudioSegment]:
    """Split an audio signal anywhere there is silence and return the segments as a list. Removes segments that are smaller than the specified size.
    
    Args:
        sounds (AudioSegment): The audio signal to split.
        size (number, optional): The minimum size of the segments in seconds. Defaults to 5.
        threshold_rms (int, optional): The threshold for the RMS value to determine silence. Defaults to 3.
        
    Returns:
        List[AudioSegment]: A list of the split audio segments.
    """
    sounds = sounds.split_to_mono() # Process each channel separately
    size = int(size * 1000) # Convert the size to milliseconds
    cut_sounds = [] # Initialize an empty list to store the cut sounds
    for sound in sounds: # Iterate through each channel
        last_cut = 0 # Initialize the last cut index
        for i in range(len(sound)): # Iterate through each sample
            if sound[i].rms <= threshold_rms: # If the RMS value is below the threshold
                if i - last_cut < size: # If the segment is too small
                    last_cut = i # Update the last cut index
                    continue # Skip to the next sample without cutting
                cut_sounds.append(sound[last_cut:i]) # Add the segment to the cut sounds if it is large enough
                last_cut = i # Update the last cut index
        if len(sound) - last_cut >= size: # If the last segment is large enough
            cut_sounds.append(sound[last_cut:]) # Add the last segment to the cut sounds
    return cut_sounds # Return the cut sounds


#################################################
# Frequency Scale Conversions
#################################################

def hz_to_mel(hz: ndarray|number) -> ndarray|number:
    """Convert a frequency value in Hertz to the Mel scale.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.

    Args:
        hz (cp.ndarray | np.ndarray | float | int): The frequency value in Hertz to convert. Will take either an array or a single value.
    
    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to the Mel scale. Return type matches the input signal type.
    
    Formula:
    --------
        mel = 2595 * log10(1 + hz / 700)
    """
    if cp and isinstance(hz, cp.ndarray):
        return 2595 * cp.log10(1 + hz / 700)
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: ndarray|number) -> ndarray|number:
    """Convert a frequency value in the Mel scale to Hertz.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.

    Args:
        mel (cp.ndarray | np.ndarray | float | int): The frequency value in the Mel scale to convert. Will take either an array or a single value.
    
    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to Hertz. Return type matches the input signal type.
    
    Formula:
    --------
        hz = 700 * (10 ** (mel / 2595) - 1)
    """
    return 700 * (10 ** (mel / 2595) - 1)


def hz_to_bark(hz: ndarray|number) -> ndarray|number:
    """Convert a frequency value in Hertz to the Bark scale.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.

    Args:
        hz (cp.ndarray | np.ndarray | float | int): The frequency value in Hertz to convert. Will take either an array or a single value.
    
    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to the Bark scale. Return type matches the input signal type.
    
    Formula:
    --------
        bark = 13 * arctan(0.00076 * hz) + 3.5 * arctan((hz / 7500) ** 2)
    """
    if cp and isinstance(hz, cp.ndarray):
        return 13 * cp.arctan(0.00076 * hz) + 3.5 * cp.arctan((hz / 7500) ** 2)
    return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500) ** 2)


CPU_BARK_KEY = np.arange(0, 20000, 1) # Generate a range of frequencies from 0 to 20000 Hz
CPU_BARK_KEY = hz_to_bark(CPU_BARK_KEY) # Convert the frequencies to the Bark scale
GPU_BARK_KEY = None # Placeholder for the GPU Bark key, will be initialized if cupy is available
if cp: # If cupy is available
    GPU_BARK_KEY = cp.arange(0, 20000, 1) # Generate a range of frequencies from 0 to 20000 Hz on the GPU
    GPU_BARK_KEY = hz_to_bark(GPU_BARK_KEY) # Convert the frequencies to the Bark scale on the GPU


def bark_to_hz(bark: ndarray|number) -> ndarray|number:
    """Convert a frequency value in the Bark scale to Hertz.
    Can only perform the conversion on the CPU, but will return a cupy array if the input is a cupy array.
    Slow and inefficient.

    Args:
        bark (cp.ndarray | np.ndarray | float | int): The frequency value in the Bark scale to convert. Will take either an array or a single value.

    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to Hertz. Return type matches the input signal type.
    """
    def single_bark_to_hz(single_bark):
        return sp.optimize.root_scalar(lambda x: hz_to_bark(x) - single_bark, bracket=[0, 24000], method='brentq').root
    
    convert_to_cupy = False # Flag to convert the output back to cupy if the input was cupy
    if cp and isinstance(bark, cp.ndarray):
        convert_to_cupy = True
        bark = bark.get() # Convert the cupy array to a numpy array
    
    vectorized_bark_to_hz = np.vectorize(single_bark_to_hz) # Vectorize the conversion function for numpy arrays
    hz = vectorized_bark_to_hz(bark) # Convert the bark values to hertz

    if convert_to_cupy: # Convert the output back to cupy if the input was cupy
        return cp.array(hz) 
    return hz

def bark_to_hz(bark: ndarray|number) -> ndarray|number:
    """Convert a frequency value in the Bark scale to Hertz by referencing a key.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.

    Args:
        bark (cp.ndarray | np.ndarray | float | int): The frequency value in the Bark scale to convert. Will take either an array or a single value.
    
    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to Hertz. Return type matches the input signal type.
    """
    if cp and isinstance(bark, cp.ndarray):
        return cp.digitize(bark, GPU_BARK_KEY) - 1
    return np.digitize(bark, CPU_BARK_KEY) - 1
    

@lru_cache(maxsize=None)
def get_bark_spaced_frequencies(num_bands: int = 100, math: Literal['numpy','cupy'] = 'numpy') -> ndarray:
  """Generate a set of frequencies between 20 and 20000 Hz spaced evenly along the Bark scale.
  The frequencies are cached for faster access and to avoid recalculating the same values.

  Args:
    num_bands (int, optional): The number of frequency bands to generate. Defaults to 100.
    math ('numpy' | 'cupy', optional): The library to use for calculations. Defaults to np.

  Returns:
    ndarray: The frequencies spaced evenly along the Bark scale.
  """
  if math not in ['numpy', 'cupy']: # Check if the math library is valid
    raise ValueError("math must be either 'numpy' or 'cupy'")
  math = np if math == 'numpy' else cp # Select the math library to use
  
  frequencies = math.linspace(hz_to_bark(20), hz_to_bark(20000), num_bands) # Generate the frequency bins
  frequencies = bark_to_hz(frequencies) # Convert the frequency bins back to Hz
  return frequencies


def hz_to_erbs(hz: ndarray|number) -> ndarray|number:
    """Convert a frequency value in Hertz to the ERB scale.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.

    Args:
        hz (cp.ndarray | np.ndarray | float | int): The frequency value in Hertz to convert. Will take either an array or a single value.

    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to the ERB scale. Return type matches the input signal type.

    Formula:
    --------
        erb = 21.4 * log10(1 + hz * 0.00437)
    """
    if cp and isinstance(hz, cp.ndarray):
        return 21.4 * cp.log10(1 + hz * 0.00437)
    return 21.4 * np.log10(1 + hz * 0.00437)


def erbs_to_hz(erbs: ndarray|number) -> ndarray|number:
    """Convert a frequency value in the ERB scale to Hertz.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.

    Args:
        erbs (cp.ndarray | np.ndarray | float | int): The frequency value in the ERB scale to convert. Will take either an array or a single value.

    Returns:
        cp.ndarray|np.ndarray|float|int: The frequency value converted to Hertz. Return type matches the input signal type.

    Formula:
    --------
        hz = (10 ** (erb / 21.4) - 1) / 0.00437
    """
    return (10 ** (erbs / 21.4) - 1) / 0.00437


###############################################
# Waveform Generators
###############################################

def generate_sine_wave(frequency: int, duration: number, sample_rate: int=44100, magnitude: float=0.5) -> AudioSegment:
    """Generate a sine wave `AudioSegment` object.

    Args:
        frequency (int): The frequency of the sine wave in Hertz.
        duration (int | float): The duration of the sine wave in seconds.
        sample_rate (int, optional): The sample rate in samples per second. Default is 44100.
        magnitude (float, optional): The magnitude of the sine wave in the range [0,1]. Default is 0.5.

    Returns:
        AudioSegment: The generated sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = magnitude * np.sin(2 * np.pi * frequency * t)
    signal = np.int16(signal * MAX_AMPLITUDE_AT_16_BIT)
    audio = AudioSegment(
        signal.tobytes(),
        frame_rate=sample_rate,
        sample_width=signal.dtype.itemsize,
        channels=1
    )
    return audio


def generate_triangle_wave(frequency: int, duration: number, sample_rate: int=44100, magnitude: float=0.5) -> AudioSegment:
    """Generate a triangle wave `AudioSegment` object.

    Args:
        frequency (int): The frequency of the triangle wave in Hz.
        duration (int | float): The duration of the audio signal in seconds.
        sample_rate (int, optional): The sample rate of the audio signal in samples per second. Defaults to 44100.
        magnitude (float, optional): The peak amplitude of the triangle wave in the range [0,1]. Defaults to 0.5.

    Returns:
        AudioSegment: The generated triangle wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = magnitude * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - magnitude
    signal = np.int16(signal * MAX_AMPLITUDE_AT_16_BIT)
    audio = AudioSegment(
        signal.tobytes(),
        frame_rate=sample_rate,
        sample_width=signal.dtype.itemsize,
        channels=1
    )
    return audio


def generate_sawtooth_wave(frequency: int, duration: number, sample_rate: int=44100, magnitude: float=0.5) -> AudioSegment:
    """Generate a sawtooth wave audio signal.

    Args:
        frequency (int): The frequency of the sawtooth wave in Hertz.
        duration (int | float): The duration of the audio signal in seconds.
        sample_rate (int, optional): The sample rate of the audio signal in samples per second. Default is 44100.
        magnitude (float, optional): The peak amplitude of the sawtooth wave in the range [0,1]. Default is 0.5.

    Returns:
        AudioSegment: The generated sawtooth wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = magnitude * (t * frequency - np.floor(t * frequency + 0.5))
    signal = np.int16(signal * MAX_AMPLITUDE_AT_16_BIT)
    audio = AudioSegment(
        signal.tobytes(),
        frame_rate=sample_rate,
        sample_width=signal.dtype.itemsize,
        channels=1)
    return audio


def generate_square_wave(frequency: int, duration: number, sample_rate: int=44100, magnitude: float=0.5) -> AudioSegment:
    """Generate a square wave audio signal.

    Args:
        frequency (int): The frequency of the square wave in Hertz.
        duration (int | float): The duration of the audio signal in seconds.
        sample_rate (int, optional): The sample rate of the audio signal in samples per second. Default is 44100.
        magnitude (float, optional): The peak amplitude of the square wave. Default is 0.5.

    Returns:
        AudioSegment: The generated square wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = magnitude * sp.signal.square(2 * np.pi * frequency * t)
    signal = np.int16(signal * MAX_AMPLITUDE_AT_16_BIT)
    audio = AudioSegment(
        signal.tobytes(),
        frame_rate=sample_rate,
        sample_width=signal.dtype.itemsize,
        channels=1)
    return audio


def generate_white_noise(duration: number, sample_rate: int=44100, magnitude: float=0.5) -> AudioSegment:
    """
    Generate white noise audio segment.

    Args:
        duration (int | float): Duration of the white noise in seconds.
        sample_rate (int, optional): Sample rate of the audio in Hz. Default is 44100 Hz.
        magnitude (float, optional): The peak amplitude of the white noise. Default is 0.5.

    Returns:
        AudioSegment: The generated white noise.
    """
    # Randomly generated samples are the same as white noise
    # Thats why blanket radiation picked up by radio telescopes is interpreted as white noise
    signal = magnitude * np.random.uniform(low=-1.0, high=1.0, size=int(sample_rate * duration))
    signal = np.int16(signal * MAX_AMPLITUDE_AT_16_BIT)
    audio = AudioSegment(
        signal.tobytes(),
        frame_rate=sample_rate,
        sample_width=signal.dtype.itemsize,
        channels=1)
    return audio


def generate_pink_noise(duration: number, sample_rate: int=44100, magnitude: float=0.5) -> AudioSegment:
    """Generates pink noise audio.
    Pink noise is a signal with a frequency power spectrum that is inversely proportional to its frequency.
    This function generates pink noise by filtering white noise.

    Args:
        duration (int | float): Duration of the pink noise in seconds.
        sample_rate (int, optional): Sampling rate of the audio in Hz. Default is 44100 Hz.
        magnitude (float, optional): The peak amplitude of the generated pink noise. Default is 0.5.

    Returns:
        AudioSegment: The generated pink noise.
    """
    signal = np.random.randn(int(sample_rate * duration))
    signal = sp.signal.lfilter(*sp.signal.butter(1, 1/50, btype='low'), signal)
    signal = (signal * magnitude * MAX_AMPLITUDE_AT_16_BIT).astype(np.int16)
    audio = AudioSegment(
        signal.tobytes(),
        frame_rate=sample_rate,
        sample_width=signal.dtype.itemsize,
        channels=1)
    return audio


##########################################################
# Visualization Utilities
##########################################################

def plot_fourier_analysis(sound: AudioSegment, title: str='Fourier Analysis', group: bool=True) -> None:
    """Plot the Fourier Transform of an audio signal.
    Can group the frequencies into a smaller number of bins for easier visualization.
    
    Args:
        sound (AudioSegment): The audio signal to plot.
        title (str, optional): The title of the plot. Defaults to 'Fourier Analysis'.
        group_frequencies (bool, optional): Dictates if the frequencies should be grouped. Defaults to 'True'.
    """
    if sound.channels != 1: # If the audio signal has multiple channels, convert it to mono
        sound = sound.split_to_mono()[0]
    
    signal = np.array(sound.get_array_of_samples()) # Get the audio signal as an array
    signal = normalize_signal(signal) # Normalize the signal to the range [-1,1]
    frequencies, freq_signal = fourier_transform(signal, sound.frame_rate) # Calculate the Fourier Transform
    if group: # If the frequencies should be grouped
        frequencies, freq_signal = group_frequencies(frequencies, freq_signal) # Group the frequencies
    freq_signal = magnitude_to_db(freq_signal) # Convert the magnitudes to decibels

    plt.figure(figsize=(15, 5)) # Set the figure size
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    xticks = np.array([20, 100, 200, 400, 600, 1000, 2000, 4000, 6000, 10000, 20000]) # Set the x-ticks
    plt.xlim(hz_to_bark(20), hz_to_bark(20000)) # Set the x-axis limits to the Bark scale
    plt.xticks(hz_to_bark(xticks), xticks) # Convert the x-ticks to the Bark scale
    plt.ylim(-150, 0) # Set the y-axis limits
    plt.yticks([0, -20, -40, -60, -80, -100, -120, -140], [0, -20, -40, -60, -80, -100, -120, -140]) # Set the y-ticks
    plt.grid(True, which="both", axis='x', ls="--") # Add a grid to the plot
    plt.plot(hz_to_bark(frequencies), freq_signal) # Plot the Fourier Transform
    plt.fill_between(hz_to_bark(frequencies), freq_signal, -150, color='tab:blue', alpha=0.2) # Fill the area under the Fourier Transform
    plt.show()