import pandas as pd
import numpy as np
import librosa

from numpy import ndarray
from pydub import AudioSegment
from pydub.utils import mediainfo


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
        print(f'Format: {mediainfo(file)["format_name"]}')
        return None
    

def print_boundry(message: str="") -> None:
    """Print a message surrounded by a line of dashes with a total length of 100 characters
    
    Args:
        message (str, optional): The message to be printed. Prints an empty line if no message is provided.
    """
    length = (100 - len(message)) // 2
    print('-' * length, message, '-' * length)


def percentage_color_conversion(input_value: number, input_min: int=0, input_max: int=100, output_min: int=0, output_max: int=110) -> float:
    """Convert a percentage value to a color in the range of red to green.
    The input value is mapped to a non-linear scale to create a more pronounced color change effect.
    The output value is in the range of 0 to 110, which corresponds to the hue values in an HSV color space for red to green.

    Args:
        input_value (int | float): The value to be converted.
        input_min (int, optional): The minimum value of the input range. Defaults to 0.
        input_max (int, optional): The maximum value of the input range. Defaults to 100.
        output_min (int, optional): The minimum value of the output range. Defaults to 0.
        output_max (int, optional): The maximum value of the output range. Defaults to 110.

    Returns:
        float: The converted value within the output range.
    """
    normalized_input = (input_value - input_min) / (input_max - input_min) # Normalize the input value to a range of 0 to 1
    scaled_value = (np.exp(normalized_input) - 1) / (np.e - 1) # Apply an exponential function to create a non-linear scale
    scaled_value = (np.exp(scaled_value) - 1) / (np.e - 1) # Repeat the function to create a more pronounced effect
    return output_min + scaled_value * (output_max - output_min) # Map the scaled value to the output range


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
    if isinstance(signal, cp.ndarray): # If the signal is a cupy array, convert it to a numpy array
        signal = cp.asnumpy(signal)
    if denormalize: # If the signal should be denormalized, scale it to the range of int16
        signal = signal * MAX_AMPLITUDE_AT_16_BIT
    signal = signal.astype(np.int16) # Convert the signal to a 16-bit integer array
    return AudioSegment(signal.tobytes(), frame_rate=sample_rate, sample_width=signal.dtype.itemsize, channels=1) # Create an audio segment object from the signal array


def bytes_to_human_readable(n: int, depth: int=2) -> str:
    """Convert a byte value into a human-readable string with appropriate units.

    Args:
        n (int): The number of bytes.
        depth (int, optional): The number of decimal places to include in the output. Default is 2.

    Returns:
        str: A string representing the byte value in a human-readable format with units (B, KB, MB, GB, TB, PB).
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.{depth}f} {unit}"
        n /= 1024
    return f"{n:.{depth}f} PB"


def time_to_human_readable(seconds: number, depth: int=3) -> str:
    """Convert a time duration in seconds to a human-readable string format.

    Args:
        seconds (int|float): The time duration in seconds.
        depth (int): The level of detail for the output format. 
                    - 0 or 1: Returns seconds (with milliseconds if `seconds` is a float).
                    - 2: Returns minutes and seconds.
                    - 3: Returns hours, minutes, and seconds.

    Returns:
        str: The time duration in a human-readable string format.
    """
    if depth == 0 or depth == 1: # If the depth is 0 or 1, return the seconds with or without milliseconds
        if isinstance(seconds, int): # If the seconds is an integer, return the number of seconds in 'x seconds' format
            return f'{seconds} seconds'
        # If the seconds is a float, return the time in 'xx:xxx' format with seconds and milliseconds
        miliseconds = (seconds - int(seconds)) * 1000
        return f'{int(seconds):02}:{int(miliseconds):03}'
    
    if depth == 2: # If the depth is 2, return the time in 'xx:xx' format with minutes and seconds
        minutes, seconds = divmod(seconds, 60)
        return f'{int(minutes):02}:{int(seconds):02}'
    
    # If the depth anything other than 0, 1 or 2 return the time in 'xx:xx:xx' format with hours, minutes, and seconds
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'


def memory_usage(df: pd.DataFrame) -> int:
    """Calculate the total memory usage of a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to calculate the memory usage of.

    Returns:
        int: The total memory usage of the DataFrame in bytes.
    """
    return df.memory_usage(deep=True).sum()


def print_memory_usage(df: pd.DataFrame) -> None:
    """Prints the memory usage of a given pandas DataFrame in a human-readable format.

    Args:
        df (pd.DataFrame): The DataFrame for which to calculate and print memory usage.
    """
    print(f"Memory Usage: {bytes_to_human_readable(memory_usage(df))}")


def calculate_time_remaining(start_time: float, iteration: int, length: int) -> str:
    """Estimates the time remaining for a process.

    Args:
        start_time (float): The start time of the process in seconds since the epoch (returned from `time.time()`).
        iteration (int): The current iteration number.
        length (int): The total number of iterations.

    Returns:
        str: The estimated time remaining in a human-readable format.
    """
    elapsed_time = time.time() - start_time
    iterations_remaining = length - iteration
    time_per_iteration = 0
    try:
        time_per_iteration = elapsed_time / iteration
    except ZeroDivisionError:
        pass # Ignore division by zero error from 0th iteration
    time_remaining = iterations_remaining * time_per_iteration
    time_remaining_str = time_to_human_readable(time_remaining)
    return time_remaining_str


def print_progress_bar(iteration: int, 
                       max_iteration: int, 
                       message: str='', 
                       display_handler: Optional[DisplayHandle]=None, 
                       treat_as_data: bool=False,
                       treat_as_time: bool=False
                       ) -> DisplayHandle:
    """Print a progress bar that updates in place.

    Args:
        iteration (int): Current iteration or progress value.
        max_iteration (int): Maximum iteration or maximum value for the progress.
        display_handler (DisplayHandle, optional): An existing display handler to update the progress bar. Defaults to None where a new display handler is generated. 
        message (str, optional): Additional message to display alongside the progress bar.
        treat_as_data (bool, optional): If True, converts iteration and length to human-readable data format. Defaults to False.

    Returns:
        DisplayHandle: The display handler used to render the progress bar.
                         - **In order to update the progress bar, pass this handler to the next call of this function.**
    
    Notes:
        This function is intended for use in Jupyter notebooks, and uses the IPython display module to render the progress bar as HTML.
        That HTML is rendered in place, allowing the progress bar to update in place.
    """
    progress = int((iteration / max_iteration) * 100) # Calculate progress as a whole number percentage

    if treat_as_data: # Convert the iteration and length to human-readable data format if specified
        iteration: str = bytes_to_human_readable(iteration)
        max_iteration: str = bytes_to_human_readable(max_iteration)
    if treat_as_time: # Convert the iteration and length to human-readable time format if specified
        iteration: str = time_to_human_readable(iteration)
        max_iteration: str = time_to_human_readable(max_iteration)

    # Get the color for the progress bar based on the percentage value using css hsl color format
    color = f'hsl({percentage_color_conversion(progress)}, 100%, 50%)'
    # Create the progress bar as a string of equal signs with the appropriate color
    # This is formatted in html to allow for color styling
    bar = f'<span style="color: {color};">{"=" * progress}</span>{" " * (100 - progress)}'
    style = 'font-family: monospace; font-size: 13px;' # Ensure monospacing for consistent character width
    # Create the message to display text in html format
    html_string = f'<pre style="{style}">[{bar}] - {iteration}/{max_iteration}'
    if message: # Add the message to the html string if provided
        html_string += f' - {message}'
    html_string += '</pre>'

    # display the message as HTML, updating the display handler if provided
    if not display_handler:
        display_handler = display(HTML(html_string), display_id=True)
    else:
        display_handler.update(HTML(html_string))
    return display_handler # Return the display handler for updating the progress bar


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


def plot_histogram(column: pd.Series, title: str='Histogram', bins: int=25) -> None:
    """Plot a histogram of the values in a column of a pandas DataFrame.
    Column must be numeric.

    Args:
        column (pd.Series): The column to plot.
        title (str, optional): The title of the plot. Defaults to 'Histogram'.
        bins (int, optional): The number of bins to use in the histogram. Defaults to 10.
    """
    if column.dtype not in [np.float64, np.int64]:
        raise ValueError('Column must be numeric')
    plt.hist(column, bins=bins)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y')
    plt.show()


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
    if isinstance(signal, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array
    if max_amplitude is None: # If the maximum amplitude is not provided find the maximum amplitude in the signal
        max_amplitude = math.max(math.abs(signal))
    return signal / max_amplitude # Normalize the signal to the range [-1,1]


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
    if isinstance(signal, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array
    return 20 * math.log10(math.abs(signal + epsilon)) # Calculate the magnitude in decibels


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
    if isinstance(hz, cp.ndarray):
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
    if isinstance(hz, cp.ndarray):
        return 13 * cp.arctan(0.00076 * hz) + 3.5 * cp.arctan((hz / 7500) ** 2)
    return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500) ** 2)


CPU_BARK_KEY = np.arange(0, 20000, 1) # Generate a range of frequencies from 0 to 20000 Hz
CPU_BARK_KEY = hz_to_bark(CPU_BARK_KEY) # Convert the frequencies to the Bark scale
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
    if isinstance(bark, cp.ndarray):
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
    if isinstance(bark, cp.ndarray):
        return cp.digitize(bark, GPU_BARK_KEY) - 1
    return np.digitize(bark, CPU_BARK_KEY) - 1
    

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
    if isinstance(hz, cp.ndarray):
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


def signal_to_time(signal: ndarray, sampling_rate: int) -> ndarray:
    """Calculate the time values for each sample in an audio signal.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.

    Args:
        signal (cp.ndarray | np.ndarray): The audio signal to calculate the time values for.
        sampling_rate (int): The sampling rate of the audio signal.
    
    Returns:
        cp.ndarray|np.ndarray: The time values for each sample in the audio signal. Return type matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array
    return math.arange(signal.size) / sampling_rate # Calculate the time values for each sample


def plot_waveform(sound: AudioSegment, title: str='Waveform', normalization_constant: int=None) -> None:
    """Plot the waveform of an audio signal.
    
    Args:
        sound (AudioSegment): The audio signal to plot.
        title (str, optional): The title of the plot. Defaults to 'Waveform'.
        normalization_constant (int, optional): The constant to normalize the signal by. Defaults to None.
    """
    if sound.channels > 1: # If the audio signal has multiple channels, convert it to mono
        sound = sound.split_to_mono()[0]

    signal = np.array(sound.get_array_of_samples()) # Get the audio signal as an array
    signal = normalize_signal(signal, normalization_constant) # Normalize the signal to the range [-1,1]
    time = signal_to_time(signal, sound.frame_rate) # Calculate the time values for each sample
    plt.figure(figsize=(15, 3)) # Long figure to show the RMS signal clearly
    plt.plot(time, signal) # Plot the waveform
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (dB)')
    plt.axhline(0, color='black', linewidth=0.5) # Add a horizontal black centerline
    yticks = np.linspace(-1, 1, 15) # Add y-ticks
    yticks_labels = [f'{int(magnitude_to_db(y))}' for y in yticks] # Convert the y-ticks to decibels
    yticks_labels[7] = '-∞' # Replace the center y-tick with negative infinity
    plt.ylim(-1, 1)
    plt.yticks(yticks, yticks_labels)
    plt.grid(True, axis='x')
    xticks = np.linspace(0, time[-1], 10) # Add x-ticks
    depth = 2 if time[-1] > 10 else 1 # Set the depth of the time values
    xticks_labels = [time_to_human_readable(x, depth=depth) for x in xticks] # Convert the x-ticks to human readable time
    plt.xlim(0, time[-1])
    plt.xticks(xticks, xticks_labels)
    plt.show()


def signal_to_rms_signal(signal: ndarray, sample_rate: int, window_size: int=300, centered: bool=False) -> ndarray:
    """Calculate the Root Mean Square (RMS) of an audio signal. The RMS is calculated over a moving window of the signal.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.

    Args:
        signal (cp.ndarray | np.ndarray): The audio signal to calculate the RMS of.
        sample_rate (int): The sample rate of the audio signal.
        window_size (int, optional): The size of the moving window in milliseconds. Defaults to 300.
        centered (bool, optional): If True, the RMS is calculated at the center of the window. This is abnormal. Defaults to False.
    
    Returns:
        cp.ndarray|np.ndarray: The RMS of the audio signal. Return type matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array
    window_size = int(window_size * sample_rate / 1000) # Convert the window size to samples

    pad_width = window_size // 2 if centered else window_size # Calculate the padding width
    padded_signal = math.pad(signal, pad_width, mode='constant') # Pad the area around the signal making it longer

    cumsum = math.cumsum(padded_signal ** 2) # Calculate the cumulative sum of the squared signal
    cumsum = cumsum[window_size:] - cumsum[:-window_size] # Calculate the difference between the cumulative sums to get the sum of the squared signal in the window
    rms_signal = math.sqrt(cumsum / window_size) # Calculate the RMS of the signal

    if signal.size > rms_signal.size: # Add 1 element of padding to the end of the RMS signal with the value 0
        rms_signal = math.concatenate((rms_signal, math.array([0])))
    del padded_signal # Delete the padded signal to free up memory
    return rms_signal[:signal.size] # Remove the padding from the RMS signal if needed

def plot_rms_signal(sound: AudioSegment, title: str='RMS Signal', normalization_constant: int=None) -> None:
    """Plots the Root Mean Square (RMS) of an audio signal over time.
    
    Args:
        sound (AudioSegment): The audio signal to plot.
        title (str, optional): The title of the plot. Defaults to 'RMS Signal'.
        normalization_constant (int, optional): The constant to normalize the signal by. Defaults to None.
    """
    if sound.channels > 1: # If the audio signal has multiple channels, convert it to mono
        sound = sound.split_to_mono()[0]

    signal = np.array(sound.get_array_of_samples()) # Get the audio signal as an array
    signal = normalize_signal(signal, normalization_constant) # Normalize the signal to the range [-1,1]
    rms_signal = signal_to_rms_signal(signal, sound.frame_rate) # Calculate the RMS of the signal
    time = signal_to_time(rms_signal, sound.frame_rate) # Calculate the time values for each RMS value
    plt.figure(figsize=(15, 3)) # Long figure to show the RMS signal clearly
    plt.plot(time, rms_signal) # Plot the RMS signal
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('RMS Amplitude (dB)')
    plt.grid(True, axis='x')
    xticks = np.linspace(0, time[-1], 10) # Add x-ticks
    depth = 2 if time[-1] > 10 else 1
    xticks_labels = [time_to_human_readable(x, depth=depth) for x in xticks] # Convert the x-ticks to human readable time
    plt.xlim(0, time[-1])
    plt.xticks(xticks, xticks_labels)
    yticks = np.linspace(0, 1, 7) # Add y-ticks. If the number of ticks is too high it will show 0 twice.
    yticks_labels = [f'{int(magnitude_to_db(y))}' for y in yticks]  # Convert the y-ticks to decibels
    yticks_labels[0] = '-∞' # Replace the first y-tick with negative infinity
    plt.ylim(0, 1)
    plt.yticks(yticks, yticks_labels)
    plt.fill_between(time, rms_signal, 0, color='tab:blue', alpha=0.5) # Fill the area under the RMS signal
    plt.show()


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
    if isinstance(signal, cp.ndarray):
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
    
    if isinstance(signal, cp.ndarray) and (cache is False): # If the operation is on the GPU and cache is not requested
        with fft.get_fft_plan(signal, value_type='R2C'): # Create the FFT plan
            return rfft(signal, sample_rate) # Calculate the Fourier Transform using the plan
    else:
        return rfft(signal, sample_rate) # Calculate the Fourier Transform


def polar_to_complex(magnitudes: ndarray|number, phases: ndarray|number) -> ndarray|complex:
    """Convert polar coordinates to complex numbers using Euler's formula.
    Can convert either a single value or an array of values.
    Can perform the conversion on either the CPU or GPU, depending on the input signal type.
    Does not check if both inputs are the same type or shape.

    Args:
        magnitudes (cp.ndarray | np.ndarray | float | int): The magnitudes of the complex numbers.
        phases (cp.ndarray | np.ndarray | float | int): The phases of the complex numbers.
    
    Returns:
        cp.ndarray|np.ndarray|complex: The complex numbers. Return type matches the input signal type.
    """
    if isinstance(magnitudes, cp.ndarray):
        return magnitudes * cp.exp(1j * phases) # In python, j is the imaginary unit
    return magnitudes * np.exp(1j * phases)     # This follows the convention used in electrical engineering


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
    if isinstance(frequencies, cp.ndarray):
        math = cp # Use cupy for GPU operations if the signal is a cupy array

    if freq_bins is None: # If the frequency bins are not provided
        freq_bins = get_bark_spaced_frequencies(num_groups, math.__name__) # Get the Bark spaced frequencies
    bin_indicies = math.digitize(frequencies, freq_bins) - 1 # Group the frequencies into bins
    reduced_freq_signal = math.zeros(len(freq_bins)) # Initialize an array to store the maximum magnitude in each bin
    for i in range(len(freq_bins)): # Iterate through each bin
        bin_values = freq_signal[bin_indicies == i] # Get the values in the bin
        reduced_freq_signal[i] = math.sqrt(math.mean(math.abs(bin_values) ** 2)) # Calculate the RMS of the bin
    return freq_bins, reduced_freq_signal # Return the grouped frequencies and maximum magnitudes


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


def short_time_fourier_transform(signal: ndarray, sample_rate: int, nperseg_constant: int=10, noverlap_constant: int=2) -> Tuple[ndarray, ndarray, ndarray]:
    """Calculate the Short Time Fourier Transform (STFT) of an audio signal.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.

    Args:
        signal (cp.ndarray | np.ndarray): The audio signal to calculate the STFT of.
        sample_rate (int): The sample rate of the audio signal.
        nperseg_constant (int, optional): The constant to determine the window size used in the formula nperseg = sample_rate / nperseg_constant. Defaults to 10.
        noverlap_constant (int, optional): The constant to determine the overlap size used in the formula noverlap = nperseg / noverlap_constant. Defaults to 2.
    
    Returns
    -------
    - **cp.ndarray|np.ndarray:** The frequencies of the STFT. Matches the input signal type.
    - **cp.ndarray|np.ndarray:** The time values of the STFT. Matches the input signal type.
    - **cp.ndarray|np.ndarray:** The complex time-frequency signal from the STFT. Matches the input signal type.
    """
    sig = sp.signal # Use scipy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        sig = cusig # Use CuPy for GPU operations if the signal is a cupy array
    nperseg = sample_rate // nperseg_constant # Calculate the window size
    noverlap = nperseg // noverlap_constant # Calculate the overlap size

    return sig.stft(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap) # Calculate the STFT


def inverse_short_time_fourier_transform(signal: ndarray, sample_rate: int, nperseg_constant: int=10, noverlap_constant: int=2) -> ndarray:
    """Calculate the Inverse Short Time Fourier Transform (ISTFT) of an audio signal.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.

    Args:
        signal (cp.ndarray | np.ndarray): The audio signal to calculate the ISTFT of.
        sample_rate (int): The sample rate of the audio signal.
        nperseg_constant (int, optional): The constant to determine the window size used in the formula nperseg = sample_rate / nperseg_constant. Defaults to 10.
        noverlap_constant (int, optional): The constant to determine the overlap size used in the formula noverlap = nperseg / noverlap_constant. Defaults to 2.
    
    Returns:
    cp.ndarray|np.ndarray: The audio signal after applying the ISTFT. Matches the input signal type.
    """
    sig = sp.signal # Use scipy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        sig = cusig # Use CuPy for GPU operations if the signal is a cupy array
    nperseg = sample_rate // nperseg_constant # Calculate the window size
    noverlap = nperseg // noverlap_constant # Calculate the overlap size

    return sig.istft(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[1] # Calculate the ISTFT. The 0th value is the time values, the 1st value is the audio signal.


def group_time_frequencies(frequencies: ndarray,
                           time_freq_signal: ndarray,
                           num_groups: int=100,
                           ) -> Tuple[ndarray, ndarray]:
    """Group the frequency values of a Short Time Fourier Transform into a smaller array.
    Each group is spaced evenly along the Bark Scale. The time values are kept the same.
    GPU operations are quite slow so this function converts the arrays to numpy arrays before processing.
    
    Args:
        frequencies (cp.ndarray | np.ndarray): The frequencies of the Short Time Fourier Transform.
        time_freq_signal (cp.ndarray | np.ndarray): The resulting complex signal from the Short Time Fourier Transform.
        num_groups (int, optional): The number of frequency groups to create. Defaults to 100.
    
    Returns
    -------
    - **cp.ndarray|np.ndarray:** The grouped frequencies. Matches the input signal type.
    - **cp.ndarray|np.ndarray:** The magnitude value for the bin. Matches the input signal type.
    """
    to_cupy = False # Marker to reconvert the arrays to cupy arrays
    if isinstance(frequencies, cp.ndarray):
        to_cupy = True # If the arrays are cupy arrays, mark them for conversion
        frequencies = frequencies.get()
        time_freq_signal = time_freq_signal.get()

    freq_bins = get_bark_spaced_frequencies(num_groups, np.__name__) # Get the Bark spaced frequencies
    reduced_time_freq_signal = np.zeros((num_groups, time_freq_signal.shape[1])) # Initialize the reduced time-frequency signal array
    for i in range(time_freq_signal.shape[1]): # Iterate over each column (time slice)
        freq_bins, reduced_time_freq_signal[:, i] = group_frequencies(frequencies, time_freq_signal[:, i], freq_bins, num_groups) # Group the frequencies for each time slice
    
    if to_cupy:
        return cp.array(freq_bins), cp.array(reduced_time_freq_signal) # Convert the arrays back to cupy arrays
    return freq_bins, reduced_time_freq_signal # Return the grouped frequencies and maximum magnitudes


def plot_spectrogram(sound: AudioSegment, title: str='Spectrogram', group: bool=True) -> None:
    """Plots the spectrogram of an audio signal from the results of a Short Time Fourier Transform.
    
    Args:
        sound (AudioSegment): The audio signal to plot.
        title (str, optional): The title of the plot. Defaults to 'Spectrogram'.
        group (bool, optional): Dictates if and how the frequencies should be grouped. Defaults to True.
    """
    if sound.channels != 1: # If the audio signal has multiple channels, convert it to mono
        sound = sound.split_to_mono()[0]

    signal = np.array(sound.get_array_of_samples()) # Get the audio signal as an array
    signal = normalize_signal(signal) # Normalize the signal to the range [-1,1]
    frequencies, times, time_freq_signal = short_time_fourier_transform(signal, sound.frame_rate) # Calculate the STFT
    if group: # If the frequencies should be grouped
        frequencies, time_freq_signal = group_time_frequencies(frequencies, time_freq_signal) # Group the frequencies
    time_freq_signal = magnitude_to_db(time_freq_signal) # Convert the magnitudes to decibels

    plt.figure(figsize=(15, 5)) # Set the figure size   
    plt.title(title)
    plt.pcolormesh(times, hz_to_bark(frequencies), time_freq_signal, cmap='plasma') # Plot the spectrogram
    yticks = np.array([20, 200, 500, 1000, 2000, 5000, 10000, 20000]) # Set the y-ticks 
    plt.ylim(hz_to_bark(20), hz_to_bark(20000)) # Set the y-axis limits to the Bark scale
    plt.yticks(hz_to_bark(yticks), yticks) # Convert the y-ticks to the Bark scale
    plt.ylabel('Frequency (Hz)')
    xticks = np.linspace(0, times[-1], 10) # Set the x-ticks
    depth = 2 if times[-1] > 10 else 1
    xticks_labels = [time_to_human_readable(x, depth=depth) for x in xticks] # Convert the x-ticks to human readable time
    plt.xlim(0, times[-1]) # Set the x-axis limits
    plt.xticks(xticks, xticks_labels)
    plt.xlabel('Time')
    plt.colorbar(label='Amplitude (dB)')
    plt.clim(-120, 0) # Set the color limits
    plt.show()


def hilbert_transform(signal: ndarray) -> ndarray:
    """Calculate the analytic signal of a real valued signal using the Hilbert Transform.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.

    Args:
        signal (cp.ndarray | np.ndarray): The signal to calculate the analytic signal of.
    
    Returns:
        cp.ndarray|np.ndarray: The analytic signal. Return type matches the input signal type.
    """
    sig = sp.signal # Use scipy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        sig = cusig # Use CuPy for GPU operations if the signal is a cupy array
    return sig.hilbert(signal) # Calculate the analytic signal


def plot_analytic_envelope(sound: AudioSegment, title: str='Analytic Envelope', emphasis: int=1, normalization_constsnt: Optional[int]=None) -> None:
    """Plot the Analytic Envelope of an audio signal using the Hilbert Transform.
    
    Args:
        sound (AudioSegment): The audio signal to plot.
        title (str, optional): The title of the plot. Defaults to 'Analytic Envelope'.
        emphasis (int, optional): The line width of the envelope. Defaults to 1.
        normalization_constant (int, optional): The constant to normalize the signal by. Defaults to None.
    """
    if sound.channels != 1: # If the audio signal has multiple channels, convert it to mono
        sound = sound.split_to_mono()[0]

    signal = np.array(sound.get_array_of_samples()) # Get the audio signal as an array
    signal = normalize_signal(signal, normalization_constsnt) # Normalize the signal to the range [-1,1]
    analytic_signal = hilbert_transform(signal) # Calculate the analytic signal
    envelope = np.abs(analytic_signal) # Calculate the envelope of the analytic signal
    time = signal_to_time(signal, sound.frame_rate) # Calculate the time values for each sample

    plt.figure(figsize=(15, 3)) # Long figure to show the RMS signal clearly
    plt.plot(time, signal, label='Signal', color='tab:blue') # Plot the original signal
    plt.plot(time, envelope, label='Envelope', color='tab:pink', alpha=0.9, linewidth=emphasis) # Plot the envelope
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (dB)')
    plt.axhline(0, color='black', linewidth=0.5) # Add a horizontal black centerline
    yticks = np.linspace(-1.2, 1.2, 17) # Add y-ticks
    yticks_labels = [f'{magnitude_to_db(y):.0f}' for y in yticks] # Convert the y-ticks to decibels and round to the nearest whole number
    yticks_labels[8] = '-∞' # Replace the center y-tick with negative infinity
    plt.ylim(-1.2, 1.2)
    plt.yticks(yticks, yticks_labels)
    plt.grid(True, axis='x')
    xticks = np.linspace(0, time[-1], 10) # Add x-ticks
    depth = 2 if time[-1] > 10 else 1 # Set the depth of the time values
    xticks_labels = [time_to_human_readable(x, depth=depth) for x in xticks] # Convert the x-ticks to human readable time
    plt.xlim(0, time[-1])
    plt.xticks(xticks, xticks_labels)
    plt.legend()
    plt.show()


def normalize_decibels(signal: ndarray) -> ndarray:
    """Normalize the decibel values of a signal to the range [0,1] from -120 to 0 db.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.
    
    Args:
        signal (cp.ndarray | np.ndarray): The signal to normalize.
    
    Returns:
        cp.ndarray|np.ndarray: The normalized signal. Return type matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        math = cp # Use CuPy for GPU operations if the signal is a cupy array

    signal = math.clip(signal, -120, 0) # Clip the signal to the range [-120,0]
    signal = signal + 120 # Shift the signal to the range [0,120]
    signal = signal / 120 # Normalize the signal to the range [0,1]
    return signal


def reduce_noise_floor(signal: ndarray, sample_rate: int, noise_floor_percentile: number=2.5, reduction_coefficient: number=7.5) -> ndarray:
    """Reduce the noise floor of a signal by determining the noise floor using the Short Time Fourier Transform.
    Can perform the calculation on either the CPU or GPU, depending on the input signal type.

    Method
    ------
    1. Calculate the Short Time Fourier Transform of the audio signal.
    2. Determine the noise floor by taking a low percentile of the magnitudes in each frequency bin.
    3. Subtract the noise floor from the magnitudes. Only the magnitude should be reduced, which means the complex value must be recomputed using the phase.
    4. Calculate the Inverse Short Time Fourier Transform on the modified time-frequency signal to get the new real time signal.

    Args:
        signal (cp.ndarray | np.ndarray): The signal to reduce the noise floor of.
        sample_rate (int): The sample rate of the signal.
        noise_floor_percentile (number, optional): The percentile of the magnitudes to use as the noise floor. Defaults to 2.5.
        reduction_coefficient (number, optional): The coefficient to reduce the noise floor by. Defaults to 10.
    
    Returns:
        cp.ndarray|np.ndarray: The signal with the noise floor reduced. Return type matches the input signal type.
    """
    math = np # Use numpy for CPU operations by default
    if isinstance(signal, cp.ndarray):
        math = cp # Use CuPy for GPU operations if the signal is a cupy array

    nperseg_constant = 100 # Set the constant to determine the window size
    noverlap_constant = 2 # Set the constant to determine the overlap size
        
    _, _, Zxx = short_time_fourier_transform(signal, sample_rate, nperseg_constant, noverlap_constant) # Calculate the Short Time Fourier Transform with high time resolution
    noise_floor = math.zeros(Zxx.shape[0]) # Initialize an array to store the noise floor
    for i in range(Zxx.shape[0]): # Iterate through each frequency bin
        # Calculate the noise floor as a percentile of the magnitudes. (min value is always 0, so a low percentile is used instead)
        noise_floor[i] = math.percentile(math.abs(Zxx[i]), noise_floor_percentile) 
        noise_floor[i] *= reduction_coefficient # Reduce the noise floor by the reduction coefficient to adjust the amount of reduction applied.
    for i in range(Zxx.shape[1]): # Iterate through each time slice
        # Subtract the noise floor from the magnitudes and recompute the complex values with the phase (minimum value magnitude is 0)
        Zxx[:, i] = polar_to_complex(math.maximum(math.abs(Zxx[:, i]) - noise_floor, 0), math.angle(Zxx[:, i]))
    del noise_floor # Clear the noise floor from memory
    return inverse_short_time_fourier_transform(Zxx, sample_rate, nperseg_constant, noverlap_constant) # Calculate the Inverse Short Time Fourier Transform to get the noise reduced signal


def split_around_silence(sounds: AudioSegment, size: number=5, threshold_rms: int=3) -> List[AudioSegment]:
    """Split an audio signal into segments around silence.
    
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