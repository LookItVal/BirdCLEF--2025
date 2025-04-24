# Standard Library
import time
from typing import Optional

# Anaconda Libraries
import numpy as np
from IPython.display import display, HTML, DisplayHandle

number = int | float

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
