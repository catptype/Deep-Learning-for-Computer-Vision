import time

class TextProgressBar():
    """
    Simple text-based progress bar for tracking the progress of a process.

    Parameters:
        total_step (int): Total number of steps in the process.
        length (int): Length of the progress bar (default is 40).
        time_delay (float): Time delay in seconds for updating the progress bar (default is 0.1).

    Attributes:
        total_step (int): Total number of steps in the process.
        length (int): Length of the progress bar.
        time_delay (float): Time delay in seconds for updating the progress bar.
        cur_step (int): Current step in the process.
        start_time (float): Time when the progress bar was initiated.
        prev_time (float): Time of the previous progress bar update.

    Methods:
        add_step(num): Adds a specified number of steps to the progress bar.

    Example:
        ```python
        # Example usage of TextProgressBar class
        progress_bar = TextProgressBar(total_step=100, length=50, time_delay=0.05)
        for _ in range(100):
            progress_bar.add_step(1)
        ```
    """
    def __init__(self, total_step, length=40, time_delay=0.1):

        if not isinstance(total_step, int):
            raise ValueError("total_step must be an integer")
        self.total_step = total_step
        self.length = length
        self.time_delay = time_delay
        self.__cur_step = 0
        self.__start_time = time.time()
        self.__prev_time = self.__start_time
        
        self.__print_progress_bar()

    # Private method
    def __eta_calculation(self):
        # Initialize ETA
        elapsed_time = time.time() - self.__start_time
        if self.__cur_step > 0:
            initial_eta = (elapsed_time / self.__cur_step) * (self.total_step - self.__cur_step)
        else:
            initial_eta = 0

        # Post processing ETA for more accurate ETA
        step_time = time.time() - self.__prev_time
        if self.__cur_step < self.total_step:
            eta = int(initial_eta - step_time)
        else:
            eta = 0  # When all steps are completed, ETA is 0

        self.__prev_time = time.time()
        return max(eta, 0)  # Ensure that ETA is not negative
    
    def __print_progress_bar(self):
        eta = self.__eta_calculation()
        progress = (self.__cur_step / self.total_step)
        bars = '█' * int(self.length * progress)
        spaces = ' ' * (self.length - len(bars))
        print(f'\r{self.__cur_step}/{self.total_step} [{bars}{spaces}] {int(progress * 100)}% | ETA: {eta}s\t', end='', flush=True)
        time.sleep(self.time_delay)
    
    # Public method
    def add_step(self, num):
        self.__cur_step += num
        self.__print_progress_bar()