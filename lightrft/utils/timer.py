"""
Timer Module for Performance Measurement

This module provides a comprehensive timing utility for measuring execution time of different
code sections. It's particularly useful for profiling machine learning training loops,
evaluation processes, and any code that requires performance monitoring.

The module includes a Timer class that can be used both as a context manager and through
class methods for flexible timing operations. It supports CUDA synchronization for accurate
GPU timing and provides step-wise statistics reporting for monitoring performance across
training iterations.

Example::

    # Using as context manager
    with Timer("data_loading"):
        # Your data loading code here
        pass

    # Using class methods
    Timer.start("forward_pass")
    # Your forward pass code
    Timer.stop("forward_pass")

    # Print statistics and reset
    Timer.step()
"""

import time
from collections import defaultdict

import torch


class Timer:
    """
    A simple timer class for measuring the execution time of different code sections.

    This class provides functionality to start, stop, and record the elapsed time for
    named timers. It also supports printing timing statistics at each step, which is
    particularly useful for monitoring training or evaluation loops.

    The Timer class maintains global state across all instances, allowing you to collect
    timing data from multiple parts of your code and then view aggregated statistics.
    It automatically handles CUDA synchronization when available to ensure accurate
    timing measurements on GPU operations.

    :param _timers: A dictionary to store the total elapsed time for each named timer.
    :type _timers: defaultdict(float)
    :param _counts: A dictionary to store the number of times each named timer has been called.
    :type _counts: defaultdict(int)
    :param _current_times: A dictionary to store the start time of the currently active timers.
    :type _current_times: dict
    :param _current_step: A counter for the current step, used for printing step-wise statistics.
    :type _current_step: int

    Example::

        # Method 1: Using as context manager
        with Timer("data_processing"):
            process_data()

        # Method 2: Using start/stop methods
        Timer.start("model_forward")
        output = model(input)
        Timer.stop("model_forward")

        # Print statistics for current step
        Timer.step()
    """

    _timers = defaultdict(float)
    _counts = defaultdict(int)
    _current_times = {}
    _current_step = 0

    def __init__(self, name: str):
        """
        Initializes a Timer instance.

        :param name: The name of this specific timer instance. This name will be
            used when entering and exiting the context manager.
        :type name: str

        Example::

            timer = Timer("my_operation")
            with timer:
                # Code to time
                pass
        """
        self.name = name
        # __enter__ and __exit__ will use self.name

    @classmethod
    def _cuda_sync_if_available(cls):
        """
        Synchronizes CUDA devices if available.

        This ensures accurate timing on GPUs by waiting for all CUDA operations to complete.
        Without synchronization, GPU operations may be asynchronous and timing measurements
        could be inaccurate.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def start(cls, name: str):
        """
        Starts a timer with the given name.

        If a timer with the same name is already running, its start time will be overwritten.
        This method automatically handles CUDA synchronization if CUDA is available to ensure
        accurate timing measurements.

        :param name: The name of the timer to start.
        :type name: str

        Example::

            Timer.start("data_loading")
            # Your code here
            Timer.stop("data_loading")
        """
        cls._cuda_sync_if_available()
        cls._current_times[name] = time.time()

    @classmethod
    def stop(cls, name: str):
        """
        Stops a timer with the given name and records the elapsed time.

        The elapsed time is added to the total time for this timer, and the call count is incremented.
        If the timer was not started, a warning message will be printed.

        :param name: The name of the timer to stop.
        :type name: str

        Example::

            Timer.start("computation")
            result = heavy_computation()
            Timer.stop("computation")
        """
        cls._cuda_sync_if_available()
        if name in cls._current_times:
            elapsed = time.time() - cls._current_times[name]
            cls._timers[name] += elapsed
            cls._counts[name] += 1
            del cls._current_times[name]
        else:
            print(f"Warning: Timer '{name}' was stopped without being started.")

    @classmethod
    def step(cls):
        """
        Prints the timing statistics for the current step and resets the timers.

        If distributed training is used, it only prints the statistics on the main process (rank 0).
        The statistics include the total time, average time, and number of calls for each timer
        recorded since the last step. After printing, all timer data is cleared to start fresh
        for the next step.

        This method is particularly useful in training loops where you want to monitor
        performance on a per-epoch or per-batch basis.

        Example::

            for epoch in range(num_epochs):
                with Timer("epoch"):
                    for batch in dataloader:
                        with Timer("forward"):
                            output = model(batch)
                        with Timer("backward"):
                            loss.backward()
                Timer.step()  # Print and reset timers for this epoch
        """
        cls._current_step += 1
        is_main_process = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main_process = torch.distributed.get_rank() == 0

        if is_main_process:
            print(f"\n--- Step {cls._current_step} Timing Statistics ---")
            if not cls._timers:
                print("  No timers recorded for this step.")
            else:
                for name in sorted(cls._timers.keys()):
                    total_time = cls._timers[name]
                    count = cls._counts[name]
                    avg_time = total_time / count if count > 0 else 0
                    print(f"  {name}: total={total_time:.4f}s, avg={avg_time:.4f}s, calls={count}", flush=True)
        cls._timers.clear()
        cls._counts.clear()
        cls._current_times.clear()

    def __enter__(self):
        """
        Starts the timer when entering the context.

        This method is called when using the Timer as a context manager with the 'with' statement.
        It starts timing for the timer name specified during initialization.

        :return: The Timer instance itself.
        :rtype: Timer

        Example::

            with Timer("my_operation") as timer:
                # Code to time
                pass
        """
        self.__class__.start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stops the timer when exiting the context.

        This method is called when exiting the context manager, regardless of whether
        an exception occurred. It stops timing for the timer name and records the elapsed time.

        :param exc_type: The exception type if an exception was raised, None otherwise.
        :param exc_val: The exception value if an exception was raised, None otherwise.
        :param exc_tb: The exception traceback if an exception was raised, None otherwise.
        :return: False to indicate that exceptions should not be suppressed.
        :rtype: bool
        """
        self.__class__.stop(self.name)
        return False
