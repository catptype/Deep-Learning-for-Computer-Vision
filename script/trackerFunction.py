import time

"""
TEMPLATE

def decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__} ... ", end="")
        func(*args, **kwargs)
        print("Complete")
       return wrapper
"""

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Execute {func.__name__} ... ", end="")
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Complete {elapsed_time:.6f} seconds.")
    return wrapper
    
def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__} ... ", end="")
        func(*args, **kwargs)
        print("Complete")
    return wrapper