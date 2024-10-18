import time

def measure_runtime(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60
    print(f"Function runtime: {runtime_minutes:.2f} minutes")
    return result

# Example usage
def example_function():
    time.sleep(120)  # Simulate a function that takes 120 seconds to run

measure_runtime(example_function)