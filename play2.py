# %%
def calculate_factors(n):
    """Return a list of factors of the given integer n."""
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:  # Check if i is a factor of n
            factors.append(i)
    return factors

def print_factor_pairs(n):
    """Print all pairs (a, b) such that a * b = n."""
    factors = calculate_factors(n)
    print(f"All pairs (a, b) such that a * b = {n}:")
    for a in factors:
        b = n // a
        print(f"({a}, {b})")

# Example usage
N = 150//2
print_factor_pairs(N)
# %%
