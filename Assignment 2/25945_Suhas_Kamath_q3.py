import numpy as np
import matplotlib.pyplot as plt

#Initial data
actual_data = ((1930,28.5),(1940,32.4),(1950,37.6),(1960,45.1),
                (1970,55.8),(1980,69.7),(1990,87.1),(2000,105.7),
                (2010,124.1),(2020,139.6))

def lagrange_coefficients(x, y):
    """
    Return coefficients (highest degree first) of the Lagrange interpolating
    polynomial passing through points (x[i], y[i]).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    # Start with the zero polynomial
    poly_coeffs = np.zeros(n)
    for i in range(n):
        # Construct L_i(x) basis polynomial
        Li = np.array([1.0])
        denom = 1.0
        xi = x[i]
        for j in range(n):
            if j == i:
                continue
            # Multiply by (x - xj)
            new_len = Li.size + 1
            new_Li = np.zeros(new_len, dtype=float)
            coeff = -x[j]
            for k in range(Li.size):
                new_Li[k] += Li[k]           # multiply by 1.0
                new_Li[k+1] += Li[k] * coeff # multiply by -x[j]
            Li = new_Li
            denom *= (xi - x[j])
        poly_coeffs = poly_coeffs + (y[i] / denom) * Li
    return poly_coeffs

if __name__ == "__main__":
    years = np.array([1930, 1940, 1950, 1960, 1970, 1980], dtype=float)
    population = np.array([28.5, 32.4, 37.6, 45.1, 55.8, 69.7], dtype=float)

    # Get polynomial coefficients in descending powers
    coeffs = lagrange_coefficients(years, population)

    # Print coefficients in descending powers: a0*x^n + a1*x^(n-1) + ... + an
    poly_str = " + ".join(f"{coeff:.6g}x^{len(coeffs)-1-i}" for i, coeff in enumerate(coeffs) if coeff != 0)
    print("Interpolating polynomial:", end=" ")
    print("P(x) =", poly_str)

    estimates = []

    # Evaluate polynomial from 1930 to 2020
    for year in range(1930, 2021, 10):
        # Evaluate polynomial at year
        population_estimate = 0.0

        for i in range(len(coeffs)):
            population_estimate += coeffs[i] * (year ** (len(coeffs) - 1 - i))
        
        estimates.append((year, population_estimate))

        print(f"Estimated population in {year}: {population_estimate:.6f} crores")

    #Plot estimated population
    years_plot, population_estimates = zip(*estimates)
    years_plot, actual_population = zip(*actual_data)
    plt.plot(years_plot, population_estimates, marker='o', label='Estimated')
    plt.plot(years_plot, actual_population, marker='o', label='Actual')
    plt.title("Population from 1930 to 2020")
    plt.xlabel("Year")
    plt.ylabel("Population (crores)")
    plt.grid()
    plt.legend()
    plt.show()

    # Calculate and print mean square error
    actual_population = np.array(actual_population)
    population_estimates = np.array(population_estimates)
    error_sum = 0.0

    for i in range(6,10):
        error_sum += (actual_population[i] - population_estimates[i]) ** 2

    error_sum /= 4

    print(f"Mean Square Error: {error_sum:.6f} crores")

    # Now evaluate polynomial for every year from 1930 to 2020
    # This is so we can plot a smooth curve
    estimates = []

    for year in range(1930, 2021):
        # Evaluate polynomial at year
        population_estimate = 0.0

        for i in range(len(coeffs)):
            population_estimate += coeffs[i] * (year ** (len(coeffs) - 1 - i))
        
        estimates.append((year, population_estimate))

        print(f"Estimated population in {year}: {population_estimate:.6f} crores")

    years_plot, population_estimates = zip(*estimates)
    years_plot_actual, actual_population = zip(*actual_data)
    plt.plot(years_plot, population_estimates, label='Estimated')
    plt.scatter(years_plot_actual, actual_population, color='red', label='Actual')
    plt.title("Population from 1930 to 2020")
    plt.xlabel("Year")
    plt.ylabel("Population (crores)")
    plt.grid()
    plt.legend()
    plt.show()

