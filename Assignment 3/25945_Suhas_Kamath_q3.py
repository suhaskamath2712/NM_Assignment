import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Save directory
save_dir = Path(r"H:\My Drive\Numerical Methods\Assignments\Assignment 3")
save_dir.mkdir(parents=True, exist_ok=True)

# 16:9 figure size and high-resolution export
FIGSIZE = (16, 9)
SAVE_DPI = 1200

def f(x):
    return math.exp(x) * math.sin(x)

def fprime(x):
    return math.exp(x) * (math.sin(x) + math.cos(x))

def three_point_backward (x, h):
    return (3*f(x) - 4*f(x-h) + f(x - 2*h))/(2*h)

def three_point_forward(x,h):
    return (4*f(x+h)-f(x+2*h)-3*f(x))/(2*h)

def five_point_central(x,h):
    return (8*f(x+h)+f(x-2*h)-8*f(x-h)-f(x+2*h))/(12*h)


x = math.pi/4
step_sizes = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-7]

three_point_backward_results = [three_point_backward(x, h) for h in step_sizes]
three_point_forward_results = [three_point_forward(x, h) for h in step_sizes]
five_point_central_results = [five_point_central(x, h) for h in step_sizes]

exact_value = fprime(x)

#Print table of results
print(f"{'h':<10} {'3-pt Backward':<20} {'3-pt Forward':<20} {'5-pt Central':<20} {'Exact Value':<20}")
for i, h in enumerate(step_sizes):
    print(f"{h:<10} {three_point_backward_results[i]:<20.10f} {three_point_forward_results[i]:<20.10f} {five_point_central_results[i]:<20.10f} {exact_value:<20.10f}")

three_point_backward_errors_log = [math.log10(abs(three_point_backward_results[i] - exact_value)) for i in range(len(step_sizes))]
three_point_forward_errors_log = [math.log10(abs(three_point_forward_results[i] - exact_value)) for i in range(len(step_sizes))]
five_point_central_errors_log = [math.log10(abs(five_point_central_results[i] - exact_value)) for i in range(len(step_sizes))]

#Print log log graph of 3-point backward error
plt.figure(figsize=FIGSIZE)
plt.plot(np.log10(step_sizes), three_point_backward_errors_log, label='3-pt Backward', marker='o')
plt.xlabel('Log10 (Step size (h))')
plt.ylabel('Log10 (Absolute Error)')
plt.title('Error in 3-point backward error method')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(str(save_dir / 'q3_error_3pt_backward.png'), dpi=SAVE_DPI, bbox_inches='tight')
print("Saved 3-pt backward error graph.")

#Print log log graph of 3-point forward error
plt.figure(figsize=FIGSIZE)
plt.plot(np.log10(step_sizes), three_point_forward_errors_log, label='3-pt Forward', marker='o')
plt.xlabel('Log10 (Step size (h))')
plt.ylabel('Log10 (Absolute Error)')
plt.title('Error in 3-point forward error method')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(str(save_dir / 'q3_error_3pt_forward.png'), dpi=SAVE_DPI, bbox_inches='tight')
print("Saved 3-pt forward error graph.")

#Print log log graph of 5-point central error
plt.figure(figsize=FIGSIZE)
plt.plot(np.log10(step_sizes), five_point_central_errors_log, label='5-pt Central', marker='o')
plt.xlabel('Log10 (Step size (h))')
plt.ylabel('Log10 (Absolute Error)')
plt.title('Error in 5-point central error method')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(str(save_dir / 'q3_error_5pt_central.png'), dpi=SAVE_DPI, bbox_inches='tight')
print("Saved 5-pt central error graph.")

#Combined plot
plt.figure(figsize=FIGSIZE)
plt.plot(np.log10(step_sizes), three_point_backward_errors_log, label='3-pt Backward', marker='o')
plt.plot(np.log10(step_sizes), three_point_forward_errors_log, label='3-pt Forward', marker='o')
plt.plot(np.log10(step_sizes), five_point_central_errors_log, label='5-pt Central', marker='o')
plt.xlabel('Log10 (Step size (h))')
plt.ylabel('Log10 (Absolute Error)')
plt.title('Error in Numerical Differentiation Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(str(save_dir / 'q3_error_methods_combined.png'), dpi=SAVE_DPI, bbox_inches='tight')
print("Saved combined error graph.")