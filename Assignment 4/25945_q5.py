#step size
import math
import matplotlib.pyplot as plt

#natural angular frequency
omega_0 = 2*math.pi

#damping coefficient
gamma = 0.5

#driving acceleration amplitude
A_0 = 2

#driving angular frequency
omega = 2

#initial position
x_0 = 1

#initial velocity
v_0 = 0

#start time
start_time = 0

#end time
end_time = 30

#directory to save plots
output_dir = "H:\\My Drive\\Numerical Methods\\Assignments\\Assignment 4\\"

def get_next_x (x_i, v_i, delta_t):
    return x_i + delta_t*v_i

def get_next_v (v_i, t_i, x_i, delta_t):
    return v_i + delta_t*(A_0*math.cos(omega*t_i) - 2*gamma*v_i - (omega_0**2)*x_i)

def f(t, y):
    x = y[0]
    v = y[1]

    f_1 = v
    f_2 = A_0*math.cos(omega*t) - 2*gamma*v - (omega_0**2)*x

    return [f_1, f_2]

def euler(delta_t):
    curr_time = start_time

    x_i = [x_0]
    v_i = [v_0]

    while curr_time < end_time:
        x_i.append(get_next_x(x_i[-1], v_i[-1], delta_t))
        v_i.append(get_next_v(v_i[-1], curr_time, x_i[-1], delta_t))
        curr_time += delta_t

    return x_i, v_i

def runge_kutta_4 (delta_t):
    curr_time = start_time

    x_i = [x_0]
    v_i = [v_0]

    while curr_time < end_time:
        #Get current state
        y_i = [x_i[-1], v_i[-1]]
        
        #Calculate the k values
        k1 = f(curr_time, y_i)
        k2 = f(curr_time + delta_t/2, [y_i[0] + delta_t/2 * k1[0], y_i[1] + delta_t/2 * k1[1]])
        k3 = f(curr_time + delta_t/2, [y_i[0] + delta_t/2 * k2[0], y_i[1] + delta_t/2 * k2[1]])
        k4 = f(curr_time + delta_t, [y_i[0] + delta_t * k3[0], y_i[1] + delta_t * k3[1]])

        #Calculate next state
        x_next = y_i[0] + (delta_t/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v_next = y_i[1] + (delta_t/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

        #Add next state to lists
        x_i.append(x_next)
        v_i.append(v_next)

        #Increment time
        curr_time += delta_t
    
    return x_i, v_i

delta_t = 0.01
euler_x, euler_v = euler(delta_t)
rk4_x, rk4_v = runge_kutta_4(delta_t)

print("Euler Method Results every 500 intervals (every 5 seconds):")
for i in range(0, len(euler_x), 500):
    print(f"t = {i*delta_t:.2f} s: x = {euler_x[i]:.4f} m, v = {euler_v[i]:.4f} m/s")

print("Runge-Kutta 4 Method Results every 500 intervals (every 5 seconds):")
for i in range(0, len(rk4_x), 500):
    print(f"t = {i*delta_t:.2f} s: x = {rk4_x[i]:.4f} m, v = {rk4_v[i]:.4f} m/s")

#part 2: visualisation
t = [i*delta_t for i in range(len(euler_x))]
plt.figure(figsize=(16,9))
plt.plot(t, euler_x, label='Euler Method', linestyle='-', linewidth=0.7)
plt.plot(t, rk4_x, label='Runge-Kutta 4 Method', linestyle='dotted', linewidth=3)
plt.title('Damped, Driven Harmonic Oscillator: Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()
plt.savefig(output_dir + "q5_2.png", dpi=600)

#part 4: convergence analysis
#values of delta_t to test
delta_ts = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]

#get true solution using very small delta_t and RK-4
true_solution_x, true_solution_v = runge_kutta_4(0.00125)

euler_rmse = []
rk4_rmse = []

for dt in delta_ts:
    euler_x, euler_v = euler(dt)
    rk4_x, rk4_v = runge_kutta_4(dt)

    #Calculate root mean square error at all time steps which are both in current and true solution
    steps = int(end_time/dt)
    euler_error = 0
    rk4_error = 0

    #Calculate squared errors
    for i in range(steps):
        true_index = int(i * (dt / 0.00125))
        euler_error += (euler_x[i] - true_solution_x[true_index])**2
        rk4_error += (rk4_x[i] - true_solution_x[true_index])**2

    #Calculate RMSE
    euler_error = math.sqrt(euler_error / steps)
    rk4_error = math.sqrt(rk4_error / steps)

    euler_rmse.append(euler_error)
    rk4_rmse.append(rk4_error)

    print(f"Delta t: {dt:.5f} s")
    print(f"Euler Method RMSE: Position RMSE = {euler_error:.6e}, Velocity RMSE = {euler_error:.6e}")
    print(f"RK4 Method RMSE: Position RMSE = {rk4_error:.6e}, Velocity RMSE = {rk4_error:.6e}")
    print("===============================")

#Close previous plot
plt.close('all')

# --- Add best-fit lines in log-log space (manual least squares, no np.polyfit) ---
def log_log_linear_fit(h_values, err_values):
    """
    Fit err = C * h^p by linear least squares on ln(err) = a + p ln(h).
    Returns (C, p).
    """
    xs = [math.log(h) for h in h_values]
    ys = [math.log(e) for e in err_values]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    # slope p
    denom = n * sum_xx - sum_x * sum_x
    p = (n * sum_xy - sum_x * sum_y) / denom
    # intercept a
    a = (sum_y - p * sum_x) / n
    C = math.exp(a)
    return C, p

euler_C, euler_p = log_log_linear_fit(delta_ts, euler_rmse)
rk4_C, rk4_p   = log_log_linear_fit(delta_ts, rk4_rmse)

euler_fit = [euler_C * h**euler_p for h in delta_ts]
rk4_fit   = [rk4_C * h**rk4_p for h in delta_ts]

plt.figure(figsize=(16,9))
plt.loglog(delta_ts, euler_rmse, 'o', label='Euler RMSE')
plt.loglog(delta_ts, euler_fit, '-', label=f'Euler best fit: {euler_p:.2f} order')
plt.loglog(delta_ts, rk4_rmse, 'o', label='RK4 RMSE')
plt.loglog(delta_ts, rk4_fit, '-', label=f'RK4 best fit: {rk4_p:.2f} order')

plt.title('Convergence Analysis: RMSE vs Step Size')
plt.xlabel('Step Size h (s)')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.savefig(output_dir + "q5_4.png", dpi=600)