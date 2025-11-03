import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

opti = ca.Opti()
np.random.seed(42)

# Direct Multiple Shooting --> whole traj is a variable that Casadi is manipulating    

t_end = 10
N = 100
time_step = t_end / N
times = np.linspace(0, t_end, N + 1)
print(f"Number of time points: {N + 1}")

# True parameter values that the algo will have to find
lambda_true = 1.53
x_0_true = 0.45

theoretical_sol = 1 - (1 - x_0_true) * np.exp(-lambda_true * times)

data = theoretical_sol + np.random.normal(0, 0.1, len(times))

# Parameters to find
lambda_opt = opti.variable()
x_0_opt = opti.variable()

# State variable : whole trajectory
x = opti.variable(N + 1)

# Initial condition constraint
opti.subject_to(x[0] - x_0_opt == 0)

# Trajectory constraints
for k in range(N):
    euler_implicit = x[k] + time_step * lambda_opt * (1 - x[k + 1])
    opti.subject_to(x[k + 1] - euler_implicit == 0)

print(f"x_10 = {x[10]}")

cost = ca.sumsqr(x - data)

opti.minimize(cost)
opti.solver("ipopt")

# Initial guesses
lambda_initial = 1.53
x_0_initial = 0.45
opti.set_initial(lambda_opt, lambda_initial)
opti.set_initial(x_0_opt, x_0_initial)
opti.set_initial(x, data)

sol = opti.solve()

lamba_estimate = sol.value(lambda_opt)
x_0_estimate = sol.value(x_0_opt)
x_estimate = sol.value(x)

print(f"True parameters value : λ = {lambda_true}, x_0 = {x_0_true}")
print(f"Initial guesses : λ = {lambda_initial}, x_0 = {x_0_initial}")
print(f"Estimated values : λ = {lamba_estimate:.4f}, x_0 = {x_0_estimate:.4f}")

estimated_sol = 1 - (1 - x_0_estimate) * np.exp(-lamba_estimate * times)

plt.plot(times, theoretical_sol, color="red", label="true solution")
plt.scatter(times, data, color='blue', label="data points")
plt.plot(times, estimated_sol, color='green', label='estimated solution')
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.savefig("codes/test_casadi.png")