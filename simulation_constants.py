# sensitivity analysis
simulation_steps = 1000
start_simulation_time = 0
end_simulation_time = 1000
sample_num = 100

# data assimilation
# Inference with rejection sampling
# batch_size defines how many simulations are performed in each passing through the graph
batch_size_c = 10
# number of samples
N = 10
quantile = 0.001
# This parameter makes range for input parameters
# (a-width, 2 * width) ---> from a-width to a + width
width = 0.1
first_x_range = 200
second_x_range = 400
third_x_range = 600
# We define start, stop and sampling for the function
start = 0
stop = 1000
# Must be 1, because of new ODE solver
sampling = 1
