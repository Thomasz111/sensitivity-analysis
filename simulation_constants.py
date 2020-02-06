# sensitivity analysis
simulation_steps = 100
start_simulation_time = 0
end_simulation_time = 15
sample_num = 100

# data assimilation
# Inference with rejection sampling
# batch_size defines how many simulations are performed in each passing through the graph
batch_size = 100
# number of samples
N = 100
# This parameter makes range for input parameters
# (a-width, 2 * width) ---> from a-width to a + width
width = 0.1
