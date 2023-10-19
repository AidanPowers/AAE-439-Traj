import numpy as np
import matplotlib.pyplot as plt

# Number of groups and samples
num_groups = 10
num_samples = 10000

# Initialize an empty list to store thrust data arrays
thrust_curves = []

# Load data from CSVs and truncate to 10,000 samples
for i in range(1, num_groups + 1):
    file_name = f"Thrusts\\Group{i}.csv"
    thrust_data = np.loadtxt(file_name, delimiter=',', skiprows=1)  # Assumes first row is header
    thrust_curves.append(thrust_data[:num_samples])

def compute_steady_state_error(data):
    """Compute the steady-state error based on the first and last second."""
    first_second_avg = np.mean(data[:2000])
    last_second_avg = np.mean(data[-2000:])
    return (first_second_avg + last_second_avg) / 2

def mse_curve(curve1, curve2):
    """Compute the Mean Squared Error between two curves."""
    return np.mean((curve1 - curve2) ** 2)

def align_curves(ref_curve, curve):
    """Align curve to the reference curve using cross-correlation."""
    cross_corr = np.correlate(ref_curve, curve, mode='full')
    # Find the shift that gives the maximum cross-correlation
    shift = np.argmax(cross_corr) - (len(curve) - 1)
    return np.roll(curve, shift)

# Correct for steady-state error
for i in range(len(thrust_curves)):
    steady_state_error = compute_steady_state_error(thrust_curves[i])
    thrust_curves[i] -= steady_state_error

# Compute pairwise MSE for all curves
mse_values = np.zeros((num_groups, num_groups))
for i in range(num_groups):
    for j in range(num_groups):
        mse_values[i, j] = mse_curve(thrust_curves[i], thrust_curves[j])

# Average MSE for each curve
avg_mse = mse_values.mean(axis=1)

# Get indices of 5 curves with the lowest average MSE
top_5_indices = np.argsort(avg_mse)[:5]

# Use the curve with the lowest MSE as the reference
ref_index = top_5_indices[0]
ref_curve = thrust_curves[ref_index]

# Align other curves to the reference curve
aligned_curves = [ref_curve]
for i in top_5_indices[1:]:
    aligned_curves.append(align_curves(ref_curve, thrust_curves[i]))

# Average the aligned curves
averaged_curve = np.mean(aligned_curves, axis=0)

# Plotting All Curves and Averaged Curve
plt.figure(figsize=(10, 6))
time_values = np.arange(0, num_samples/2000, 1/2000)  # x-axis values for time in seconds

for i, thrust_data in enumerate(thrust_curves):
    plt.plot(time_values, thrust_data, alpha=0.4, label=f'Group {i+1}')

plt.plot(time_values, averaged_curve, color='black', linewidth=2, label='Averaged Curve (Aligned)')
plt.title("All Thrust Curves with Aligned and Averaged Curve")
plt.xlabel("Time (seconds)")
plt.ylabel("Thrust (lbf)")
plt.legend()
plt.grid(True)
plt.show()

# Plotting Aligned Curves
plt.figure(figsize=(10, 6))

for i, idx in enumerate(top_5_indices):
    plt.plot(time_values, aligned_curves[i], label=f'Aligned Group {idx+1}')

plt.title("Aligned Thrust Curves for Top 5 Most Similar Groups")
plt.xlabel("Time (seconds)")
plt.ylabel("Thrust (lbf)")
plt.legend()
plt.grid(True)
plt.show()

# Constants
LB_TO_N = 4.44822  # Conversion factor from lbf to Newtons

def get_longest_continuous_section(data, threshold=0.1):
    """
    Identify the longest continuous section in data where values exceed threshold.
    Return the start and end indices of this section.
    """
    start_idx = None
    end_idx = None
    max_duration = 0
    current_duration = 0
    current_start_idx = None
    
    for i, value in enumerate(data):
        if value > threshold:
            if current_start_idx is None:
                current_start_idx = i
            current_duration += 1
        else:
            if current_start_idx is not None:
                if current_duration > max_duration:
                    start_idx = current_start_idx
                    end_idx = i  # i-1 is the last point above threshold, but we want to include it
                    max_duration = current_duration
                current_duration = 0
                current_start_idx = None
    
    # Check in case the longest section reaches the end of the data
    if current_start_idx is not None and current_duration > max_duration:
        start_idx = current_start_idx
        end_idx = len(data)
    
    return start_idx, end_idx

start, end = get_longest_continuous_section(averaged_curve)
trimmed_curve = averaged_curve[start:end]
trimmed_time_values = time_values[start:end]

# Convert thrust from lbf to Newtons
trimmed_curve_newtons = trimmed_curve * LB_TO_N

# Adjust the trimmed time values so that they start from zero
trimmed_time_values = trimmed_time_values - trimmed_time_values[0]

# Export to CSV
export_data = np.column_stack((trimmed_time_values, trimmed_curve_newtons))
np.savetxt('DMS_H100W_14A.csv', export_data, delimiter=',')

print(f"Trimmed and exported curve from {trimmed_time_values[0]:.2f}s to {trimmed_time_values[-1]:.2f}s.")