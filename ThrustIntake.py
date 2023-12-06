import numpy as np
import matplotlib.pyplot as plt

# File name of the reference thrust curve
file_name_ref = 'Thrusts\\AeroTech_H100W_DMS.eng'

# Custom function to load the reference thrust curve file
def load_custom_format(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # Only process non-empty lines
                # Splitting the line by spaces and converting to float
                time, thrust = map(float, stripped_line.split())
                data.append([time, thrust])
    return np.array(data)

# Load the file using the custom function
ref_thrust_curve = load_custom_format(file_name_ref)

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

from scipy.interpolate import interp1d
from scipy.signal import correlate

# Constants
LB_TO_N = 4.44822  # Conversion factor from lbf to Newtons

# Convert the reference curve from Newtons to lbf, if necessary
ref_thrust_curve[:, 1] /= LB_TO_N

# Step 1: Resample the reference curve to match the time intervals of the averaged curve
# We assume the ref_thrust_curve[:, 0] is the time and ref_thrust_curve[:, 1] is the thrust
ref_interp_func = interp1d(ref_thrust_curve[:, 0], ref_thrust_curve[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
ref_thrust_interpolated = ref_interp_func(time_values)

# Step 2: Find the peak of the averaged curve
peak_index_averaged = np.argmax(averaged_curve)

# Step 3: Find the peak of the reference curve
peak_index_reference = np.argmax(ref_thrust_interpolated)

# Step 4: Calculate the shift required to align the peaks
required_shift = peak_index_averaged - peak_index_reference

# Step 5: Apply the shift
# Shift the reference curve's thrust values to align the peaks
# Note: This assumes that the peak should be shifted to the right; if it needs to be shifted to the left, the sign should be reversed
if required_shift > 0:
    ref_thrust_aligned = np.concatenate((np.full(required_shift, np.nan), ref_thrust_interpolated[:-required_shift]))
else:
    ref_thrust_aligned = np.concatenate((ref_thrust_interpolated[-required_shift:], np.full(-required_shift, np.nan)))

# Step 6: Plot the manually aligned curves
plt.figure(figsize=(10, 6))
plt.plot(time_values, averaged_curve, color='black', linewidth=2, label='Averaged Curve')
plt.plot(time_values, ref_thrust_aligned, color='red', linewidth=2, label='Reference Curve (Manually Aligned)')
plt.title("Averaged and Reference Thrust Curve (Manual Alignment)")
plt.xlabel("Time (seconds)")
plt.ylabel("Thrust (N)")
plt.legend()
plt.grid(True)
plt.show()