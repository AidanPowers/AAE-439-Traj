# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:19:36 2023
https://docs.rocketpy.org/en/latest/user/first_simulation.html
@author: power105
"""

# Import necessary libraries and modules
import pathlib
from rocketpy import Environment, SolidMotor, Rocket, Flight
import datetime

# Define environmental conditions for the launch
env = Environment(latitude=40.4237, longitude=-86.9212, elevation=190)

# Get tomorrow's date
tomorrow = datetime.date.today() + datetime.timedelta(days=1)

# Set the environment's date to tomorrow at 12:00 UTC
env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

# Set the atmospheric model to be used, based on a forecast file
env.set_atmospheric_model(type="Forecast", file="GFS")

# Uncomment to print environment information
#print(env.info())

# Resolve the current directory path and convert it to a string
fileLoc = str(pathlib.Path().resolve())

# Create a solid motor object with specified properties and data file
Pro75M1670 = SolidMotor(
    thrust_source=fileLoc + "/data/motors/Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

# Uncomment to print motor information
#print(Pro75M1670.info())

# Create a rocket object with specified properties and drag curves
calisto = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag=fileLoc + "/data/calisto/powerOffDragCurve.csv",
    power_on_drag=fileLoc + "/data/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

# Add the solid motor to the rocket at a specified position
calisto.add_motor(Pro75M1670, position=-1.255)

# Set the positions of the rail buttons on the rocket
rail_buttons = calisto.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.6182,
    angular_position=45,
)

# Add a nose cone to the rocket
nose_cone = calisto.add_nose(
    length=0.55829, kind="von karman", position=1.278
)

# Add a set of trapezoidal fins to the rocket
fin_set = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0.5,
    airfoil=(fileLoc + "/data/calisto/NACA0012-radians.csv","radians"),
)

# Add a tail section to the rocket
tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)


# Add a main parachute to the rocket, with a specified deployment condition
main = calisto.add_parachute(
    name="main",
    cd_s=10.0,
    trigger=800,      # ejection altitude in meters
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

# Add a drogue parachute to the rocket, with a specified deployment condition
drogue = calisto.add_parachute(
    name="drogue",
    cd_s=1.0,
    trigger="apogee",  # ejection at apogee
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)


# Uncomment to plot and print the rocket's static margin
# print(calisto.plots.static_margin())

# Uncomment to create a Flight object to simulate a rocket flight
# test_flight = Flight(
#     rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0
#     )

# Uncomment to print all information about the simulated flight
# print(test_flight.all_info())

# Notify that setup is complete
print("setup complete")


import scipy.optimize as opt
import numpy as np

def simulate_flight(params):
    inclination, heading = params
    test_flight = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.2,
        inclination=inclination,
        heading=heading
    )
    
    launch_position = np.array([0, 0])
    landing_position = np.array([test_flight.x_impact, test_flight.y_impact])
    
    distance_from_rail = np.linalg.norm(launch_position - landing_position)
    
    print(f'Inclination: {inclination:.2f}, Heading: {heading:.2f}, Distance from Rail: {distance_from_rail:.2f}')
    
    return distance_from_rail


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy.optimize as opt
import scipy.interpolate as interp

# Define the range of values for each parameter
inclination_values = np.linspace(40, 90, 25)  # 25 points between 40 and 90
heading_values = np.linspace(0, 360, 36)  # 36 points between 0 and 360

# Create an empty array to hold the objective function values
distance_from_rail_values = np.empty((len(inclination_values), len(heading_values)))

def evaluate_inclination(inclination):
    # Evaluate the objective function over all heading values for a given inclination
    results = np.empty(len(heading_values))
    for j, heading in enumerate(heading_values):
        params = [inclination, heading]
        results[j] = simulate_flight(params)
    return results

if __name__ == '__main__':
    # Use a Pool of workers to parallelize the outer loop
    with Pool() as pool:
        # The results will be a list of arrays, one array for each inclination value
        results_list = pool.map(evaluate_inclination, inclination_values)
    
    # Convert the list of arrays into a single 2D array
    distance_from_rail_values = np.stack(results_list, axis=1)
    
    # Create a meshgrid for plotting
    heading_mesh, inclination_mesh = np.meshgrid(heading_values, inclination_values)


    # Create an interpolated function from the phase space data using RegularGridInterpolator
    interpolated_function = interp.RegularGridInterpolator(
        (heading_values, inclination_values), 
        distance_from_rail_values, 
        method='cubic'
    )

    def objective(params):
        # Objective function to be minimized
        return interpolated_function(params)

    # Initial guess for optimization (middle of the parameter ranges)
    initial_guess = [(heading_values[0] + heading_values[-1]) / 2, (inclination_values[0] + inclination_values[-1]) / 2]
    bounds=[(heading_values[0], heading_values[-1]), (inclination_values[0], inclination_values[-1])]

    # Optimize to find the minimum distance from rail
    #result = opt.minimize(objective, initial_guess, bounds=bounds)
    result = opt.differential_evolution(objective, bounds)
    
    # Extract the optimized parameters
    optimized_heading, optimized_inclination = result.x
    minimum_distance = result.fun



    print(f'Minimum distance from rail: {minimum_distance:.2f} meters at Heading: {optimized_heading:.2f} degrees, Inclination: {optimized_inclination:.2f} degrees')
    
    
    # Plot the phase space map
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(heading_mesh, inclination_mesh , np.transpose(distance_from_rail_values), cmap='viridis')
    plt.colorbar(cp, label='Distance from Rail (m)')
    plt.xlabel('Heading (degrees)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Phase Space Map')
    plt.show()
    