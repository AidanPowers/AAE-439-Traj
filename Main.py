# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:19:36 2023
https://docs.rocketpy.org/en/latest/user/first_simulation.html
@author: AidanPowers
"""

# Import necessary libraries and modules
import pathlib
from rocketpy import Environment, SolidMotor, Rocket, Flight
import datetime
import copy

# Define environmental conditions for the launch
env = Environment(latitude=40.4237, longitude=-86.9212, elevation=190)

# Get tomorrow's date
tomorrow = datetime.date.today() + datetime.timedelta(days=1)

# Set the environment's date to tomorrow at 12:00 UTC
env.set_date(
    (2023, 11, 5, 12)#tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

# Set the atmospheric model to be used, based on a forecast file
env.set_atmospheric_model(type="Forecast", file="GFS")

# Uncomment to print environment information
#print(env.info())

# Resolve the current directory path and convert it to a string
fileLoc = str(pathlib.Path().resolve())

# Create a solid motor object with specified properties and data file
DMS_H100W_14A = SolidMotor(
    thrust_source=fileLoc + "/DMS_H100W_14A.csv",
    #thrust_source=120,
    dry_mass=.154,
    dry_inertia=(0.0125, 0.0125, 0.0002),
    nozzle_radius=10.5 / 2 / 1000,
    grain_number=1,
    grain_density=1820.26,
    grain_outer_radius=33/ 2 / 1000,
    grain_initial_inner_radius=22 / 2 / 1000,
    grain_initial_height=140 / 1000,
    grain_separation=0 / 1000,
    grains_center_of_mass_position=0.076,
    center_of_dry_mass_position=0.076,
    nozzle_position=0,
    burn_time=2.4,
    throat_radius=5 / 2 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)
deploy_charge_time = 15


# Uncomment to print motor information
#print(Pro75M1670.info())

# Create a rocket object with specified properties and drag curves
loc_iv = Rocket(
    radius=10.2/100/2,
    mass=1.022,
    inertia=(.11675, .11675, .0028950),
    power_off_drag=fileLoc + "//PowerOff.csv",
    power_on_drag=fileLoc + "//PowerOn.csv",
    center_of_mass_without_motor=0.76,
    coordinate_system_orientation="nose_to_tail",
)

# Add the solid motor to the rocket at a specified position
loc_iv.add_motor(DMS_H100W_14A, position=1.19)

# Set the positions of the rail buttons on the rocket
rail_buttons = loc_iv.set_rail_buttons(
    upper_button_position=0.6418,
    lower_button_position=1.0182,
    angular_position=45,
)

# Add a nose cone to the rocket
nose_cone = loc_iv.add_nose(
    length=0.325, kind="ogive", position=0
)

# Add a set of trapezoidal fins to the rocket
fin_set = loc_iv.add_trapezoidal_fins(
    n=3,
    root_chord=0.171,
    tip_chord=0.063,
    span=0.1080,
    position=1.02,
    cant_angle=0,
    sweep_length=0.143 
    #airfoil=(fileLoc + "/data/calisto/NACA0012-radians.csv","radians"),
)


# print(loc_iv.plots.static_margin())
# print(loc_iv.all_info())


#define main parachute on duplicate rocket

def main_trigger(p, h, y):
    # activate main when vz < 0 m/s and z < 800 m
    return True
loc_iv_chute = copy.deepcopy(loc_iv)
main = loc_iv_chute.add_parachute(
    name="main",
    cd_s=0.80,
    trigger=main_trigger,      # ejection altitude in meters
    sampling_rate=105,
    lag=0,
    noise=(0, 8.3, 0.5)

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
    
    #flight pre parachute deploy
    phase1_flight = Flight(
        rocket=loc_iv,
        environment=env,
        rail_length=5.2,
        inclination=inclination,
        heading=heading,
        #max_time=deploy_charge_time,
        max_time_step = .1
        #verbose = True
    )
    
    initial_solution = [
        deploy_charge_time,
        phase1_flight.x(deploy_charge_time), phase1_flight.y(deploy_charge_time), phase1_flight.z(deploy_charge_time),
        phase1_flight.vx(deploy_charge_time), phase1_flight.vy(deploy_charge_time), phase1_flight.vz(deploy_charge_time),
        phase1_flight.e0(deploy_charge_time), phase1_flight.e1(deploy_charge_time), phase1_flight.e2(deploy_charge_time), phase1_flight.e3(deploy_charge_time),
        phase1_flight.w1(deploy_charge_time), phase1_flight.w2(deploy_charge_time), phase1_flight.w3(deploy_charge_time)
    ]
    
    
    #flight post parachute deploy
    test_flight = Flight(
        rocket=loc_iv_chute,
        environment=env,
        rail_length=5.2,
        inclination=inclination,
        heading=heading,
        initial_solution=initial_solution
        
    )
    
    
    launch_position = np.array([0, 0])
    landing_position = np.array([test_flight.x_impact, test_flight.y_impact])
    
    distance_from_rail = np.linalg.norm(launch_position - landing_position)
    
    print(f'Inclination: {inclination:.2f}, Heading: {heading:.2f}, Distance from Rail: {distance_from_rail:.2f}')
    
    return distance_from_rail


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import scipy.optimize as opt
import scipy.interpolate as interp

# Get the number of cores (not threads)
num_cores = cpu_count() // 2  # Assumes hyper-threading is enabled

# Define the range of values for each parameter
inclination_values = np.linspace(60, 90, 15)  # 25 points between 40 and 90
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

def plot_distance_from_rail(constant_heading, interpolated_function):
    # Ensure headings wrap around at 0 and 360 degrees
    headings = [(constant_heading + 360) % 360, 
                (constant_heading + 5 + 360) % 360, 
                (constant_heading - 5 + 360) % 360]

    plt.figure(figsize=(10, 8))

    for heading in headings:
        # Create arrays to hold the inclination values and corresponding distances
        inclinations = np.linspace(60, 90, 25)
        distances = np.empty_like(inclinations)
        
        # Evaluate the interpolated function for each inclination at the current heading
        for i, inclination in enumerate(inclinations):
            params = np.array([heading, inclination])
            distances[i] = interpolated_function(params)
        
        # Plot the distances for this heading
        plt.plot(inclinations, distances, label=f'Heading {heading}°')
    
    plt.xlabel('Inclination (degrees)')
    plt.ylabel('Distance from Rail (m)')
    plt.title(f'Distance from Rail at Heading {constant_heading}° and ±5°')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Use a Pool of workers to parallelize the outer loop
    with Pool(processes=num_cores) as pool:
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
    
    plot_distance_from_rail(90, interpolated_function)
    