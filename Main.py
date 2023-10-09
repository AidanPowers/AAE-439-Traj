# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:19:36 2023
https://docs.rocketpy.org/en/latest/user/first_simulation.html
@author: power105
"""
import pathlib
from rocketpy import Environment, SolidMotor, Rocket, Flight


env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

import datetime

tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

env.set_atmospheric_model(type="Forecast", file="GFS")

#print(env.info())

fileLoc = str(pathlib.Path().resolve())

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


#print(Pro75M1670.info())

calisto = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag=fileLoc + "/data/calisto/powerOffDragCurve.csv",
    power_on_drag=fileLoc + "/data/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

calisto.add_motor(Pro75M1670, position=-1.255)

rail_buttons = calisto.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.6182,
    angular_position=45,
)

nose_cone = calisto.add_nose(
    length=0.55829, kind="von karman", position=1.278
)

fin_set = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0.5,
    airfoil=(fileLoc + "/data/calisto/NACA0012-radians.csv","radians"),
)

tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)


main = calisto.add_parachute(
    name="main",
    cd_s=10.0,
    trigger=800,      # ejection altitude in meters
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

drogue = calisto.add_parachute(
    name="drogue",
    cd_s=1.0,
    trigger="apogee",  # ejection at apogee
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

#print(calisto.plots.static_margin())

# test_flight = Flight(
#     rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0
#     )

#print(test_flight.all_info())

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

# # Define bounds for the launch parameters
# bounds = [(80, 90),  # bounds for inclination
#           (0, 360)]  # bounds for heading

# # Perform parallelized optimization using differential evolution
# optimized_result = opt.differential_evolution(simulate_flight, bounds, workers=-1, updating='deferred')

# # The optimized inclination and heading are now in optimized_result.x
# optimized_inclination, optimized_heading = optimized_result.x

# # Now simulate the optimized flight to verify
# optimized_flight = Flight(
#     rocket=calisto,
#     environment=env,
#     rail_length=5.2,
#     inclination=optimized_inclination,
#     heading=optimized_heading
# )


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Define the range of values for each parameter
inclination_values = np.linspace(80, 90, 3)  # 20 points between 80 and 90
heading_values = np.linspace(0, 360, 5)  # 36 points between 0 and 360

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
    distance_from_rail_values = np.stack(results_list, axis=0)
    
    # Create a meshgrid for plotting
    inclination_mesh, heading_mesh = np.meshgrid(heading_values, inclination_values)
    
    # Plot the phase space map
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(inclination_mesh, heading_mesh, distance_from_rail_values, cmap='viridis')
    plt.colorbar(cp, label='Distance from Rail (m)')
    plt.xlabel('Heading (degrees)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Phase Space Map')
    plt.show()
