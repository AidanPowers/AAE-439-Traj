# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:10:11 2023

@author: power105
"""

import pathlib
from rocketpy import Environment, SolidMotor, Rocket, Flight, plots
import netCDF4
import numpy as np
import datetime




HIRESW_dictionary = {
    "time": "time",
    "latitude": "lat",
    "longitude": "lon",
    "level": "lev",
    "temperature": "tmpprs",
    "surface_geopotential_height": "hgtsfc",
    "geopotential_height": "hgtprs",
    "u_wind": "ugrdprs",
    "v_wind": "vgrdprs",
}

date_info = (2023, 11, 5, 18)
date_string = f"{date_info[0]}{date_info[1]:02}{date_info[2]:02}"

env = Environment(latitude=40.4237, longitude=-86.9212, elevation=190)

tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env.set_date(
    (2023, 11, 5, 18)
)  # Hour given in UTC time

# env.set_atmospheric_model(type="Forecast", file="RAP")

# env.set_atmospheric_model(
#     type="Forecast",
#     file=f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/hiresw_conusarw_12z",
#     dictionary=HIRESW_dictionary,
# )

# env.set_atmospheric_model(
#     type="custom_atmosphere",
#     pressure=None,
#     temperature=300,
#     wind_u=[(0, 10/3.281)],
#     wind_v=[(0, 0)],
# )

## Grib re-analysis
# conda install -c conda-forge cfgrib
import cfgrib
import numpy as np
from rocketpy import Environment

# Open GRIB file
ds = cfgrib.open_datasets('Forecasts\\rap_130_20231105_1800_000.grb2')

# Extract data - modify keys as per your GRIB file structure
temperature = ds[0].tmpprs.values
pressure = ds[0].prmsl.values
u_wind = ds[0].ugrdprs.values
v_wind = ds[0].vgrdprs.values

# Calculate wind speed and direction
wind_speed = np.sqrt(u_wind**2 + v_wind**2)
wind_direction = np.arctan2(u_wind, v_wind) * 180 / np.pi

# Format data for RocketPy - assuming altitude is an array of altitudes
# You might need to interpolate your data to match these altitudes
formatted_temperature = np.interp(altitude, pressure, temperature)
formatted_wind_speed = np.interp(altitude, pressure, wind_speed)
formatted_wind_direction = np.interp(altitude, pressure, wind_direction)

# Set up RocketPy environment
env = Environment()
env.set_atmospheric_model(type='Tabular', 
                          altitude=altitude, 
                          temperature=formatted_temperature, 
                          pressure=pressure, 
                          wind_speed=formatted_wind_speed, 
                          wind_direction=formatted_wind_direction)


print(env.info())

fileLoc = str(pathlib.Path().resolve())

# Create a solid motor object with specified properties and data file
DMS_H100W_14A = SolidMotor(
    thrust_source=fileLoc + "/DMS_H100W_14A.csv",
    #thrust_source=120,
    dry_mass=.154,
    #dry_inertia=(0.0125, 0.0125, 0.0002),
    dry_inertia=(0.0, 0.0, 0.0002),
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
deploy_charge_time = 14


# Uncomment to print motor information
print(DMS_H100W_14A.all_info())

loc_iv = Rocket(
    radius=10.2/100/2,
    mass=1.022,
    #inertia=(.11675, .11675, .0028950),
    inertia=(0, 0, 0),
    power_off_drag=fileLoc + "//PowerOff.csv",
    power_on_drag=fileLoc + "//PowerOn.csv",
    center_of_mass_without_motor=0.76,
    coordinate_system_orientation="nose_to_tail",
)

loc_iv.add_motor(DMS_H100W_14A, position=1.19)

rail_buttons = loc_iv.set_rail_buttons(
    upper_button_position=0.6418,
    lower_button_position=1.0182,
    angular_position=45,
)

nose_cone = loc_iv.add_nose(
    length=0.325, kind="ogive", position=0
)

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

# tail = loc_iv.add_tail(
#     top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
# )

# #unused
# def fake_trigger(p, h, y):
#     # activate main when vz < 0 m/s and z < 800 m
#     return True if h > 0 else False


# #adds a zero drag chute to move into the parahute phase
# main = loc_iv.add_parachute(
#     name="false",
#     cd_s=0,
#     trigger="apogee",      # ejection altitude in meters
#     sampling_rate=105,
#     lag=1.5,
#     noise=(0, 8.3, 0.5),
# )

# drogue = loc_iv.add_parachute(
#     name="drogue",
#     cd_s=1.0,
#     trigger="apogee",  # ejection at apogee
#     sampling_rate=105,
#     lag=1.5,
#     noise=(0, 8.3, 0.5),
# )

#time of parahute charge (name from old)
burnout_t = 14
inc =80.32
head = 266.8


print(loc_iv.plots.static_margin())
print(loc_iv.all_info())
print(loc_iv.draw())

#first flight, result is the same if I run the entire flight, and then pick out a time using the matrix.
flight_phase1 = Flight(
    rocket=loc_iv, environment=env, rail_length=1.828, inclination=inc, heading=head, max_time_step = .1# , max_time=burnout_t, verbose = True
    )

print(flight_phase1.all_info())
print("phase 1 complete")

#creates the state matrix for the second phase
initial_solution = [
    burnout_t,
    flight_phase1.x(burnout_t), flight_phase1.y(burnout_t), flight_phase1.z(burnout_t),
    flight_phase1.vx(burnout_t), flight_phase1.vy(burnout_t), flight_phase1.vz(burnout_t),
    flight_phase1.e0(burnout_t), flight_phase1.e1(burnout_t), flight_phase1.e2(burnout_t), flight_phase1.e3(burnout_t),
    flight_phase1.w1(burnout_t), flight_phase1.w2(burnout_t), flight_phase1.w3(burnout_t)
]

#trigger ASAP
def main_trigger(p, h, y):
    return True


main = loc_iv.add_parachute(
    name="main",
    cd_s=0.80,
    trigger=main_trigger,
    sampling_rate=105,
    lag=0, 
    noise=(0, 8.3, 0.5),
)

#simulate a chute only flight
flight_phase2 = Flight(
    rocket=loc_iv, environment=env, rail_length=1.828, inclination=inc, heading=head, initial_solution=initial_solution #,max_time_step = .1
    )

#print(flight_phase2.info())

#flight_phase2.trajectory_3d.plot()
flight_phase2.plots.trajectory_3d()

launch_position = np.array([0, 0])
landing_position = np.array([flight_phase2.x_impact, flight_phase2.y_impact])
distance_from_rail = np.linalg.norm(launch_position - landing_position)
print(f'Inclination: {inc:.2f}, Heading: {head:.2f}, Distance from Rail: {distance_from_rail:.2f}')

#flight.prints.impact_conditions()






