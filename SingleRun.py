# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:10:11 2023

@author: power105
"""

import pathlib
from rocketpy import Environment, SolidMotor, Rocket, Flight, plots
import netCDF4

env = Environment(latitude=40.4237, longitude=-86.9212, elevation=190)

import datetime

tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

env.set_atmospheric_model(type="Forecast", file="GFS")

print(env.info())

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


print(Pro75M1670.info())

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

#unused
def fake_trigger(p, h, y):
    # activate main when vz < 0 m/s and z < 800 m
    return True if h > 0 else False


#adds a zero drag chute to move into the parahute phase
main = calisto.add_parachute(
    name="false",
    cd_s=0,
    trigger="apogee",      # ejection altitude in meters
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

# drogue = calisto.add_parachute(
#     name="drogue",
#     cd_s=1.0,
#     trigger="apogee",  # ejection at apogee
#     sampling_rate=105,
#     lag=1.5,
#     noise=(0, 8.3, 0.5),
# )

#time of parahute charge (name from old)
burnout_t = 10

#print(calisto.plots.static_margin())

#first flight, set up for complete analysis results is the smae if max_time is set, and passed on
flight_phase1 = Flight(
    rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0 , max_time=burnout_t, max_time_step = .1
    )

print(flight_phase1.all_info())
print("phase 1 complete")

#creates the state matrix for the second phase
# initial_solution = [
#     burnout_t,
#     flight_phase1.x(burnout_t), flight_phase1.y(burnout_t), flight_phase1.z(burnout_t),
#     flight_phase1.vx(burnout_t), flight_phase1.vy(burnout_t), flight_phase1.vz(burnout_t),
#     flight_phase1.e0(burnout_t), flight_phase1.e1(burnout_t), flight_phase1.e2(burnout_t), flight_phase1.e3(burnout_t),
#     flight_phase1.w1(burnout_t), flight_phase1.w2(burnout_t), flight_phase1.w3(burnout_t)
# ]

#trigger ASAP
def main_trigger(p, h, y):
    # activate main when vz < 0 m/s and z < 800 m
    return True


main = calisto.add_parachute(
    name="main",
    cd_s=1.0,
    trigger=main_trigger,      # ejection altitude in meters
    sampling_rate=105,
    lag=1.5, 
    noise=(0, 8.3, 0.5),
)

#simulate a chute only flight
flight_phase2 = Flight(
    rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0, initial_solution=flight_phase1 #,max_time_step = .1
    )

#print(flight_phase2.info())

#flight_phase2.trajectory_3d.plot()
flight_phase2.plots.trajectory_3d()





