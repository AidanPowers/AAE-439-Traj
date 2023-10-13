# AAE-439-Traj

In progress code for AAE-439 grad group. \
Heavily based on https://docs.rocketpy.org/en/latest/user/first_simulation.html with some advice from GPT-4 \
Creates a Phase Space map inspired by https://www.youtube.com/watch?v=b-pLRX3L-fg \
Finds the minimum point in the phase space approximation \
Phase space map generation is parallelized. Run code in annaconda prompt to see printed output. 



TODO:

1) Update drag with loc 4 data https://www.apogeerockets.com/Rocket_Kits/Skill_Level_3_Kits/LOC_IV
2) ~~Fix parachute deploy to timed (May need to run twice to determine deploy altitude) https://docs.rocketpy.org/en/latest/user/rocket.html#adding-parachutes~~
3) Update motor data https://docs.rocketpy.org/en/latest/user/motors/solidmotor.html
4) Make code more readable
5) Implement DMS H100W-14A White Lightning https://aerotech-rocketry.com/products/product_f89c2d4d-4c3c-9a99-bfd8-b2d0b1dcef8a 
    -Can use this plot digitizer on published thrust curve https://plotdigitizer.com/app
6) Run and render the selected output angle
    
    
![Phase Space Demo Map](https://github.com/AidanPowers/AAE-439-Traj/blob/main/ExamplePhaseSpace.png?raw=true)