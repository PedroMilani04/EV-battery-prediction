
###########################################################################
#       DEVRT: Dataset of Electric Vehicle Real Trips                     #
#              Iñaki Cejudo    Eider Irigoyen    Iker Arandia             #
#                 Harbil Arregui    Estí­baliz Loyo                        #
#                                                                         #
#                                                                         #
#                           Vicomtech Foundation                          #
#                Basque Research and Technology Alliance (BRTA)           #
#                             www.vicomtech.org                           #
###########################################################################

Title
=====

Dataset of Electric Vehicle Real Trips

Description
===========

This dataset contains CSV files of two electric vehicle real trips. A Nissan Leaf e+ 62kWh and a Dacia Spring 33kW.
There are 29 trips for each car, completed during 4 consecutive days. Both cars where doing the exact same routes. These routes are urban and medium distance, and have different elevation profiles. Routes take place in the Basque Country, Spain.

The folders for vehicle data are "DACIA SPRING" and "NISSAN LEAF". Each folder of the dataset has the name of a vehicle. Inside each folder, the name of each CSV file indicates:

DATE_VEHICLE_ORIGIN_DESTINATION_UNIQUECODE.csv

- DATE: date of the trip in YYYYMMDD format. Dates are between 18/04/2023 and 21/04/2023.
- VEHICLE: make of the car.
- ORIGIN: origin municipality.
- DESTINATION: destination municipality.
- UNIQUECODE: unique numeric code for each CSV file.

For example: "20230420_NISSAN_EIBAR_DONOSTIA_054.csv"

CSV files have the following data:

- timestamp_data_utc: Date and time (in UTC) of the vehicle data.
- elv_spy (m): Elevation provided by the Leaf Spy Pro application.
- speed (km/h): Vehicle speed.
- soc (%): Vehicle battery level.
- amb_temp (Cº): Ambient temperature.
- soh (%): State of Health of the vehicle's battery.
- regenwh (W): Regeneration power.
- Motor Pwr (W): Engine power.
- Aux Pwr (100w): Auxiliary systems power.
- Motor Temp (Cº): Vehicle engine temperature.
- Torque (Nm): Torque
- rpm: Revolutions per minute of the vehicle's engine.
- route_id: Unique numerical identifier of the route.
- longitude: Longitude of vehicle position.
- latitude: Latitude of vehicle position.
- altitude: Altitude  of vehicle position.
- cumul_dist: Cumulative distance in km.
- timestamp_gps_utc: Date and time (UTC) of the positioning data.
- car_id: Vehicle identifier.
- time_diff: Difference in milliseconds between the date and time of the vehicle data and positioning data.
- point_geom: Position of the vehicle in geometry format.
- route_code: Unique route identifier.
- route_description: Route description.
- driver: Route driver.
- start_timestamp: Date and time of the beginning of the route.
- end_timestamp: Date and time of the end of the route.
- car_code: Code of the vehicle.
- car_description: Description of the vehicle.
- capacity (Wh): Vehicle battery capacity.
- ref_consumption (Wh/km): Reference consumption of the vehicle.
- timestamp_weather_utc: Date and time of weather data.
- wind_mph: Wind in mph.
- wind_kph: Wind in kph.
- wind_degree: Wind direction in degrees.
- wind_dir: Wind direction.
- Frontal_Wind (km/h): Frontal effect of wind on the vehicle.
- Veh_deg: Vehicle direction in degrees.
- totalVehicles: Total vehicles on the road section.
- speedAvg (km/h): Average speed on the road section.
- cars_by_speed_interval_0_80: Number of vehicles with a speed between 0-80km/h on the road section.
- cars_by_speed_interval_80_100: Number of vehicles with a speed between 80-100km/h on the road section.
- cars_by_speed_interval_100_120: Number of vehicles with a speed between 100-120km/h on the road section.
- cars_by_speed_interval_0_50: Number of vehicles with a speed between 0-50km/h on the road section.
- cars_by_speed_interval_50_80:Number of vehicles with a speed between 50-80km/h on the road section.
- cars_by_speed_interval_80_120: Number of vehicles with a speed between 80-120km/h on the road section.
- cars_by_speed_interval_120_inf: Number of vehicles with a speed higher than 120km/h on the road section.
- cars_by_length_interval_0_7: Number of vehicles with a length of less than 7 meters on the road section.
- cars_by_length_interval_7_inf: Number of vehicles with a length greater than 7 meters on the road section.
- max_speed (km/h): Maximum speed allowed on the road.
- radius: Distance from the position of the car and the traffic meter in km.

Regarding frontal wind, this can be positive or negative. Positive means the wind pushes in the same direction of the car and negative is the oposite.

The python script usage_example_DEVRT.py includes some utilities and examples to load the data files, describe the main statistics and plot the tracks. 

Note
====
*2026 April 1st: update with additional usage example python script (usage_example_DEVRT.py)
*2026 February 17th: update with additional information containing route plans (additional folder) and cumul_dist column in all CSV files.*

License
=======
The datasets is developed by Vicomtech and is made avaliable under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

This license requires that reusers give credit to the creator. In the event that datasets is used, in whole or in part, in any project, publication, or product, appropriate credit must be given to the rights holder as follows: "This dataset was developed by and is the property of Fundación Centro de Tecnologías de Interacción Visual y Comunicaciones Vicomtech, available at https://github.com/Vicomtech/d-EVD_dual-Electric-Vehicle-Dataset". Failure to provide proper attribution constitutes a violation of the terms of this license.

The license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, for noncommercial purposes only. If others modify or adapt the material, they must license the modified material under identical terms.

Contact
=======

* Eider Irigoyen: eirigoyen@vicomtech.org
* Iñaki Cejudo: icejudo@vicomtech.org
* Harbil Arregui: harregui@vicomtech.org
* Estíbaliz Loyo: eloyo@vicomtech.org
