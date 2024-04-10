# solar-battery-optimizer
Python LP Optimizer for solar power plant and battery dimensioning utilizing PuLP solver.

Program will output cumulative cost and savings information in .png file

**Inputs**

Currently codebase has been set to input information for two years (one year per file) from the following sources:
- Electricity price information from Entso-E in .csv format
- Solar panel production from PVWatss in .csv format
- Solar panel production data from Fingrid Datahub has been commented out
- Houly consumption data from Fingrid Datahub. Converter from 15 min data to 60 min data is included as separate program

**Set up**

To set up program the following parameters should be considered defined for simulation:
- flex_load_percentage can be used to define flexible load which can be moved to cheapest hours. Default 0 will not include any load shifting
- battery_capacity_kWh is used to define size of battery in kWh. Defauls is 5 and 0 would simulate system with only solar panels
- max_charge_discharge_rate is used to define max discharge rate of battery in kW. Default is 2.5
- start_date is used to define first day of optimization calculation
- Last for loop in main program branch defines how many days are simulated. Default is 364
