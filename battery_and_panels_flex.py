import pandas as pd
import matplotlib.pyplot as plt
import pulp
from datetime import datetime, time

##################################################
###Beginning of data loading segment##############
##################################################

## instruction to use
## activate venv -> source venv/bin/activate
## deactivate venv -> deactivate
## https://oma.datahub.fi/#/ consumption data
## PV production estimate fmi_main.py estimated PV production


# Start date
str_start_date = '2023-06-01' #Set start date
days_to_count = 1 #Count dates from start date

# Define battery parameters
battery_capacity_kWh = 15 #Battery capacity in kWh
max_charge_discharge_rate = 10 #Max battery discharge rate in kW
pv_total_kwp = 10 # PV panel total

distribution = 0.0507
tax = 0.0279372
margin = 0.0045
night_time_distribution = False  # DONT USE, under development - set true if night-time distribution fee is in use
distribution_fee = 0.0492  # Example value, replace with your actual daytime or single distribution fee
night_time_distribution_fee = 0.0301  # Example value, replace with your actual night-time dostribution fee value
night_time_distribution_start = "22:00"  # Example value, replace with your actual start time
night_time_distribution_end = "06:59"  # Example value, replace with your actual end time


max_value = 100  # Example maximum value, replace with your actual max value
min_value = -100  # Example maximum value, replace with your actual max value
ticks = 10 # Example tick value, replace with your actual tick value
flex_load = 10 # 20 # Flexible load in percentage




# Function to load and preprocess price CSV file
def load_and_preprocess_prices(file_path):
    df = pd.read_csv(file_path)
    # Parse "MTU (CET/CEST)" without timezone initially
    df['MTU (CET/CEST)'] = pd.to_datetime(df['MTU (CET/CEST)'].str[:16], format='%d.%m.%Y %H:%M')
    
    # Localize to CET/CEST, handling ambiguous times by shifting backward
    df['MTU (CET/CEST)'] = df['MTU (CET/CEST)'].dt.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_backward')
    
    # Drop rows with NaT (ambiguous) timestamps if any
    df = df.dropna(subset=['MTU (CET/CEST)'])
    
    # Convert to UTC
    df['MTU (CET/CEST)'] = df['MTU (CET/CEST)'].dt.tz_convert('UTC')
    df["Day-ahead Price [EUR/kWh]"] = pd.to_numeric(df["Day-ahead Price [EUR/MWh]"], errors='coerce') / 1000
    return df.set_index('MTU (CET/CEST)')

# Adjusted function to load and preprocess consumption CSV file
def load_and_preprocess_consumption(file_path):
    df = pd.read_csv(file_path, sep=';', decimal=',')
    # Assuming the "Alkuaika" column is already in UTC
    df['Alkuaika'] = pd.to_datetime(df['Alkuaika'], utc=True)
    df.rename(columns={'Alkuaika': 'Timestamp', 'Määrä': 'Consumption'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    return df

def load_and_preprocess_solar(file_path, year):
    df_solar = pd.read_csv(file_path)
    
    # Construct a datetime index from the Month, Day, Hour columns with the correct year
    df_solar['Timestamp'] = pd.to_datetime(df_solar[['Month', 'Day']].assign(Year=year), errors='coerce') + pd.to_timedelta(df_solar['Hour'], unit='h')
    
    # Localize the Timestamp to Finland's timezone without considering daylight saving time initially
    # Adjust the localization to handle DST if necessary, depending on your analysis needs
    df_solar['Timestamp'] = df_solar['Timestamp'].dt.tz_localize('Etc/GMT-2', ambiguous='infer')
    
    # Set the timestamp as the index
    df_solar.set_index('Timestamp', inplace=True)
    
    # Select relevant columns for solar production
    df_solar = df_solar[['AC System Output (W)']]  # Adjust based on what you need
    
    return df_solar

# Fingrid datahub production import
#def load_and_preprocess_solar(file_path, year):
#    df_solar = pd.read_csv(file_path, sep=';', decimal=',')
#    # Assuming the "Alkuaika" column is already in UTC
#    df_solar['Alkuaika'] = pd.to_datetime(df_solar['Alkuaika'], utc=True)
#    df_solar.rename(columns={'Alkuaika': 'Timestamp', 'Määrä': 'AC System Output (W)'}, inplace=True)
#    df_solar.set_index('Timestamp', inplace=True)    
#    return df_solar



# Load and preprocess the data for both years
price_df_2022 = load_and_preprocess_prices('price2022.csv')
price_df_2023 = load_and_preprocess_prices('price2023.csv')
price_df_2024 = load_and_preprocess_prices('price2024.csv')
consumption_df_2022 = load_and_preprocess_consumption('consumption2022.csv')
consumption_df_2023 = load_and_preprocess_consumption('consumption2023.csv')
consumption_df_2024 = load_and_preprocess_consumption('consumption2024.csv')
# Load and preprocess solar data for both years with correct years passed
solar_df_2022 = load_and_preprocess_solar('pvwatts.csv', 2022)
solar_df_2023 = load_and_preprocess_solar('pvwatts.csv', 2023)
solar_df_2024 = load_and_preprocess_solar('pvwatts.csv', 2024)


# Concatenate the DataFrames
df_combined_prices = pd.concat([price_df_2022, price_df_2023, price_df_2024])
df_combined_consumption = pd.concat([consumption_df_2022, consumption_df_2023, consumption_df_2024])
df_combined_solar = pd.concat([solar_df_2022, solar_df_2023, solar_df_2024])

# Merge price and consumption data on index (timestamp)
df_merged = pd.merge(df_combined_prices, df_combined_consumption, left_index=True, right_index=True, how='inner')

# Ensure this calculation is performed to add 'Total Price [EUR/kWh]' to df_merged
# df_merged['Total Price [EUR/kWh]'] = (df_merged["Day-ahead Price [EUR/kWh]"] * 1.24) + distribution_fee + tax + margin
if night_time_distribution:
    # Convert start and end times to datetime objects
    start_time = datetime.strptime(night_time_distribution_start, "%H:%M")
    end_time = datetime.strptime(night_time_distribution_end, "%H:%M")

    # Calculate price based on time of day
    df_merged['Total Price [EUR/kWh]'] = (df_merged["Day-ahead Price [EUR/kWh]"] * 1.24) + \
                                          df_merged.apply(lambda row: night_time_distribution_fee if start_time <= row['Time'] <= end_time else distribution_fee, axis=1) + \
                                          tax + margin
else:
    df_merged['Total Price [EUR/kWh]'] = (df_merged["Day-ahead Price [EUR/kWh]"] * 1.24) + distribution_fee + tax + margin


# Calculate 'Hourly Cost [EUR]' based on the 'Total Price [EUR/kWh]' and 'Consumption'
df_merged['Hourly Cost [EUR]'] = df_merged['Total Price [EUR/kWh]'] * df_merged['Consumption']

# Renaming the 'Hourly Cost [EUR]' column to reflect costs without solar panel production
df_merged.rename(columns={'Hourly Cost [EUR]': 'Hourly Cost Without Panels [EUR]'}, inplace=True)



# Merge solar production data with the combined price and consumption data on index (timestamp)
df_final = pd.merge(df_merged, df_combined_solar, left_index=True, right_index=True, how='outer')

# Converting 'AC System Output (W)' to kWh
df_final['Solar Production [kWh]'] = df_final['AC System Output (W)'] / 1000 * 1

# Calculating net consumption; if production exceeds consumption, this will be negative, indicating a credit
df_final['Net Consumption [kWh]'] = df_final['Consumption'] - df_final['Solar Production [kWh]']

# Calculating hourly cost with solar panels, applying credit if production exceeds consumption
df_final['Hourly Cost With Panels [EUR]'] = df_final.apply(lambda row: row['Net Consumption [kWh]'] * row['Total Price [EUR/kWh]'] if row['Net Consumption [kWh]'] >= 0 else row['Net Consumption [kWh]'] * row['Day-ahead Price [EUR/kWh]'], axis=1)

df_final = df_final[~df_final.index.duplicated(keep='first')]

# Ensure the data is sorted by index (timestamp) if not already
df_final_sorted = df_final.sort_index()
##################################################
#########End of data loading segment##############
##################################################



##################################################
###Beginning of problem statement segment#########
##################################################

def optimize_with_flexibility(df_segment, battery_capacity_kWh, max_charge_discharge_rate, initial_SoC, day_total_energy, consumed_before_14):

    """
    Optimizes battery charging and discharging within a segment, ensuring realistic SoC management,
    with the objective to minimize the net cost of electricity, considering solar production.
    Battery cannot be charged and discharged during the same hour, and discharging is limited by current SoC.
    """



    flex_load_percentage = flex_load/100 #0.2 #Share of fully flexible load

    # Initialize problem
    problem = pulp.LpProblem("Daily_Cost_Minimization_with_Flexibility", pulp.LpMinimize)

    # Decision variables
    is_charging = pulp.LpVariable.dicts("is_charging", df_segment.index, cat='Binary')
    is_discharging = pulp.LpVariable.dicts("is_discharging", df_segment.index, cat='Binary')
    charge_amounts = pulp.LpVariable.dicts("Charge", df_segment.index, 0, max_charge_discharge_rate, cat='Continuous')
    discharge_amounts = pulp.LpVariable.dicts("Discharge", df_segment.index, 0, max_charge_discharge_rate, cat='Continuous')
    solar_to_battery = pulp.LpVariable.dicts("SolarToBattery", df_segment.index, 0, cat='Continuous')
    solar_to_grid = pulp.LpVariable.dicts("SolarToGrid", df_segment.index, 0, cat='Continuous')
    solar_to_immediate_use = pulp.LpVariable.dicts("SolarToImmediateUse", df_segment.index, 0, cat='Continuous')


    # Assume grid_charging_amounts and solar_charging_amounts as new decision variables
    grid_charging_amounts = pulp.LpVariable.dicts("GridCharging", df_segment.index, 0, max_charge_discharge_rate, cat='Continuous')
    solar_charging_amounts = pulp.LpVariable.dicts("SolarCharging", df_segment.index, 0, max_charge_discharge_rate, cat='Continuous')
    flex_consumption_amounts = pulp.LpVariable.dicts("FlexConsumption", df_segment.index, lowBound=0, cat='Continuous')
    inflex_consumption_amounts = {t: (1 - flex_load_percentage) * df_segment.loc[t, 'Consumption'] for t in df_segment.index}

    # Calculate the target sum for flexible consumption
    target_flex_consumption = day_total_energy * flex_load_percentage


    # Auxiliary variables for discharge types
    discharge_within_consumption = pulp.LpVariable.dicts("DischargeWithinConsumption", df_segment.index, 0, max_charge_discharge_rate, cat='Continuous')
    discharge_excess = pulp.LpVariable.dicts("DischargeExcess", df_segment.index, 0, max_charge_discharge_rate, cat='Continuous')

    # State of Charge (SoC) constraints and variables
    soc_variables = pulp.LpVariable.dicts("SoC", df_segment.index, 0, battery_capacity_kWh, cat='Continuous')
    

    net_costs = []  # To accumulate net costs for the objective function
    hourly_costs = {}
    solar_to_immediate_use = {}
    solar_to_flex_use = {}
    for t in df_segment.index:
        row = df_segment.loc[t]  # Access the row directly using 't'
        soc_variables[t] = pulp.LpVariable(f"SoC_{t}", 0, battery_capacity_kWh, cat='Continuous')
        if t == df_segment.index[0]:
            problem += soc_variables[t] == initial_SoC
        else:
            prev_t = df_segment.index[df_segment.index.get_loc(t)-1]
            problem += soc_variables[t] == soc_variables[prev_t] + charge_amounts[prev_t] + solar_to_battery[prev_t] - discharge_amounts[prev_t]
        
        # Now directly use 'row' to access DataFrame information, such as solar production
        solar_production = row['Solar Production [kWh]']   
        # Correctly using dictionaries with 't' as the key
        solar_to_immediate_use[t] = pulp.LpVariable(f"solar_to_immediate_use_{t}", 0, None, cat='Continuous')
        solar_to_flex_use[t] = pulp.LpVariable(f"solar_to_flex_use_{t}", 0, None, cat='Continuous')

        # Continue with other operations, replacing direct timestamp indexing with 'i' for lists
        # and using 'row' for DataFrame operations as needed


        # Constraint to ensure solar allocation does not exceed total consumption or solar production
        problem += solar_to_immediate_use[t] + solar_to_flex_use[t] + solar_to_battery[t] + solar_to_grid[t] == df_segment.loc[t, 'Solar Production [kWh]']
        problem += solar_to_immediate_use[t] + solar_to_flex_use[t] <= inflex_consumption_amounts[t] + flex_consumption_amounts[t]

        # Additional constraints remain unchanged, ensuring solar to flex use is properly limited
        problem += solar_to_flex_use[t] <= flex_consumption_amounts[t]
    
        # Constraints for solar allocation and battery operation
        problem += charge_amounts[t] <= is_charging[t] * max_charge_discharge_rate
        problem += discharge_amounts[t] <= is_discharging[t] * max_charge_discharge_rate
        problem += discharge_within_consumption[t] + discharge_excess[t] == discharge_amounts[t]
        problem += discharge_within_consumption[t] <= df_segment.loc[t, 'Consumption']
        # Updated constraints for solar allocation to include immediate use
        problem += solar_to_battery[t] + solar_to_grid[t] + solar_to_immediate_use[t] <= df_segment.loc[t, 'Solar Production [kWh]']

        # Constraint to ensure solar to immediate use does not exceed consumption
        problem += solar_to_immediate_use[t] <= df_segment.loc[t, 'Consumption']


        # Prevent charging and discharging in the same hour
        problem += is_charging[t] + is_discharging[t] <= 1

        problem += charge_amounts[t] == grid_charging_amounts[t] + solar_charging_amounts[t]
        problem += solar_charging_amounts[t] <= df_segment.loc[t, 'Solar Production [kWh]']
        # Constraints for solar allocation and battery operation
        problem += charge_amounts[t] <= is_charging[t] * max_charge_discharge_rate
        problem += discharge_amounts[t] <= is_discharging[t] * max_charge_discharge_rate
        problem += discharge_within_consumption[t] + discharge_excess[t] == discharge_amounts[t]
        problem += discharge_within_consumption[t] <= df_segment.loc[t, 'Consumption']
        # Updated constraints for solar allocation to include immediate use
        problem += solar_to_battery[t] + solar_to_grid[t] + solar_to_immediate_use[t] == df_segment.loc[t, 'Solar Production [kWh]']

        # Constraint to ensure solar to immediate use does not exceed consumption
        problem += solar_to_immediate_use[t] <= df_segment.loc[t, 'Consumption']


        # Prevent charging and discharging in the same hour
        problem += is_charging[t] + is_discharging[t] <= 1

        problem += charge_amounts[t] == grid_charging_amounts[t] + solar_charging_amounts[t]
        problem += solar_charging_amounts[t] <= df_segment.loc[t, 'Solar Production [kWh]']


        # Discharge limit based on SoC
        if t > df_segment.index[0]:  # For all but the first period
            problem += discharge_amounts[t] <= soc_variables[prev_t]

        # Adjust net cost calculation to include the benefit of solar to immediate use

        grid_charging_cost = grid_charging_amounts[t] * df_segment.loc[t, 'Total Price [EUR/kWh]'] * 1.1
        solar_charging_cost = solar_charging_amounts[t] * df_segment.loc[t, 'Day-ahead Price [EUR/kWh]'] * 1.1
        problem += grid_charging_amounts[t] + solar_charging_amounts[t] <= max_charge_discharge_rate
        problem += solar_to_grid[t] <= df_segment.loc[t, 'Solar Production [kWh]'] - solar_charging_amounts[t]


        # Calculate revenues from discharging
        discharge_excess_revenue = discharge_excess[t] * df_segment.loc[t, 'Day-ahead Price [EUR/kWh]'] * 0.9
        discharge_within_consumption_revenue = discharge_within_consumption[t] * df_segment.loc[t, 'Total Price [EUR/kWh]'] * 0.9

        # Assuming 'Net Consumption [kWh]' includes consumption net of solar production
        inflex_consumption_cost = inflex_consumption_amounts[t] * df_segment.loc[t, 'Total Price [EUR/kWh]']
        flex_consumption_cost = flex_consumption_amounts[t] * df_segment.loc[t, 'Total Price [EUR/kWh]']

        # Revenue from selling solar to the grid
        solar_to_grid_revenue = solar_to_grid[t] * df_segment.loc[t, 'Day-ahead Price [EUR/kWh]']
        own_use_benefit = (solar_to_immediate_use[t] + solar_to_flex_use[t]) * df_segment.loc[t, 'Total Price [EUR/kWh]'] 
 
        # Total cost for the hour (considering consumption, charging costs, and discharging revenues)
        hourly_cost = inflex_consumption_cost + flex_consumption_cost + grid_charging_cost + solar_charging_cost - discharge_excess_revenue - solar_to_grid_revenue - own_use_benefit - discharge_within_consumption_revenue
        hourly_costs[t] = hourly_cost
        
        # Flexible consumption within allowed limits (set by 20% of original load)
        problem += flex_consumption_amounts[t] <= 5 

        # Adjust net cost calculation to include solar charging at day-ahead price and grid charging at total price
        net_costs.append(inflex_consumption_cost + flex_consumption_cost + grid_charging_cost + solar_charging_cost - solar_to_grid_revenue - discharge_excess_revenue - own_use_benefit - discharge_within_consumption_revenue)
        # Objective: Minimize the total net cost of electricity


    problem += pulp.lpSum([flex_consumption_amounts[t] for t in df_segment.index]) == target_flex_consumption, "TotalFlexConsumption"
    problem += pulp.lpSum(net_costs)

    # Solve the problem
    problem.solve()

    # Evaluate the hourly costs to get their numerical values
    evaluated_hourly_costs = {t: pulp.value(hourly_costs[t]) for t in df_segment.index}


    # Extract optimized decisions, costs, and solar allocations
    optimized_decisions = {
        "Charging": {t: charge_amounts[t].varValue for t in df_segment.index},
        "Discharging": {t: discharge_amounts[t].varValue for t in df_segment.index},
        "Discharge within Consumption": {t: discharge_within_consumption[t].varValue for t in df_segment.index},
        "Discharge Excess": {t: discharge_excess[t].varValue for t in df_segment.index},

        "Solar to Battery": {t: solar_to_battery[t].varValue for t in df_segment.index},
        "Solar to Grid": {t: solar_to_grid[t].varValue for t in df_segment.index},
        "Solar to Immediate Use": {t: pulp.value(solar_to_immediate_use[t]) for t in df_segment.index},
        "Solar to Flex Use": {t: pulp.value(solar_to_flex_use[t]) for t in df_segment.index},
        "Total Net Cost": pulp.value(problem.objective),
        "Status": pulp.LpStatus[problem.status],
        "Hourly Costs": evaluated_hourly_costs,  # Include hourly costs here
        "flex_consumption": {t: pulp.value(flex_consumption_amounts[t]) for t in df_segment.index},
        "inflex_consumption": inflex_consumption_amounts,
        'SoC': {t: pulp.value(soc_variables[t]) for t in df_segment.index}

    }

    return optimized_decisions

##################################################
#########End of problem statement segment#########
##################################################

##################################################
###Beginning of parameters definition#########
##################################################



# Define battery parameters
#battery_capacity_kWh = 90 #Battery capacity in kWh
#max_charge_discharge_rate = 15 #Max battery discharge rate in kW
#pv_total_kwp = 15 # PV panel total

# Initialize the start date and time for the first segment
start_date = pd.to_datetime(str_start_date).tz_localize('UTC')
#start_date = pd.to_datetime('2023-04-01').tz_localize('UTC')  #Define start date for calculation
cumulative_total_profit = 0  # Initialize cumulative profit
profits_first_24_hours = []  # Initialize list to store profits for the first 24 hours of each segment
initial_SoC = 20 # Starting SoC, adjust based on your actual starting conditions
day_total_energy = 0
consumed_before_14 = 0 
battery_discharging_hours = 0
battery_charging_hours = 0

# Initialize lists for plotting
cumulative_profits = []
cumulative_revenues_within = []
cumulative_revenues_excess = []
segment_dates = []
cumulative_total_net_cost = []  # Initialize as an empty list
cumulative_charging_costs_list = []
cumulative_solar_to_grid_list = []
cumulative_hourly_costs_list = []

# Initialize variables to store cumulative values
cumulative_profit = 0.0
cumulative_revenue_within = 0.0
cumulative_revenue_excess = 0.0
# Initialize a variable to track the cumulative total net cost as a scalar
total_net_cost_cumulative = 0.0  # This is the correct way to track the cumulative total
cumulative_charging_costs = 0  # Initialize cumulative charging cost
cumulative_solar_to_grid = 0
cumulative_hourly_cost = 0


##################################################
##########Beginning of daily calculations#########
##################################################
#For loop defines how many days is going to be iterated

for i in range(days_to_count): # default 364
    datetime_format = '%Y-%m-%d %H:%M:%S+00:00'  # Adjust format as needed

    # Define segment start and end times
    segment_start = (start_date + pd.DateOffset(days=i)).normalize() + pd.Timedelta(hours=14)
    segment_end = segment_start + pd.DateOffset(days=1) + pd.Timedelta(hours=9, minutes=59, seconds=59)
    print(f"Segment Detailed Decisions:")
    print(f"Start: {segment_start}, End: {segment_end}\n")
    print("Hour\t\t\tCharging (kWh)\tDischarging (kWh)\tSoC\t\tProfit (EUR)\tDay-ahead Price (EUR/kWh)")

    # Ensure no NaN values in the segment
    df_segment = df_final_sorted.loc[segment_start:segment_end].fillna(0)


    # Calculate the energy consumed before 14:00 UTC
    #consumed_before_14 = df_segment.between_time('00:00', '13:59')['Consumption'].sum()




    # Optimize the segment with the additional parameters
    optimized_decisions = optimize_with_flexibility(
        df_segment, 
        battery_capacity_kWh, 
        max_charge_discharge_rate, 
        initial_SoC, 
        day_total_energy, 
        consumed_before_14
    )
    sorted_soc_times = sorted(optimized_decisions['SoC'].keys())

    # Calculate start time for consumption as midnight on the second day
    start_datetime_for_consumption = (segment_start + pd.DateOffset(days=1)).normalize()
    
    # Calculate end time for consumption as 14:00 on the second day
    end_datetime_for_consumption = start_datetime_for_consumption + pd.Timedelta(hours=13)
    
    # Initialize consumed_before_14 for this segment
    consumed_before_14 = 0


    
    # Iterate through the flex and inflex consumption within the desired time range
    for t, _ in df_segment.iterrows():
        if start_datetime_for_consumption <= t <= end_datetime_for_consumption:
            # Assuming t is a datetime index in df_segment that matches keys in optimized_decisions
            flex_value = optimized_decisions['flex_consumption'].get(t.strftime('%Y-%m-%d %H:%M:%S+00:00'), 0) 
            inflex_value = optimized_decisions['inflex_consumption'][t]  # Direct access since inflex_consumption uses df_segment's index
            
            consumed_before_14 += flex_value + inflex_value

    # Calculate start time for consumption as midnight on the second day
    start_datetime_for_total_consumption = (segment_start + pd.DateOffset(days=1)).normalize()
    
    # Calculate end time for consumption as 14:00 on the second day
    end_datetime_for_total_consumption = start_datetime_for_consumption + pd.Timedelta(hours=23)

   

    # Filter df_segment for the specified range
    filtered_segment = df_segment.loc[start_datetime_for_total_consumption:end_datetime_for_total_consumption]
    # Calculate day_total_energy from the filtered_segment
    day_total_energy = filtered_segment['Consumption'].sum()

    




    # Ensure there are enough entries to extract the 25th SoC value
    if len(sorted_soc_times) >= 25:
        # The 25th value, given the list starts at 14:00, should represent the SoC at 14:00 the next day
        twenty_fifth_time = sorted_soc_times[24]  # Indexing starts at 0, so 24 represents the 25th item
        initial_SoC = optimized_decisions['SoC'][twenty_fifth_time]
    else:
        # Handle cases where there are not enough SoC entries (e.g., fallback or error handling)
        print("Not enough SoC entries to update initial SoC for the next segment.")



    # Step 2: Extract "Total Net Cost" from optimized decisions
    total_net_cost = optimized_decisions.get("Total Net Cost", 0)
    

    # Update initial_SoC for the next segment based on the ending SoC
    next_day_14_time = segment_start + pd.DateOffset(days=1)
    next_day_14_time = next_day_14_time.replace(hour=14, minute=0, second=0, microsecond=0)
    # At the end of each segment processing, you have:
    next_day_14_str = next_day_14_time.strftime(datetime_format)
    SoC_at_14 = optimized_decisions["SoC"].get(next_day_14_str, initial_SoC)
    
    segment_profit = 0
    segment_revenue_within = 0
    segment_revenue_excess = 0
    segment_charging_cost = 0  # Initialize segment charging cost
    segment_revenue_solar_to_grid = 0
    segment_evaluated_hourly_costs = 0
    for hour, t in enumerate(df_segment.index[:24]):
        charge = optimized_decisions["Charging"].get(t, 0)
        discharge = optimized_decisions["Discharging"].get(t, 0)

        inflex_cons = optimized_decisions["inflex_consumption"].get(t, 0)
        flex_cons = optimized_decisions["flex_consumption"].get(t, 0)
        discharge_within = optimized_decisions["Discharge within Consumption"].get(t, 0)
        discharge_excess = optimized_decisions["Discharge Excess"].get(t, 0)
        SoC = optimized_decisions["SoC"].get(t, initial_SoC)
        price1 = df_segment.at[t, 'Day-ahead Price [EUR/kWh]']
        price2 = df_segment.at[t, 'Total Price [EUR/kWh]']
        solar_to_grid = optimized_decisions["Solar to Grid"].get(t, 0)
        solar_to_battery = optimized_decisions["Solar to Battery"].get(t, 0)
        evaluated_hourly_costs = optimized_decisions["Hourly Costs"].get(t, 0)

        # Check if the battery is discharging this hour and increment the counter if so
        if discharge > 0:
            battery_discharging_hours += 1

        if charge > 0:
            battery_charging_hours += 1

    
        cost = charge * price2 + solar_to_battery * price1  # Cost for charging
        revenue_within = discharge_within * price2  # Revenue from discharging within consumption
        revenue_excess = discharge_excess * price1  # Revenue from discharging excess

        profit = revenue_within + revenue_excess - cost  # Profit for this timestamp
        revenue_solar_to_grid = solar_to_grid * price1  # Profit for this timestamp
    
        # Accumulate segment values
        segment_profit += profit
        segment_revenue_within += revenue_within
        segment_revenue_excess += revenue_excess
        segment_revenue_solar_to_grid += revenue_solar_to_grid
        segment_evaluated_hourly_costs += evaluated_hourly_costs
        # Accumulate charging cost for the segment
        segment_charging_cost += cost    
 #       print(f"{t}\tCharge: {charge:.2f}\tDischarge: {discharge:.2f}\tDischarge within: {discharge_within:.2f}\tDischarge excess: {discharge_excess:.2f}\tSoC: {SoC:.2f}\tProfit: {profit:.2f}")
    
    

    print(f"\nTotal profit for segment {i+1}: {segment_profit:.2f} EUR")
    print(f"State of Charge at 14:00 next day: {SoC_at_14:.2f} kWh\n")
    print(f"Total Net Cost for segment {i+1}: {total_net_cost:.2f} EUR")


    # Update initial_SoC for the next segment
    initial_SoC = SoC_at_14
    cumulative_profit += segment_profit
    cumulative_revenue_within += segment_revenue_within
    cumulative_revenue_excess += segment_revenue_excess
    # After processing all hours in the segment, add the segment's charging cost to the cumulative total
    cumulative_charging_costs += segment_charging_cost 
    cumulative_solar_to_grid += segment_revenue_solar_to_grid
    cumulative_hourly_cost += segment_evaluated_hourly_costs
    
    total_net_cost = optimized_decisions.get("Total Net Cost", 0)
    total_net_cost_cumulative += total_net_cost  # Add each segment's cost to the cumulative total

    # Append the updated cumulative total to the list
    cumulative_total_net_cost.append(total_net_cost_cumulative)


    cumulative_profits.append(cumulative_profit)
    cumulative_revenues_within.append(cumulative_revenue_within)
    cumulative_revenues_excess.append(cumulative_revenue_excess)
    cumulative_charging_costs_list.append(cumulative_charging_costs)
    cumulative_solar_to_grid_list.append(cumulative_solar_to_grid)
    cumulative_hourly_costs_list.append(cumulative_hourly_cost)


    segment_dates.append(segment_start)  # Assuming you want to plot against segment start dates


# Print cumulative totals after all segments
print(f"Cumulative profit from battery: {cumulative_profit:.2f} EUR")
print(f"Cumulative revenue within consumption: {cumulative_revenue_within:.2f} EUR")
print(f"Cumulative revenue excess: {cumulative_revenue_excess:.2f} EUR")
# Corrected print statement to display the last element of the list, formatted as a floating-point number
#print(f"Cumulative Total Net Cost from first 24 hours of all cycles: {cumulative_total_net_cost[-1]:.2f} EUR")
print(f"Cumulative charging cost over all segments: {cumulative_charging_costs:.2f} EUR")
print(f"Cumulative solar to grid: {cumulative_solar_to_grid:.2f} EUR")
print(f"Cumulative cost: {cumulative_hourly_cost:.2f} EUR")
# After the loop
print(f"Battery Discharging Hours in the first 24 hours: {battery_discharging_hours}")
# After the loop
print(f"Battery Charging Hours in the first 24 hours: {battery_charging_hours}")

# Plotting with y-axis minimum set to 0
plt.figure(figsize=(13, 7))
plt.plot(segment_dates, cumulative_profits, label='Cumulative Profit')
#plt.plot(segment_dates, cumulative_revenues_within, label='Cumulative Revenue Own Use')
plt.plot(segment_dates, cumulative_revenues_excess, label='Cumulative Revenue Exceses')
#plt.plot(segment_dates, cumulative_charging_costs_list, label='Cumulative Charging cost')
plt.plot(segment_dates, cumulative_solar_to_grid_list, label='Cumulative Solar to Grid')
plt.plot(segment_dates, cumulative_hourly_costs_list, label='Cumulative Costs', linestyle='--', color='red')
#plt.plot(segment_dates, cumulative_total_net_cost, label='Cumulative Total Net Cost', linestyle='--', color='red')  # Adding the cumulative total net cost plot

plt.xlabel('Date')
plt.ylabel('EUR')
#plt.title('Cumulative Profits and Revenues Over Time')
plt.title(f"Cumulative Profits and Revenues Over Time,\nPV power {pv_total_kwp:.2f} kWp, Batt {battery_capacity_kWh:.2f} kWh, max dis/chrg rate {int(max_charge_discharge_rate):.2f} kW")
plt.legend()
plt.xticks(rotation=45)

#max_value = 1200  # Example maximum value, replace with your actual max value
#min_value = -700  # Example maximum value, replace with your actual max value
#ticks = 100 # Example tick value, replace with your actual tick value


plt.ylim(bottom=min_value)  # Set the minimum value of the y-axis
y_ticks = range(min_value, max_value + 100, ticks)  # Generates ticks from min_value to max_value with a step of 200€

# Set the y-ticks
plt.yticks(y_ticks)

# Enable the grid for y-axis
plt.grid(axis='y')

plt.tight_layout()
#plt.show()
# Save the plot as an image file
# plot_filename = "cumulative_profits_over_time.png"
plot_filename = f"cumulative_profits_{str_start_date}_pv_{pv_total_kwp}kWp_batt_{battery_capacity_kWh}kWh.png"
plt.savefig(plot_filename)
