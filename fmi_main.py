import xml.etree.ElementTree as ET
import csv
import math
from datetime import datetime, timedelta, timezone

from fmi_weather import (
    fetch_location_data, fetch_radiation_data, fetch_weather_data
)
from fmi_pvwatts import (
    calculate_equation_of_time, calculate_solar_declination, calculate_solar_time, calculate_hour_angle,
    calculate_solar_zenith, calculate_solar_azimuth, calculate_extraterrestrial_irradiance,
    calculate_diffuse_horizontal_irradiance, calculate_POA_irradiance, estimate_cell_temperature,
    calculate_power_output, calculate_diffuse_irradiance
)

# Define Main Variables
start_date = '2024-05-06'  # Given in date without time
days_to_count = 42  # Collect details ahead from start date, max 42 days at the time 
location = 100949  # Location ID to fetch data available in FMI
pv_total_kwp = 10  # PV system total kW peak
tilt_angle = 30  # Tilt angle for PV modules
azimuth_degree = 205  # Azimuth degree for PV modules
inverter_efficiency = 0.96  # Typical inverter efficiency, no need to change
temp_coefficient = -0.0035 # no need to change
standard_meridian = 30  # in Finland no need to change

# Fetch geo-coordinates and radiation & weather data
location_name, latitude, longitude = fetch_location_data(location)
latitude = float(latitude)  # Ensure latitude is float
longitude = float(longitude)  # Ensure longitude is float

radiation_root = fetch_radiation_data(start_date, days_to_count, location)
weather_root = fetch_weather_data(start_date, days_to_count, location)

if not radiation_root or not weather_root:
    print("Failed to fetch radiation or weather data")
    exit(1)

# Namespaces for parsing XML
radiation_namespaces = {'wml2': 'http://www.opengis.net/waterml/2.0'}
weather_namespaces = {'wml2': 'http://www.opengis.net/waterml/2.0'}

# Extract radiation and weather data
def extract_radiation_and_weather_data(radiation_root, weather_root, radiation_namespaces, weather_namespaces):
    radiation_times = radiation_root.findall('.//wml2:time', radiation_namespaces)
    radiation_values = radiation_root.findall('.//wml2:value', radiation_namespaces)
    temperatures = []
    wind_speeds = []

    for series in weather_root.findall('.//wml2:MeasurementTimeseries', weather_namespaces):
        series_id = series.attrib['{http://www.opengis.net/gml/3.2}id']
        if 't2m' in series_id:
            temps = series.findall('.//wml2:MeasurementTVP/wml2:value', weather_namespaces)
            temperatures.extend(temp.text for temp in temps)
        elif 'ws_10min' in series_id:
            winds = series.findall('.//wml2:MeasurementTVP/wml2:value', weather_namespaces)
            for wind in winds:
                wind_speed = float(wind.text) if not math.isnan(float(wind.text)) else 0
                wind_speeds.append(wind_speed)
            #wind_speeds.extend(wind.text for wind in winds)

    return radiation_times, radiation_values, temperatures, wind_speeds

radiation_times, radiation_values, temperatures, wind_speeds = extract_radiation_and_weather_data(
    radiation_root, weather_root, radiation_namespaces, weather_namespaces)

# Albedo values by month
albedo_values = {1: 0.6, 2: 0.6, 3: 0.53, 4: 0.15, 5: 0.16, 6: 0.17,
                 7: 0.17, 8: 0.16, 9: 0.19, 10: 0.19, 11: 0.19, 12: 0.39}

# Combined CSV writing function
import csv
import math
from datetime import datetime

# Combined CSV writing function
def write_combined_data_to_csv(radiation_times, radiation_values, temperatures, wind_speeds, filename):
    fieldnames = [
        'Month', 'Day', 'Hour', 'Beam Irradiance (W/m²)', 'Diffuse Irradiance (W/m²)', 'Ambient Temperature (°C)', 'Wind Speed (m/s)',
        'Albedo', 'Plane of Array Irradiance (W/m²)', 'Cell Temperature (°C)', 
        'DC Array Output (W)', 'AC System Output (W)'
    ]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Determine the minimum length of the data lists to avoid IndexError
        length = min(len(radiation_times), len(radiation_values), len(temperatures), len(wind_speeds))

        for i in range(length):
            timestamp = datetime.strptime(radiation_times[i].text, "%Y-%m-%dT%H:%M:%S%z")
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour
            beam_irradiance = float(radiation_values[i].text)
            ambient_temp = float(temperatures[i])
            wind_speed = float(wind_speeds[i])
            
            # Handle NaN values in wind speed
            if math.isnan(wind_speed):
                wind_speed = 0

            albedo = albedo_values[month]

            # Solar calculations
            day_of_year = timestamp.timetuple().tm_yday
            solar_zenith, solar_declination = calculate_solar_zenith(latitude, day_of_year, hour, longitude, standard_meridian)
            equation_of_time = calculate_equation_of_time(day_of_year)
            solar_azimuth = calculate_solar_azimuth(latitude, hour, solar_declination, solar_zenith, longitude, standard_meridian, equation_of_time)
            extraterrestrial_irradiance = calculate_extraterrestrial_irradiance(day_of_year, solar_zenith)
            global_horizontal_irradiance = beam_irradiance / math.cos(math.radians(solar_zenith)) if math.cos(math.radians(solar_zenith)) > 0 else 0
            diffuse_horizontal_irradiance = calculate_diffuse_horizontal_irradiance(global_horizontal_irradiance, extraterrestrial_irradiance)
            POA_irradiance = calculate_POA_irradiance(beam_irradiance, diffuse_horizontal_irradiance, tilt_angle, azimuth_degree, solar_zenith, solar_azimuth, albedo)
            cell_temp = estimate_cell_temperature(POA_irradiance, ambient_temp, wind_speed)
            AC_output, DC_output = calculate_power_output(pv_total_kwp, POA_irradiance, cell_temp, inverter_efficiency, temp_coefficient)
            diffuse_irradiance = calculate_diffuse_irradiance(global_horizontal_irradiance, beam_irradiance, solar_zenith)

            writer.writerow({
                'Month': month, 'Day': day, 'Hour': hour, 'Beam Irradiance (W/m²)': beam_irradiance,
                'Diffuse Irradiance (W/m²)': diffuse_irradiance, 'Ambient Temperature (°C)': ambient_temp, 
                'Wind Speed (m/s)': wind_speed, 'Albedo': albedo, 'Plane of Array Irradiance (W/m²)': POA_irradiance, 
                'Cell Temperature (°C)': cell_temp, 'DC Array Output (W)': DC_output, 'AC System Output (W)': AC_output
            })

# Output filename
output_filename = f"pvwatts{start_date}_days_{days_to_count}_{pv_total_kwp}kWp_{location_name}.csv"

# Write to CSV
write_combined_data_to_csv(radiation_times, radiation_values, temperatures, wind_speeds, output_filename)