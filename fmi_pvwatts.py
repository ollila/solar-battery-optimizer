import math
import numpy as np

# Constants
SOLAR_CONSTANT = 1367  # W/m^2

def calculate_equation_of_time(day_of_year):
    B = 360 * (day_of_year - 81) / 365
    EoT = 9.87 * math.sin(math.radians(2 * B)) - 7.53 * math.cos(math.radians(B)) - 1.5 * math.sin(math.radians(B))
    return EoT

def calculate_solar_declination(day_of_year):
    return 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

def calculate_solar_time(hour, longitude, standard_meridian, equation_of_time):
    return hour + (4 * (standard_meridian - longitude) + equation_of_time) / 60

def calculate_hour_angle(solar_time):
    return 15 * (solar_time - 12)

def calculate_solar_zenith(latitude, day_of_year, hour, longitude, standard_meridian):
    solar_declination = calculate_solar_declination(day_of_year)
    equation_of_time = calculate_equation_of_time(day_of_year)
    solar_time = calculate_solar_time(hour, longitude, standard_meridian, equation_of_time)
    hour_angle = calculate_hour_angle(solar_time)
    
    lat_rad = math.radians(latitude)
    decl_rad = math.radians(solar_declination)
    ha_rad = math.radians(hour_angle)
    
    cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) + 
                  math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad))
    solar_zenith = math.degrees(math.acos(max(min(cos_zenith, 1), -1)))  # Ensure the value is within valid range
    
    return solar_zenith, solar_declination

def calculate_solar_azimuth(latitude, hour, solar_declination, solar_zenith, longitude, standard_meridian, equation_of_time):
    time_offset = 4 * (longitude - standard_meridian) + equation_of_time
    tst = hour * 60 + time_offset
    ha = (tst / 4) - 180
    ha_rad = math.radians(ha)
    
    lat_rad = math.radians(latitude)
    decl_rad = math.radians(solar_declination)
    zen_rad = math.radians(solar_zenith)

    acos = (math.sin(decl_rad) * math.cos(lat_rad) - math.cos(decl_rad) * math.sin(lat_rad) * math.cos(ha_rad)) / math.cos(zen_rad)
    azimuth = math.degrees(math.acos(max(min(acos, 1), -1)))  # Ensure the value is within valid range

    if ha > 0:
        azimuth = 360 - azimuth
    
    return azimuth

def calculate_extraterrestrial_irradiance(day_of_year, solar_zenith):
    # Extraterrestrial irradiance on a horizontal surface
    G0 = SOLAR_CONSTANT
    cos_zenith = math.cos(math.radians(solar_zenith))
    G0h = G0 * cos_zenith if cos_zenith > 0 else 0
    return G0h

def calculate_diffuse_horizontal_irradiance(GHI, extraterrestrial_irradiance):
    kt = GHI / extraterrestrial_irradiance if extraterrestrial_irradiance > 0 else 0
    if kt > 0.82:
        DHI = 0.15 * GHI
    elif kt > 0.65:
        DHI = GHI * (1.41 - 1.56 * kt + 0.43 * kt ** 2)
    elif kt > 0.5:
        DHI = GHI * (1.39 - 1.48 * kt + 0.28 * kt ** 2)
    else:
        DHI = GHI * (1 - 0.9 * kt)
    return DHI

def calculate_POA_irradiance(direct_normal_irradiance, diffuse_horizontal_irradiance, tilt_angle, azimuth_angle, solar_zenith, solar_azimuth, albedo):
    tilt_rad = math.radians(tilt_angle)
    azimuth_rad = math.radians(azimuth_angle)
    zenith_rad = math.radians(solar_zenith)
    solar_azimuth_rad = math.radians(solar_azimuth)
    
    cos_incidence_angle = (math.sin(zenith_rad) * math.sin(tilt_rad) * math.cos(solar_azimuth_rad - azimuth_rad) + 
                           math.cos(zenith_rad) * math.cos(tilt_rad))
    
    beam_radiation = direct_normal_irradiance * max(cos_incidence_angle, 0)
    
    sky_diffuse_radiation = diffuse_horizontal_irradiance * (1 + math.cos(tilt_rad)) / 2
    
    ground_reflected_radiation = albedo * (direct_normal_irradiance * math.cos(zenith_rad) + diffuse_horizontal_irradiance) * (1 - math.cos(tilt_rad)) / 2
    
    POA_irradiance = beam_radiation + sky_diffuse_radiation + ground_reflected_radiation
    return round(POA_irradiance, 2)

def estimate_cell_temperature(POA_irradiance, ambient_temp, wind_speed, NOCT=45):
    # Simplified cell temperature estimation
    T_cell = ambient_temp + (NOCT - 20) * (POA_irradiance / 800) * (1 - 0.05 * wind_speed)
    return round(T_cell, 2)

def calculate_power_output(pv_total_kwp, poa_irradiance, cell_temp, inverter_efficiency, temp_coefficient):
    power_output_dc = pv_total_kwp * poa_irradiance * (1 + temp_coefficient * (cell_temp - 25))
    power_output_ac = power_output_dc * inverter_efficiency
    return round(power_output_ac, 2),round(power_output_dc, 2)

def calculate_diffuse_irradiance(GHI, beam_irradiance, solar_zenith):
    # Calculate Direct Normal Irradiance (DNI) and then Diffuse Horizontal Irradiance (DHI).

    theta_z_rad = math.radians(solar_zenith)
    if math.cos(theta_z_rad) > 0:
        DNI = beam_irradiance / math.cos(theta_z_rad)
    else:
        DNI = 0  # Handle case when sun is not up
    DHI = GHI - DNI * math.cos(theta_z_rad)    
    return round(DHI, 2)