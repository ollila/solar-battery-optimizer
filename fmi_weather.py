import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone  # Ensure 'timezone' is imported
import pytz
import time
import random

def fetch_location_data(location_id):
    location_name = "Unknown_Location"
    latitude = "Unknown"
    longitude = "Unknown"
    initial_url = f"https://opendata.fmi.fi/wfs?request=getFeature&storedquery_id=fmi::observations::radiation::timevaluepair&parameters=GLOB_1MIN&timestep=300&timezone=Europe/Helsinki&starttime=2024-05-28T22:00:00Z&endtime=2024-05-29T22:00:00Z&fmisid={location_id}"
    
    initial_response = requests.get(initial_url)
    if initial_response.status_code == 200:
        initial_root = ET.fromstring(initial_response.content)
        gml_name_element = initial_root.find('.//gml:name', {'gml': 'http://www.opengis.net/gml/3.2'})
        if gml_name_element is not None:
            # location_name = gml_name_element.text
            location_name = gml_name_element.text.replace(' ', '_')
        gml_pos_element = initial_root.find('.//gml:pos', {'gml': 'http://www.opengis.net/gml/3.2'})
        if gml_pos_element is not None:
            pos = gml_pos_element.text.split()
            latitude = pos[0]
            longitude = pos[1]
  
    return location_name, latitude, longitude


def fetch_radiation_data(start_date, days_to_count, location):
    def fetch_data_for_period(start_time_utc, end_time_utc):
        initial_url = f"https://opendata.fmi.fi/wfs?request=getFeature&storedquery_id=fmi::observations::radiation::timevaluepair&parameters=GLOB_1MIN&timestep=60&timezone=Europe/Helsinki&starttime={start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}&endtime={end_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}&fmisid={location}"
        response = requests.get(initial_url)

        wait_time = random.uniform(0, 1)
        time.sleep(wait_time)

        if response.status_code == 200:
            return ET.fromstring(response.content)
        return None

    helsinki_tz = pytz.timezone('Europe/Helsinki')
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    start_date_dt = helsinki_tz.localize(start_date_dt)
    
    all_data = None
    
    for day_offset in range(0, days_to_count, 7):
        start_time_utc = (start_date_dt + timedelta(days=day_offset)).astimezone(pytz.utc)
        end_time_utc = start_time_utc + timedelta(days=min(7, days_to_count-day_offset)) - timedelta(seconds=1)
        now_utc = datetime.now(timezone.utc)
        
        if end_time_utc > now_utc:
            end_time_utc = now_utc
        
        period_data = fetch_data_for_period(start_time_utc, end_time_utc)
        if period_data is None:
            return None
        
        if all_data is None:
            all_data = period_data
        else:
            for element in period_data.findall('.//wml2:point', {'wml2': 'http://www.opengis.net/waterml/2.0'}):
                all_data[0].append(element)
    
    return all_data


def fetch_weather_data(start_date, days_to_count, location):
    def fetch_data_for_period(start_time_utc, end_time_utc):
        initial_url = f"https://opendata.fmi.fi/wfs?request=getFeature&storedquery_id=fmi::observations::weather::timevaluepair&timestep=10&starttime={start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}&endtime={end_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}&fmisid={location}"
        response = requests.get(initial_url)

        wait_time = random.uniform(0, 1)
        time.sleep(wait_time)
        
        if response.status_code == 200:
            return ET.fromstring(response.content)
        return None

    helsinki_tz = pytz.timezone('Europe/Helsinki')
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    start_date_dt = helsinki_tz.localize(start_date_dt)
    
    all_data = None
    
    for day_offset in range(0, days_to_count, 7):
        start_time_utc = (start_date_dt + timedelta(days=day_offset)).astimezone(pytz.utc)
        end_time_utc = start_time_utc + timedelta(days=min(7, days_to_count-day_offset)) - timedelta(seconds=1)
        now_utc = datetime.now(timezone.utc)
        
        if end_time_utc > now_utc:
            end_time_utc = now_utc
        
        period_data = fetch_data_for_period(start_time_utc, end_time_utc)
        if period_data is None:
            return None
        
        if all_data is None:
            all_data = period_data
        else:
            for element in period_data.findall('.//wml2:point', {'wml2': 'http://www.opengis.net/waterml/2.0'}):
                all_data[0].append(element)
    
    return all_data