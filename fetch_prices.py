import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
import configparser

# create .secrets file and add following lines to it
# [entsoe]
# API_KEY = your_api_key

# Read API key from .secrets file
config = configparser.ConfigParser()
config.read('.secrets')
API_KEY = config['entsoe']['api_key']
ENTSOE_API_URL = 'https://web-api.tp.entsoe.eu/api'
START_DATE= 20240101

# Set periodStart to the beginning of the year
start_date_obj = datetime.strptime(str(START_DATE), '%Y%m%d')
previous_day = start_date_obj - timedelta(days=1)
period_start = previous_day.strftime('%Y%m%d') + '2200'
#period_start = start_date_obj.strftime('%Y%m%d') + '2200'

# Get the current date and end of tomorrow's date
today = datetime.now(timezone.utc)
tomorrow = today + timedelta(days=1)
end_of_tomorrow = tomorrow + timedelta(days=1)
period_end = end_of_tomorrow.strftime('%Y%m%d0000')
#period_end = 202501010000

# Define the parameters for the API request
params = {
    'securityToken': API_KEY,
    'documentType': 'A44',  # Day-ahead prices
    'in_Domain': '10YFI-1--------U',  # BZN|FI
    'out_Domain': '10YFI-1--------U',
    'periodStart': period_start,  # Start date
    'periodEnd': period_end  # End date 
}

# Make the API request
response = requests.get(ENTSOE_API_URL, params=params)

# Check for a successful response
if response.status_code == 200:
    # Parse the XML response
    xml_data = response.content

    # Register namespaces
    namespaces = {
        'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0'
    }

    # Parse the XML
    root = ET.fromstring(xml_data)
    
    # Extract relevant data
    data = []
    for time_series in root.findall('.//ns:TimeSeries', namespaces):
        for period in time_series.findall('.//ns:Period', namespaces):
            start_time = period.find('.//ns:timeInterval/ns:start', namespaces).text
            for point in period.findall('.//ns:Point', namespaces):
                position = int(point.find('.//ns:position', namespaces).text)
                price_amount = float(point.find('.//ns:price.amount', namespaces).text)
                
                # Calculate the exact start time for this point
                current_start_time = datetime.fromisoformat(start_time) + timedelta(hours=position)
                current_end_time = current_start_time + timedelta(hours=1)
                
                data.append({
                    'start_time': current_start_time,
                    'end_time': current_end_time,
                    'price_amount': price_amount
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert and format the times
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['price_amount'] = pd.to_numeric(df['price_amount'])

    # Create a new DataFrame with the desired columns
    formatted_data = pd.DataFrame({
        'MTU (CET/CEST)': df['start_time'].dt.strftime('%d.%m.%Y %H:%M') + ' - ' + df['end_time'].dt.strftime('%d.%m.%Y %H:%M'),
        'Day-ahead Price [EUR/MWh]': df['price_amount'],
        'Currency': 'EUR'
    })

    # Ensure the 'BZN|FI' column is included in the CSV
    formatted_data = formatted_data[['MTU (CET/CEST)', 'Day-ahead Price [EUR/MWh]', 'Currency']]

    # Save the DataFrame to a CSV file
    year = today.year
    file_name = f'price{year}.csv'
    formatted_data.to_csv(file_name, index=False, header=True, quoting=1)  # quoting=1 to quote non-numeric values

    print(f"Data successfully saved to {file_name}")

else:
    print(f"Failed to fetch data: {response.status_code}")
    print(response.text)
