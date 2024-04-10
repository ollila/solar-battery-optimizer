#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 07:54:17 2024

@author: ollila
"""

import csv
from datetime import datetime
from collections import defaultdict

# Function to convert the consumption value from string to float
def parse_consumption(consumption_str):
    return float(consumption_str.replace(',', '.'))

# Your input and output file paths
input_file_path = 'fingrid_input.csv'
output_file_path = 'fingrid_output.csv'

# Dictionary to hold aggregated consumption values, keyed by timestamp
aggregated_data = defaultdict(float)

# Read the CSV and aggregate consumption values by hour
with open(input_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader)  # Skip the header row
    for row in reader:
        timestamp_str = row[5]
        consumption_str = row[6]
        # Parse the timestamp and consumption value
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        consumption = parse_consumption(consumption_str)
        # Aggregate consumption by year, month, day, and hour
        key = timestamp.strftime("%Y-%m-%dT%H:00:00Z")
        aggregated_data[key] += consumption

# Write the aggregated data to a new CSV file
with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    # Optionally write a header to the output file
    writer.writerow(['Alkuaika', 'Määrä'])
    for key, total_consumption in aggregated_data.items():
        writer.writerow([key, f'{total_consumption:.6f}'.replace('.', ',')])

print("Aggregation complete. Output saved to:", output_file_path)
