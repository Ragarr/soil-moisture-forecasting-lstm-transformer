# Soil Moisture and Meteorological Data Dataset Documentation

## General Description
This dataset combines soil moisture sensor data with meteorological data for humidity prediction analysis.

## Dataset Structure
- **Total number of records**: 59,329
- **Number of columns**: 12
- **Time period**: 2024-02-15 15:00:00 to 2025-03-09 19:00:00
- **Unique devices**: 91

## Attribute Description

### Identification and Time
- **device**: Unique sensor device identifier (format: 18-digit number)
- **ts**: Measurement timestamp (hourly frequency)

### Soil Moisture Sensors
- **sensor1**: Reading from the first soil moisture sensor
- **sensor2**: Reading from the second soil moisture sensor
- **var_s1**: Variance or variability of sensor 1
- **var_s2**: Variance or variability of sensor 2
- **voltaje**: Voltage level of the sensor device

### Meteorological Variables
- **precipitacion**: Precipitation in liters per square meter (l/m²)
- **humedad_ambiente**: Relative ambient humidity in percentage (%)
- **temperatura**: Temperature in degrees Celsius (°C)
- **radiacion_solar**: Solar radiation measurement
- **viento**: Wind speed

## Data Sources
1. **Humidity sensors**: Data collected from IoT devices installed in the field
2. **Meteorological data**: Weather station #59

## Data Processing
- Sensor data was filtered to remove missing values and invalid records
- Meteorological data was reshaped from daily to hourly structure
- Sensor data was downsampled to synchronize with hourly meteorological frequency
- Datasets were combined using timestamp-based merge

## Data Quality
- **Missing values**: Records with critical missing data were removed
- **Validation**: Only meteorological measurements marked as valid ('V') were included
- **Filtering**: Devices with invalid IDs were excluded from analysis

## Usage Notes
- Timestamps are in local time without timezone information
- Sensor measurements represent hourly averages when multiple readings per hour existed
- Missing values in meteorological variables may indicate periods without station coverage

---
*Documentation automatically generated on 2025-07-03 10:10:00*
