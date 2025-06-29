import pandas as pd
import numpy as np
import os

hum_sensors_path = "data/humidity/raw.csv"
meteo_path = "data/meteo/"
output_path = "data/merged.csv"


# load humidity df the raw datasets
hum_sensors = pd.read_csv(hum_sensors_path, sep=';', decimal='.', dtype=str)
hum_sensors = hum_sensors.dropna() # las filas con missing values estan tan mal que no se pueden recuperar

# hay algunas filas (la ultima) que tienen los datos tambien mal, 
# se puede identificar porque el device no cumple el regex: 898822806661323382
hum_sensors = hum_sensors[hum_sensors['device'].str.match(r'\d{18}')]


hum_sensors['device'] = hum_sensors['device'].astype(str)
hum_sensors['sensor1'] = hum_sensors['sensor1'].astype(float)
hum_sensors['sensor2'] = hum_sensors['sensor2'].astype(float)
hum_sensors['var_s1'] = hum_sensors['var_s1'].astype(float)
hum_sensors['var_s2'] = hum_sensors['var_s2'].astype(float)
hum_sensors['voltaje'] = hum_sensors['voltaje'].astype(int)
hum_sensors['ts'] = pd.to_datetime(hum_sensors['ts'], format='mixed').dt.tz_localize(None)
hum_sensors = hum_sensors.sort_values('ts')

# load all meteo datasets
meteo_files = os.listdir(meteo_path)
meteo_files = [f for f in meteo_files if f.endswith('.csv')] # solo por si acaso

meteo_df = pd.concat([pd.read_csv(meteo_path+f, sep=';') for f in meteo_files])

# seleccionar las filas que nos interesan
meteo_df = meteo_df[meteo_df['ESTACION']==59]
# MAGNITUD 89 = PRECIPIACION (l/m2 ), 86 = HUMEDAD RELATIVA (%), 83 = TEMPERATURA (ºC)
prec_df = meteo_df[meteo_df['MAGNITUD']==89]
hum_df = meteo_df[meteo_df['MAGNITUD']==86]
temp_df = meteo_df[meteo_df['MAGNITUD']==83]
sun_df = meteo_df[meteo_df['MAGNITUD']==88]
wind_df = meteo_df[meteo_df['MAGNITUD']==81]

meteo_df = pd.DataFrame()

# los datos estan estructurados de la siguiente manera:
# cada fila es un dia, y hay columnas H01, H02, H03, ..., H24 que representan la hora del dia
# tambien hay V01, V02, V03, ..., V24 que representan la validez de cada medicion
# la fecha se encuentra en las columnas ANO MES DIA
# los demas campos no son relevantes para nosotros

# vamos a combinar todos los datos en un solo dataframe
# de estructura: ts, precipitacion, humedad_ambiente, temperatura

# primero vamos a hacer un reshape de los dataframes para que tengan la estructura deseada
def reshape_df(df, magnitud):
    new_df = pd.DataFrame()
    for i, row in df.iterrows():
        for h in range(1, 25):
            ts = pd.Timestamp(row['ANO'], row['MES'], row['DIA'], h-1)
            value = row[f'H{h:02d}']
            valid = row[f'V{h:02d}']
            if valid == 'V':
                new_df=pd.concat([new_df, pd.DataFrame({'ts': [ts], magnitud: [value]})])
    return new_df


prec_df = reshape_df(prec_df, 'precipitacion')
hum_df = reshape_df(hum_df, 'humedad_ambiente')
temp_df = reshape_df(temp_df, 'temperatura')
sun_df = reshape_df(sun_df, 'radiacion_solar')
wind_df = reshape_df(wind_df, 'viento')


# ahora vamos a combinar los dataframes
meteo_df = pd.merge(prec_df, hum_df, on='ts')
meteo_df = pd.merge(meteo_df, temp_df, on='ts')
meteo_df = pd.merge(meteo_df, sun_df, on='ts')
meteo_df = pd.merge(meteo_df, wind_df, on='ts')


# ahora vamos a combinar los dataframes de humedad y meteo
# primero hacer downsample de hum_sensors para que tenga la misma frecuencia que meteo (1h)
hum_sensors['ts'] = hum_sensors['ts'].dt.floor('h').dt.tz_localize(None)
hum_sensors = hum_sensors.groupby(['device', 'ts']).mean().reset_index()

# ahora vamos a hacer un merge de los dos dataframes
merged_df = pd.merge(hum_sensors, meteo_df, on='ts', how='left')

# guardar el dataframe en un archivo csv
print(f"Guardando el dataframe en {output_path}")
merged_df.to_csv(output_path, index=False)
print("Sample del dataframe:")
print(merged_df.head())

