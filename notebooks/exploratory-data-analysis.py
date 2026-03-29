import scipy.io as sio
import pandas as pd
import numpy as np

mat_data = sio.loadmat(
    './data/raw/5.+Battery+Data+Set/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat',
    simplify_cells=True  # <-- isso achata structs aninhados automaticamente
)

battery = mat_data['B0005']

print(type(battery))          # dict ou ndarray?
print(battery.keys())         # quais campos existem?

# Se tiver 'cycle':
cycles = battery['cycle']
print(type(cycles[0]))        # inspeciona o primeiro ciclo
print(cycles[0].keys())       # campos dentro do ciclo
print(cycles[0]['data'].keys()) # campos dentro de data


cycles = battery['cycle']  # array de ciclos

rows = []

for i, cycle in enumerate(cycles):
    cycle_type = cycle['type']
    ambient_temp = cycle['ambient_temperature']
    data = cycle['data']

    # Pula ciclos de impedance (estrutura muito diferente)
    if cycle_type == 'impedance':
        continue

    n = len(data['Time'])

    for j in range(n):
        row = {
            'cycle':                i,
            'type':                 cycle_type,
            'ambient_temperature':  ambient_temp,
            'Time':                 data['Time'][j],
            'Voltage_measured':     data['Voltage_measured'][j],
            'Current_measured':     data['Current_measured'][j],
            'Temperature_measured': data['Temperature_measured'][j],
            # Campos exclusivos de charge
            'Current_charge':       data['Current_charge'][j] if cycle_type == 'charge' else None,
            'Voltage_charge':       data['Voltage_charge'][j] if cycle_type == 'charge' else None,
            # Campos exclusivos de discharge
            'Current_load':         data['Current_load'][j] if cycle_type == 'discharge' else None,
            'Voltage_load':         data['Voltage_load'][j] if cycle_type == 'discharge' else None,
            'Capacity':             data['Capacity'] if cycle_type == 'discharge' else None,
        }
        rows.append(row)

df = pd.DataFrame(rows)
print(df.head(20))
print(df.shape)

df.to_csv('./data/processed/B0005.csv', index=False)
