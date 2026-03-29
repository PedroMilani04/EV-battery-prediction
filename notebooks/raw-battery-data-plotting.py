import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/processed/B0005.csv')

# ─── 1. ESTATÍSTICAS BÁSICAS ───────────────────────────────────────────────
print("=== SHAPE ===")
print(df.shape)

print("\n=== TIPOS ===")
print(df.dtypes)

print("\n=== ESTATÍSTICAS GERAIS ===")
print(df.describe())

print("\n=== CONTAGEM POR TIPO ===")
print(df['type'].value_counts())

print("\n=== NULOS ===")
print(df.isnull().sum())

# ─── 2. DIFERENÇAS CHARGE vs DISCHARGE ────────────────────────────────────
charge    = df[df['type'] == 'charge']
discharge = df[df['type'] == 'discharge']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Charge vs Discharge', fontsize=14)

# Tensão
axes[0,0].hist(charge['Voltage_measured'],    bins=50, alpha=0.6, label='charge')
axes[0,0].hist(discharge['Voltage_measured'], bins=50, alpha=0.6, label='discharge')
axes[0,0].set_title('Voltage_measured')
axes[0,0].set_xlabel('Tensão (V)')
axes[0,0].set_ylabel('Quantidade de pontos')
axes[0,0].legend()

# Corrente
axes[0,1].hist(charge['Current_measured'],    bins=50, alpha=0.6, label='charge')
axes[0,1].hist(discharge['Current_measured'], bins=50, alpha=0.6, label='discharge')
axes[0,1].set_title('Current_measured')
axes[0,1].set_xlabel('Corrente (A)')
axes[0,1].set_ylabel('Quantidade de pontos')
axes[0,1].legend()

# Temperatura
axes[1,0].hist(charge['Temperature_measured'],    bins=50, alpha=0.6, label='charge')
axes[1,0].hist(discharge['Temperature_measured'], bins=50, alpha=0.6, label='discharge')
axes[1,0].set_title('Temperature_measured')
axes[1,0].set_xlabel('Temperatura (°C)')
axes[1,0].set_ylabel('Quantidade de pontos')
axes[1,0].legend()

# Duração
charge_duration    = charge.groupby('cycle')['Time'].max()
discharge_duration = discharge.groupby('cycle')['Time'].max()
axes[1,1].plot(charge_duration.index,    charge_duration.values,    label='charge')
axes[1,1].plot(discharge_duration.index, discharge_duration.values, label='discharge')
axes[1,1].set_title('Duração dos ciclos (Time máx)')
axes[1,1].set_xlabel('Número do ciclo')
axes[1,1].set_ylabel('Duração (segundos)')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('./data/processed/charge_vs_discharge.png', dpi=150)
plt.show()

# ─── 3. CAPACIDADE AO LONGO DOS CICLOS ────────────────────────────────────
# Capacity é escalar por ciclo, pega um valor por ciclo
capacity_per_cycle = (
    discharge.groupby('cycle')['Capacity']
    .first()
    .reset_index()
)

print("\n=== CAPACIDADE ===")
print(capacity_per_cycle.describe())

plt.figure(figsize=(10, 4))
plt.plot(capacity_per_cycle['cycle'], capacity_per_cycle['Capacity'], marker='o', markersize=2)
plt.axhline(y=1.4, color='red', linestyle='--', label='Fim de vida (1.4 Ah)')
plt.title('Degradação da Capacidade — B0005')
plt.xlabel('Ciclo')
plt.ylabel('Capacidade (Ah)')
plt.legend()
plt.tight_layout()
plt.savefig('./data/processed/capacity_degradation.png', dpi=150)
plt.show()