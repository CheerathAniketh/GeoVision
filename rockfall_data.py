import pandas as pd
import numpy as np

# Small original dataset
data = {
    "slope_angle": [45,20,55,30,15,40,60,25,50,35,10,65,28,48,38,52,18,42,58,33],
    "rainfall_mm": [5,5,30,10,2,0,35,8,20,12,0,40,9,18,14,22,3,6,28,11],
    "vibration": [0.03,0.01,0.14,0.03,0.005,0.01,0.16,0.02,0.1,0.05,0.0,0.18,0.025,0.09,0.06,0.11,0.007,0.035,0.13,0.045],
    "pore_pressure": [100,60,170,90,50,80,180,70,150,120,40,200,85,140,125,155,55,105,165,115],
    "temperature_c": [29,27,34,29,26,28,35,28,32,30,25,36,28,31,30,33,26,29,34,30]
}

df_small = pd.DataFrame(data)

# Generate 1000 synthetic rows
n_synthetic = 1000
np.random.seed(42)
synthetic_rows = []

for _ in range(n_synthetic):
    row = {}
    row["slope_angle"] = np.clip(np.random.choice(df_small["slope_angle"]) + np.random.randint(-10,11), 10, 70)
    row["rainfall_mm"] = np.clip(int(row["slope_angle"] * 0.5 + np.random.randint(-5,6)), 0, 50)
    row["vibration"] = np.clip(np.random.choice(df_small["vibration"]) + np.random.uniform(-0.02,0.02), 0, 0.2)
    row["pore_pressure"] = np.clip(int(row["rainfall_mm"]*5 + np.random.randint(-10,11)), 40, 200)
    row["temperature_c"] = np.clip(np.random.choice(df_small["temperature_c"]) + np.random.randint(-3,4), 25, 36)
    synthetic_rows.append(row)

df_synthetic = pd.DataFrame(synthetic_rows)

# Save to CSV
df_synthetic.to_csv("demo_data.csv", index=False)
print("demo_data.csv created with", len(df_synthetic), "rows")
