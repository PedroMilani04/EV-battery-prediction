from pandas import read_csv
from numpy import random
from pathlib import Path



root_path = Path('./data/raw/DEVST')

for path in root_path.iterdir():
    if path.is_dir():
        files = [f for f in path.iterdir() if f.is_file()]

        if files:
            chosen_file = random.choice(files)
            #print(f"File picked: {chosen_file}")

            df = read_csv(chosen_file, sep=';')
            #EVs 1, 4, 7, 10, 13
            #Normal driving style
            vehicles = ['EV1', 'EV4', 'EV7', 'EV10', 'EV13']

            df = df[(df['vehID'].isin(vehicles))]
            print(f"file is: {chosen_file}")
            print(df.head(10))

            df.to_csv(f"./data/processed/EV/{chosen_file.name}")