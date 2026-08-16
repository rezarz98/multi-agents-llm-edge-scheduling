import pandas as pd
import random
import math

def read_benchmark(file_path):
    tasks = []
    with open(file_path, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 5 and parts[0].isdigit():
                tasks.append({
                    'taskid': int(parts[0]),
                    'Processing Time': int(parts[1]),
                    'Release Time': int(parts[2]),
                    'Due Date': int(parts[3]),
                    'Benefit': int(parts[4])
                })
    df = pd.DataFrame(tasks)
    return df

def add_additional_columns(df):
    df['Vehicle ID'] = [random.randint(1, 8) for _ in range(len(df))]

    # 3 access points: 1–3 → AP1, 4–6 → AP2, 7–8 → AP3
    def ap(v):
        if v <= 3:
            return 1
        elif v <= 6:
            return 2
        else:
            return 3
    df['Access Point'] = df['Vehicle ID'].apply(ap)

    # 2 brokers: assign randomly (or you can tie to AP)
    df['Broker ID'] = [random.randint(1, 2) for _ in range(len(df))]

    vehicle_distances = {1: 50, 2: 150, 3: 100, 4: 50, 5: 150, 6: 100, 7: 50, 8: 100}
    df['Access Point Communication Time'] = df.apply(
        lambda row: calculate_communication_time(
            row['Processing Time'], vehicle_distances[row['Vehicle ID']]), axis=1
    )

    # Adjust 40% of Access Point Communication Time
    adjust_communication_time(df, 'Access Point Communication Time')

    # Broker communication time based on Access Point
    broker_distances = {
        1: {1: 200, 2: 260},
        2: {1: 300, 2: 350},
        3: {1: 400, 2: 450},
    }
    df['Broker Communication Time'] = df.apply(
        lambda r: calculate_communication_time(
            r['Processing Time'],
            broker_distances[r['Access Point']][r['Broker ID']]
        ),
        axis=1
    )

    # Adjust 40% of Broker Communication Time
    adjust_communication_time(df, 'Broker Communication Time')

    # Base Station communication times
    base_station_distances = {
        1: 150,  # Distance from broker to base station 1
        2: 250,  # Distance from broker to base station 2
        3: 200,  # Distance from broker to base station 3
        4: 100,   # Distance from broker to base station 4
        5: 200,   # Distance from broker to base station 5
        6: 250   # Distance from broker to base station 6
    }

    for bs_id in range(1, 7):
        column_name = f'Base Station {bs_id} Communication Time'
        df[column_name] = df.apply(
            lambda row: calculate_communication_time(row['Processing Time'], base_station_distances[bs_id]), axis=1
        )
        adjust_communication_time(df, column_name)

    return df

def calculate_communication_time(processing_time, distance):
    processing_cost = math.ceil(processing_time / 3) * 0.008
    distance_cost = math.ceil(distance / 50) * 0.009
    total_cost = processing_cost + distance_cost
    return round(total_cost, 1)  # Round to one decimal place

def adjust_communication_time(df, column_name):
    # Adjust 40% of the tasks
    mask = random.sample(range(len(df)), int(0.4 * len(df)))
    for i in mask:
        df.at[i, column_name] = round(random.uniform(0.1, 0.5), 1)

def save_dataframe_to_file(df, file_path):
    df.to_csv(file_path, index=False)

def assign_priority_classes(df, seed: int = 42):
    rng = random.Random(seed)
    df['PriorityClass'] = 'NTC'
    high_count = int(0.6 * len(df))
    high_indices = rng.sample(list(df.index), high_count)
    df.loc[high_indices, 'PriorityClass'] = 'TC'
    return df

def read_csv_to_dataframe(file_path, seed: int = 42):
    df = pd.read_csv(file_path)
    # now every run uses the same TC/NTC split
    df = assign_priority_classes(df, seed)
    return df
