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
    df['Vehicle ID'] = [random.randint(1, 4) for _ in range(len(df))]

    df['Access Point'] = df['Vehicle ID'].apply(lambda x: 1 if x in [1, 2, 3] else 2)

    vehicle_distances = {1: 50, 2: 150, 3: 100, 4: 50}
    df['Access Point Communication Time'] = df.apply(
        lambda row: calculate_communication_time(
            row['Processing Time'], vehicle_distances[row['Vehicle ID']]), axis=1
    )

    # Adjust 40% of Access Point Communication Time
    adjust_communication_time(df, 'Access Point Communication Time')

    # Broker communication time based on Access Point
    access_point_distances = {1: 250, 2: 300}
    df['Broker Communication Time'] = df.apply(
        lambda row: calculate_communication_time(
            row['Processing Time'], access_point_distances[row['Access Point']]), axis=1
    )

    # Adjust 40% of Broker Communication Time
    adjust_communication_time(df, 'Broker Communication Time')

    # Base Station communication times
    base_station_distances = {
        1: 150,  # Distance from broker to base station 1
        2: 250,  # Distance from broker to base station 2
        3: 200,  # Distance from broker to base station 3
        4: 100   # Distance from broker to base station 4
    }

    for bs_id in range(1, 5):
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
