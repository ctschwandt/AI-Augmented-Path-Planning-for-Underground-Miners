# sensor.py

import numpy as np
from src.utils import *
from src.constants import *

# Computing the transmission energy based on the distance and message size
def transmission_energy(k_bits, distance):
    if distance > MAX_COMM_RANGE :
        return 0.0

    if distance <= THRESHOLD_DIST:
        return k_bits * (ALPHA_ELEC + ALPHA_SHORT * distance **2)
    else:
        return k_bits * (ALPHA_ELEC + ALPHA_LONG * distance **4)

# Compute energy to recieve a message
def reception_energy(k_bits):
    return k_bits * ALPHA_ELEC

# Energy Loss per sensor

def compute_sensor_energy_loss (sensor_pos, miners, base_stations):

    total_energy = 0.0

    # Broadcast to general area
    total_energy += transmission_energy(K_BROADCAST, MAX_COMM_RANGE)

    # Communicate with each miner in range
    for miner_pos in miners:
        dist =  euclidean_distance(sensor_pos, miner_pos)
        
        if dist <= MAX_COMM_RANGE:
            total_energy += transmission_energy(K_TO_MINER, dist)
            total_energy += reception_energy(K_TO_MINER)

    if base_stations:
            distances = [euclidean_distance(sensor_pos, bs_pos) for bs_pos in base_stations]
            nearest_dist =  min(distances)
            if nearest_dist <=  MAX_COMM_RANGE:
                total_energy +=transmission_energy(K_TO_MINER, nearest_dist)
                total_energy += reception_energy(K_TO_BASE)
        
    return total_energy

#=======================
# Battery Update Function
#=======================

# Update the battery levels of all sensors after one timestep.
def update_single_sensor_battery(sensor_batteries, sensor_pos, miner, base_stations):
    if sensor_pos not in sensor_batteries:
        return sensor_batteries

    energy_used = compute_sensor_energy_loss(sensor_pos, [miner], base_stations)
    battery_loss_percent = (energy_used / BATTERY_CAPACITY_JOULES) * 100

    '''
    print(f"[Closest] Sensor at {sensor_pos}: energy used = {energy_used:.6f} J, "
          f"battery loss = {battery_loss_percent:.4f}%, before = {sensor_batteries[sensor_pos]:.2f}")
    '''
    
    sensor_batteries[sensor_pos] = max(0.0, sensor_batteries[sensor_pos] - battery_loss_percent)
    return sensor_batteries
