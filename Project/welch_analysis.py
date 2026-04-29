import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Voeg de map met de simulatiecode toe aan het pad
sys.path.append(os.path.join(os.getcwd(), "smaproject2026", "python-code"))

from simulation import Simulation

def calculate_welch_moving_avg(data, window_size):
    T = len(data)
    moving_avgs = np.zeros(T)
    for t in range(T):
        if t < window_size:
            end_idx = 2 * t + 1
            moving_avgs[t] = np.mean(data[:min(end_idx, T)])
        else:
            start_idx = t - window_size
            end_idx = t + window_size + 1
            moving_avgs[t] = np.mean(data[start_idx:min(end_idx, T)])
    return moving_avgs

def run_welch_analysis(R=100, W=200, window_size=15):
    print(f"Start Welch Analyse: R={R} replicaties, W={W} weken...")
    input_file = "smaproject2026/input-S1-14.txt"
    
    all_reps_data = np.zeros((R, W))
    sim = Simulation(input_file, W, R, 1)
    sim.setWeekSchedule()
    
    for r in range(R):
        if (r + 1) % 20 == 0:
            print(f"  Replicatie {r+1}/{R}...")
        sim.resetSystem()
        random.seed(r)
        sim.runOneSimulation()
        all_reps_data[r] = sim.getWeeklyObjectiveValues()
    
    # Gemiddelde over replicaties
    mean_per_week = np.mean(all_reps_data, axis=0)
    
    # Welch Moving Average
    moving_avg = calculate_welch_moving_avg(mean_per_week, window_size)
    
    # Plotten zoals vorig jaar
    plt.figure(figsize=(12, 6))
    # plt.plot(range(W), mean_per_week, alpha=0.3, label='Gemiddelde OV per week') # Raw data removed as requested
    plt.plot(range(W), moving_avg, color='red', linewidth=2, label=f'Welch Moving Average (w={window_size})')
    
    # Horizontale lijn voor steady state (geschat op laatste 50 weken)
    steady_state_avg = np.mean(moving_avg[-50:])
    plt.axhline(y=steady_state_avg, color='green', linestyle='--', label=f'Steady State Avg (~{steady_state_avg:.4f})')
    
    # Hulplijn voor week 10 en week 20
    plt.axvline(x=10, color='blue', linestyle=':', label='Week 10')
    plt.axvline(x=20, color='orange', linestyle=':', label='Week 20')
    
    plt.title("Welch Analyse voor Warm-up Periode (Objective Function)")
    plt.xlabel("Weken")
    plt.ylabel("Objective Function Waarde")
    plt.legend()
    plt.grid(True)
    plt.savefig("welch_graph.png")
    
    # Detectie van warm-up (zoals vorig jaar: wanneer komt het binnen een marge van de steady state)
    warmup_week = 0
    marge = 0.01 * steady_state_avg # 1% marge
    for w in range(W):
        if abs(moving_avg[w] - steady_state_avg) < marge:
            warmup_week = w
            break
            
    print(f"\nWelch analyse voltooid.")
    print(f"Berekende Steady State waarde: {steady_state_avg:.4f}")
    print(f"Aanbevolen warm-up periode (1% marge): week {warmup_week}")
    print("Grafiek opgeslagen als 'welch_graph.png'.")

if __name__ == "__main__":
    run_welch_analysis()
