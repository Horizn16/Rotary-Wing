import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.optimize import root

# Import project modules
from utils.data_loader import load_config
from utils.atmosphere import get_isa_conditions
from performance_estimator.models import Rotor, calculate_fuselage_forces
from mission_planner.segments import calculate_power_curve, get_vehicle_performance_at_speed

def run_pilot_input_tests(config):
    """Task 3: Analyzes forces and moments vs. pilot inputs."""
    print("\n[INFO] Running Task 3: Pilot Input Tests...")
    flight_conditions = {'altitude_m': 2000, 'airspeed_kmh': 200}
    _, _, rho = get_isa_conditions(flight_conditions['altitude_m'])
    V_inf = flight_conditions['airspeed_kmh'] * 1000 / 3600
    
    main_rotor = Rotor(config['main_rotor'])
    base_inputs_rad = {
        'collective': np.deg2rad(8.0),
        'cyclic_lat': np.deg2rad(0.0),
        'cyclic_lon': np.deg2rad(-2.0),
    }

    collective_range_deg = np.linspace(2, 12, 15)
    results = {'Thrust': [], 'H_force': [], 'Y_force': [], 'Roll_moment': [], 'Pitch_moment': [], 'Torque': []}

    for coll_deg in collective_range_deg:
        current_inputs = base_inputs_rad.copy()
        current_inputs['collective'] = np.deg2rad(coll_deg)
        perf = main_rotor.calculate_forces_moments(V_inf, rho, current_inputs)
        for key in results:
            results[key].append(perf[key])

    output_dir = Path('../output/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Forces & Moments vs. Main Rotor Collective (200 km/h @ 2000m)', fontsize=16)
    axes = axes.flatten()
    labels = ['$F_Z$ (Thrust)', '$F_X$ (H-Force)', '$F_Y$ (Y-Force)', '$M_X$ (Roll)', '$M_Y$ (Pitch)', '$M_Z$ (Torque)']
    keys = list(results.keys())
    
    for i, ax in enumerate(axes):
        ax.plot(collective_range_deg, results[keys[i]], 'o-')
        ax.set_xlabel('Collective Pitch (deg)')
        ax.set_ylabel(labels[i])
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'pilot_input_tests.png')
    plt.show()

def run_trim_analysis(config):
    """Task 4 & 7: Finds and displays trim settings."""
    print("\n[INFO] Running Task 4 & 7: Trim Analysis...")
    altitude_m, speed_kmh = 2000, 200
    speed_mps = speed_kmh * 1000 / 3600
    
    pilot_guess = np.deg2rad([8.0, 0.0, -2.0, 5.0]) # Initial guess
    power, success, trimmed_inputs_rad = get_vehicle_performance_at_speed(speed_mps, altitude_m, config, pilot_guess)

    if success:
        trimmed_deg = np.rad2deg(trimmed_inputs_rad)
        print("\n--- Trim Settings Found ---")
        print(f"Flight: {speed_kmh} km/h at {altitude_m} m")
        print("-" * 30)
        print(f"  Main Rotor Collective (theta_0): {trimmed_deg[0]:.2f} deg")
        print(f"  Main Rotor Lateral Cyclic (theta_1c): {trimmed_deg[1]:.2f} deg")
        print(f"  Main Rotor Longitudinal Cyclic (theta_1s): {trimmed_deg[2]:.2f} deg")
        print(f"  Tail Rotor Collective (theta_0_t): {trimmed_deg[3]:.2f} deg")
        print(f"\n  Resultant Power Required: {power:.2f} kW")
        print("-" * 30)
    else:
        print("\n[ERROR] Trim solver failed to converge.")

def run_mission_performance_test(config):
    """Task 5 & 8: Calculates and plots mission performance."""
    print("\n[INFO] Running Task 5 & 8: Mission Performance Test...")
    altitude_m = 2000
    
    speeds, powers = calculate_power_curve(config, altitude_m)
    
    if not speeds:
        print("[ERROR] Could not calculate power curve. Trim failed at all speeds.")
        return
        
    power_available = config['engine']['max_power_kw']
    
    # Max speed is where power required intersects power available
    max_speed_idx = np.where(np.array(powers) > power_available)[0]
    max_speed = speeds[max_speed_idx[0]-1] if len(max_speed_idx) > 0 else speeds[-1]

    # Max endurance is at minimum power
    min_power_idx = np.argmin(powers)
    max_endurance_speed = speeds[min_power_idx]
    
    # Max range speed (simplified: find tangent from origin to power curve)
    # Equivalent to finding min of P/V
    power_per_speed = np.array(powers) / np.array(speeds)
    max_range_idx = np.argmin(power_per_speed)
    max_range_speed = speeds[max_range_idx]

    print("\n--- Mission Performance Results ---")
    print(f"Altitude: {altitude_m} m")
    print("-" * 40)
    print(f"  Max Speed (Power Limited): {max_speed:.2f} km/h")
    print(f"  Max Endurance Speed: {max_endurance_speed:.2f} km/h")
    print(f"  Max Range Speed: {max_range_speed:.2f} km/h")
    print("-" * 40)
    
    output_dir = Path('../output/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, powers, 'o-', label='Power Required')
    plt.axhline(y=power_available, color='r', linestyle='--', label='Power Available')
    plt.title(f'Helicopter Power Curve @ {altitude_m} m')
    plt.xlabel('Airspeed (km/h)')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(output_dir / 'power_curve.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="AE 667 Helicopter Analysis Tool")
    parser.add_argument('--task', type=str, required=True, choices=['pilot_test', 'trim', 'mission'], help="The analysis task to run.")
    parser.add_argument('--config', type=str, default='../configs/team_helicopter.json', help="Path to helicopter config JSON.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load configuration. Error: {e}")
        return

    # To ensure you can import from src, we add it to the path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    if args.task == 'pilot_test':
        run_pilot_input_tests(config)
    elif args.task == 'trim':
        run_trim_analysis(config)
    elif args.task == 'mission':
        run_mission_performance_test(config)

if __name__ == "__main__":
    main()
