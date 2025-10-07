import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import project modules
from utils.data_loader import load_config
from utils.atmosphere import get_isa_conditions
from performance_estimator.models import Rotor
from mission_planner.segments import calculate_power_curve, get_vehicle_performance_at_speed

def run_pilot_input_tests(config):
    """Task 3: Analyzes forces and moments vs. pilot inputs."""
    print("\n[INFO] Running Task 3: Pilot Input Tests...")
    flight_conditions = {'altitude_m': 2000, 'airspeed_kmh': 200}
    _, _, rho = get_isa_conditions(flight_conditions['altitude_m'])
    V_inf = flight_conditions['airspeed_kmh'] * 1000 / 3600
    
    main_rotor = Rotor(config['main_rotor'])
    base_inputs_rad = {
        'collective': np.deg2rad(8.0), 'cyclic_lat': np.deg2rad(0.0), 'cyclic_lon': np.deg2rad(-2.0),
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
    labels = ['$F_Z$ (Thrust, N)', '$F_X$ (H-Force, N)', '$F_Y$ (Y-Force, N)', '$M_X$ (Roll, Nm)', '$M_Y$ (Pitch, Nm)', '$M_Z$ (Torque, Nm)']
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
    """Task 4 & 7: Finds and displays trim settings in a detailed table."""
    print("\n[INFO] Running Task 4 & 7: Trim Analysis...")
    altitude_m, speed_kmh = 2000, 200
    speed_mps = speed_kmh * 1000 / 3600
    
    pilot_guess = np.deg2rad([8.0, 0.0, -2.0, 5.0])
    results, success = get_vehicle_performance_at_speed(speed_mps, altitude_m, config, pilot_guess)

    if success:
        trimmed_deg = np.rad2deg(results['trimmed_inputs_rad'])
        theta_0, theta_1c, theta_1s, theta_0_t = trimmed_deg[0], trimmed_deg[1], trimmed_deg[2], trimmed_deg[3]
        alpha_tpp_deg = np.rad2deg(results['alpha_tpp_rad'])
        beta_0_deg = np.rad2deg(results['main_rotor_perf']['beta_0'])
        
        mr, tr, fd, fl = results['main_rotor_perf'], results['tail_rotor_perf'], results['fuselage_drag'], results['fuselage_lift']
        weight, alpha_tpp_rad = config['vehicle']['gross_weight_kg'] * 9.81, results['alpha_tpp_rad']

        Fx = mr['Thrust'] * np.sin(alpha_tpp_rad) - mr['H_force'] - fd - tr['Thrust']
        Fy = mr['Y_force']
        Fz = mr['Thrust'] * np.cos(alpha_tpp_rad) - weight + fl
        Mx, My = mr['Roll_moment'], mr['Pitch_moment']
        Mz = mr['Torque'] - tr['Thrust'] * config['tail_rotor']['arm_m']
        
        print("\n--- Trim Results for 200 km/h at 2000m ---")
        print("+" + "-"*25 + "+" + "-"*25 + "+")
        print(f"| {'Input':<23} | {'Resultant':<23} |")
        print("+" + "-"*25 + "+" + "-"*25 + "+")
        print(f"| theta_0      = {theta_0:8.2f} deg | Fx (Vehicle) = {Fx:10.2f} N |")
        print(f"| theta_1s     = {theta_1s:8.2f} deg | Fy (Vehicle) = {Fy:10.2f} N |")
        print(f"| theta_1c     = {theta_1c:8.2f} deg | Fz (Vehicle) = {Fz:10.2f} N |")
        print(f"| theta_0_t    = {theta_0_t:8.2f} deg | Mx (Vehicle) = {Mx:10.2f} Nm|")
        print(f"| alpha_tpp    = {alpha_tpp_deg:8.2f} deg | My (Vehicle) = {My:10.2f} Nm|")
        print(f"| beta_0       = {beta_0_deg:8.2f} deg | Mz (Vehicle) = {Mz:10.2f} Nm|")
        print("+" + "-"*25 + "+" + "-"*25 + "+")
    else:
        print("\n[ERROR] Trim solver failed to converge.")

def run_mission_performance_test(config):
    """Task 5 & 8: Calculates and reports final mission performance metrics."""
    print("\n[INFO] Running Task 5 & 8: Full Mission Performance Analysis...")
    altitude_m = 2000
    
    # This function now returns all successful trim points
    all_speeds, all_powers, stall_speed = calculate_power_curve(config, altitude_m)
    
    if not all_speeds:
        print("\n[ERROR] Could not calculate any valid trim points for the power curve.")
        print("This could be due to an unstable helicopter design or incorrect parameters.")
    else:
        # Filter the data to only include speeds up to the stall speed for performance calculation
        perf_speeds = [s for s in all_speeds if s <= stall_speed]
        perf_powers = all_powers[:len(perf_speeds)]

        if not perf_speeds:
            print("\n[ERROR] Helicopter stalls at the lowest tested speed. No valid performance envelope.")
        else:
            power_available = config['engine']['max_power_kw']
            
            power_limited_speed = np.interp(power_available, perf_powers, perf_speeds) if perf_powers and perf_powers[-1] > power_available else perf_speeds[-1]
            
            min_power_idx = np.argmin(perf_powers)
            min_power = perf_powers[min_power_idx]
            endurance_hours = config['vehicle']['fuel_capacity_kg'] / (min_power * config['engine']['sfc_kg_per_kwh'])
            
            power_per_speed = np.array(perf_powers) / np.array(perf_speeds)
            max_range_idx = np.argmin(power_per_speed)
            max_range_speed_kmh = perf_speeds[max_range_idx]
            power_at_max_range = perf_powers[max_range_idx]
            range_km = max_range_speed_kmh * (config['vehicle']['fuel_capacity_kg'] / (power_at_max_range * config['engine']['sfc_kg_per_kwh']))

            print("\n--- Mission Performance Findings at 2000 m AMSL ---")
            print("+" + "-"*50 + "+")
            print(f"| {'Metric':<30} | {'Value':>17} |")
            print("+" + "-"*50 + "+")
            print(f"| 4.1 Max Speed (Blade Stall)      | {stall_speed:12.2f} km/h |")
            print(f"| 4.2 Max Speed (Power Requirement)| {power_limited_speed:12.2f} km/h |")
            print(f"| 4.3 Maximum Range                | {range_km:12.2f} km   |")
            print(f"| 4.4 Maximum Endurance            | {endurance_hours:12.2f} hours|")
            print("+" + "-"*50 + "+")
    
    # ALWAYS plot the graph with whatever data was successfully collected
    output_dir = Path('../output/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    if all_speeds:
        plt.plot(all_speeds, all_powers, 'o-', label='Power Required (Trimmed Points)')
        plt.axvline(x=stall_speed, color='purple', linestyle=':', label=f'Stall Speed Limit ({stall_speed:.0f} km/h)')
    else:
        plt.title(f'Helicopter Power Curve @ {altitude_m} m - NO VALID POINTS')

    plt.axhline(y=config['engine']['max_power_kw'], color='r', linestyle='--', label='Power Available')
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

    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    if args.task == 'pilot_test':
        run_pilot_input_tests(config)
    elif args.task == 'trim':
        run_trim_analysis(config)
    elif args.task == 'mission':
        run_mission_performance_test(config)

if __name__ == "__main__":
    main()

