import numpy as np
from performance_estimator.models import Rotor, calculate_fuselage_forces
from utils.atmosphere import get_isa_conditions
from scipy.optimize import root

def get_vehicle_performance_at_speed(speed_mps, altitude_m, config, pilot_inputs_guess):
    """
    Calculates the trimmed performance of the helicopter at a specific speed and altitude.
    Returns a dictionary with all performance metrics upon successful convergence.
    """
    def trim_residuals(pilot_inputs_vec, V, alt, cfg):
        pilot_inputs = {
            'collective': pilot_inputs_vec[0],
            'cyclic_lat': pilot_inputs_vec[1],
            'cyclic_lon': pilot_inputs_vec[2],
        }
        tr_collective_rad = pilot_inputs_vec[3]
        _, _, rho = get_isa_conditions(alt)
        main_rotor = Rotor(cfg['main_rotor'])
        tail_rotor = Rotor(cfg['tail_rotor'], is_main_rotor=False)
        
        fuselage_drag_est = 0.5 * rho * V**2 * cfg['vehicle']['fuselage_flat_plate_area_m2']
        weight_est = cfg['vehicle']['gross_weight_kg'] * 9.81
        alpha_tpp_rad = np.arctan2(fuselage_drag_est, weight_est)
        
        mr_perf = main_rotor.calculate_forces_moments(V, rho, pilot_inputs, alpha_tpp_rad)
        tr_perf = tail_rotor.calculate_forces_moments(0, rho, {'collective': tr_collective_rad, 'cyclic_lat': 0, 'cyclic_lon': 0})
        
        alpha_f_rad = alpha_tpp_rad - mr_perf['beta_1c'] - np.deg2rad(cfg['vehicle']['shaft_inclination_deg'])
        fuselage_drag, fuselage_lift = calculate_fuselage_forces(cfg, V, rho, alpha_f_rad)
        
        weight = cfg['vehicle']['gross_weight_kg'] * 9.81
        
        Fx = mr_perf['Thrust'] * np.sin(alpha_tpp_rad) - mr_perf['H_force'] - fuselage_drag - tr_perf['Thrust']
        Fz = mr_perf['Thrust'] * np.cos(alpha_tpp_rad) - weight + fuselage_lift
        My = mr_perf['Pitch_moment']
        Mz = mr_perf['Torque'] - tr_perf['Thrust'] * cfg['tail_rotor']['arm_m']
        
        return [Fx, Fz, My, Mz]

    sol = root(trim_residuals, pilot_inputs_guess, args=(speed_mps, altitude_m, config), method='hybr', tol=1e-3, options={'maxfev': 500})

    if sol.success:
        trimmed_inputs_rad = {'collective': sol.x[0], 'cyclic_lat': sol.x[1], 'cyclic_lon': sol.x[2]}
        tr_collective_rad = sol.x[3]
        _, _, rho = get_isa_conditions(altitude_m)
        main_rotor = Rotor(config['main_rotor'])
        tail_rotor = Rotor(config['tail_rotor'], is_main_rotor=False)
        
        fuselage_drag_est = 0.5 * rho * speed_mps**2 * config['vehicle']['fuselage_flat_plate_area_m2']
        weight_est = config['vehicle']['gross_weight_kg'] * 9.81
        alpha_tpp_rad = np.arctan2(fuselage_drag_est, weight_est)
        
        mr_perf = main_rotor.calculate_forces_moments(speed_mps, rho, trimmed_inputs_rad, alpha_tpp_rad)
        tr_perf = tail_rotor.calculate_forces_moments(0, rho, {'collective': tr_collective_rad, 'cyclic_lat': 0, 'cyclic_lon': 0})
        
        alpha_f_rad = alpha_tpp_rad - mr_perf['beta_1c'] - np.deg2rad(config['vehicle']['shaft_inclination_deg'])
        fuselage_drag, fuselage_lift = calculate_fuselage_forces(config, speed_mps, rho, alpha_f_rad)
        total_power_kw = mr_perf['Power_kw'] + tr_perf['Power_kw']
        
        full_results = {
            "trimmed_inputs_rad": sol.x, "power_kw": total_power_kw, "alpha_tpp_rad": alpha_tpp_rad,
            "main_rotor_perf": mr_perf, "tail_rotor_perf": tr_perf, "fuselage_drag": fuselage_drag,
            "fuselage_lift": fuselage_lift, "stall_detected": mr_perf['stall_detected']
        }
        return full_results, True
    else:
        return None, False

def calculate_power_curve(config, altitude_m):
    """
    Calculates the power required versus airspeed curve. It tries every speed
    and returns all the points where trim was successfully found.
    """
    all_speeds_kmh = np.linspace(20, 420, 30)
    results_dict = {}
    
    pilot_guess = np.deg2rad([8.0, 0.0, -2.0, 5.0])
    
    print("Calculating power curve... (This may take a moment)")
    for i, speed_kmh in enumerate(all_speeds_kmh):
        print(f"  Testing speed {i+1}/{len(all_speeds_kmh)}: {speed_kmh:.0f} km/h...", end="")
        speed_mps = speed_kmh * 1000 / 3600
        
        results, success = get_vehicle_performance_at_speed(speed_mps, altitude_m, config, pilot_guess)
        
        if success:
            print(" Trim SUCCESS")
            results_dict[speed_kmh] = results
            pilot_guess = results['trimmed_inputs_rad']
        else:
            print(" Trim FAILED")

    if not results_dict:
        return [], [], 0

    sorted_speeds = sorted(results_dict.keys())
    
    # First, gather all successful data.
    all_valid_speeds = []
    all_valid_powers = []
    
    for speed in sorted_speeds:
        all_valid_speeds.append(speed)
        all_valid_powers.append(results_dict[speed]['power_kw'])

    # Second, determine the stall speed from the successful data.
    stall_limited_speed_kmh = all_valid_speeds[-1] # Default to max speed
    for speed in sorted_speeds:
        if results_dict[speed]['stall_detected']:
            # Stall speed is the last valid speed *before* the stall happened.
            stall_idx = all_valid_speeds.index(speed)
            stall_limited_speed_kmh = all_valid_speeds[stall_idx - 1] if stall_idx > 0 else 0
            break
            
    return all_valid_speeds, all_valid_powers, stall_limited_speed_kmh

