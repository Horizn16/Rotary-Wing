import numpy as np

from performance_estimator.models import Rotor, calculate_fuselage_forces
from utils.atmosphere import get_isa_conditions
from scipy.optimize import root

def get_vehicle_performance_at_speed(speed_mps, altitude_m, config, pilot_inputs_guess):
    """
    Calculates the trimmed performance of the helicopter at a specific speed and altitude.
    """
    # Objective function for the trim solver
    def trim_residuals(pilot_inputs_vec, V, alt, cfg):
        pilot_inputs = {
            'collective': pilot_inputs_vec[0],
            'cyclic_lat': pilot_inputs_vec[1],
            'cyclic_lon': pilot_inputs_vec[2],
            'tr_collective': pilot_inputs_vec[3],
        }
        
        _, _, rho = get_isa_conditions(alt)
        main_rotor = Rotor(cfg['main_rotor'])
        tail_rotor = Rotor(cfg['tail_rotor'], is_main_rotor=False)
        
        # Initial guess for TPP angle
        alpha_tpp_rad = np.deg2rad(-2.0)
        
        mr_perf = main_rotor.calculate_forces_moments(V, rho, pilot_inputs, alpha_tpp_rad)
        tr_perf = tail_rotor.calculate_forces_moments(0, rho, {'collective': pilot_inputs['tr_collective'], 'cyclic_lat': 0, 'cyclic_lon': 0})
        
        alpha_f_rad = alpha_tpp_rad - mr_perf['beta_1c'] - np.deg2rad(cfg['vehicle']['shaft_inclination_deg'])
        fuselage_drag, fuselage_lift = calculate_fuselage_forces(cfg, V, rho, alpha_f_rad)
        
        weight = cfg['vehicle']['gross_weight_kg'] * 9.81
        
        # Force and Moment balance equations (residuals to be zeroed)
        Fx = mr_perf['Thrust'] * np.sin(alpha_tpp_rad) - mr_perf['H_force'] - fuselage_drag - tr_perf['Thrust'] # Simplified
        Fz = mr_perf['Thrust'] * np.cos(alpha_tpp_rad) - weight + fuselage_lift
        My = mr_perf['Pitch_moment']
        Mz = mr_perf['Torque'] - tr_perf['Thrust'] * cfg['tail_rotor']['arm_m']
        
        return [Fx, Fz, My, Mz]

    # Run the trim solver
    sol = root(trim_residuals, pilot_inputs_guess, args=(speed_mps, altitude_m, config), method='hybr', tol=1e-3)
    
    if sol.success:
        trimmed_inputs = {
            'collective': sol.x[0], 'cyclic_lat': sol.x[1],
            'cyclic_lon': sol.x[2], 'tr_collective': sol.x[3],
        }
        
        # Recalculate final performance with trimmed inputs
        _, _, rho = get_isa_conditions(altitude_m)
        main_rotor = Rotor(config['main_rotor'])
        tail_rotor = Rotor(config['tail_rotor'], is_main_rotor=False)
        
        mr_power = main_rotor.calculate_forces_moments(speed_mps, rho, trimmed_inputs)['Power_kw']
        tr_power = tail_rotor.calculate_forces_moments(0, rho, {'collective': trimmed_inputs['tr_collective'], 'cyclic_lat': 0, 'cyclic_lon': 0})['Power_kw']
        
        total_power_kw = mr_power + tr_power # Add transmission losses later
        return total_power_kw, True, sol.x
    else:
        return -1, False, sol.x

def calculate_power_curve(config, altitude_m):
    """
    Calculates the power required versus airspeed curve.
    """
    speeds_kmh = np.linspace(10, 420, 25)
    power_req_kw = []
    valid_speeds = []
    
    # Initial guess for pilot inputs [coll, lat, lon, tr_coll] in radians
    pilot_guess = np.deg2rad([8.0, 0.0, -2.0, 5.0])

    for speed_kmh in speeds_kmh:
        speed_mps = speed_kmh * 1000 / 3600
        power, success, new_guess = get_vehicle_performance_at_speed(speed_mps, altitude_m, config, pilot_guess)
        if success:
            power_req_kw.append(power)
            valid_speeds.append(speed_kmh)
            pilot_guess = new_guess # Use last good result as next guess

    return valid_speeds, power_req_kw

