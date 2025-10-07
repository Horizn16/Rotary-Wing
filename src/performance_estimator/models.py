import numpy as np

def get_airfoil_coeffs(alpha_deg):
    """
    Simplified airfoil model providing Cl and Cd.
    Uses linear lift slope and a parabolic drag polar, with a stall model.
    """
    alpha_rad = np.deg2rad(alpha_deg)
    a = 2 * np.pi  # Lift curve slope (per radian)
    stall_angle_deg = 15.0
    
    if abs(alpha_deg) < stall_angle_deg:
        # Pre-stall region
        cl = a * alpha_rad
        cd = 0.01 + 0.01 * cl**2  # Simplified drag polar
    else:
        # Post-stall region (simplified model)
        cl = a * np.sin(np.deg2rad(stall_angle_deg)) * np.sign(alpha_deg)
        cd = 0.5 + 0.5 * np.sin(alpha_rad - np.deg2rad(stall_angle_deg)*np.sign(alpha_deg))**2
    return cl, cd, stall_angle_deg

class Rotor:
    """
    Models a helicopter rotor, calculating its forces, moments, and flapping.
    """
    def __init__(self, config, is_main_rotor=True):
        self.config = config
        self.is_main_rotor = is_main_rotor
        self.radius = config['radius_m']
        self.num_blades = config['num_blades']
        self.chord = config['chord_m']
        self.twist_deg = config['twist_deg']
        self.rpm = config['rpm']
        self.omega = self.rpm * np.pi / 30
        self.tip_speed = self.omega * self.radius
        self.disk_area = np.pi * self.radius**2
        self.solidity = (self.num_blades * self.chord) / (np.pi * self.radius)
        self.lock_number = config.get('lock_number', 8)
        self.root_cutout = config.get('root_cutout_ratio', 0.1)

    def calculate_forces_moments(self, V_inf, rho, pilot_inputs, alpha_tpp_rad=0.0):
        """
        Calculates total rotor forces and moments using a BEMT approach.
        Also checks for blade stall on the retreating side.
        """
        mu = V_inf / self.tip_speed
        
        # Simplified inflow model (Glauert's method)
        thrust_guess = 1.2 * 9.81 * 4000
        ct_guess = thrust_guess / (rho * self.disk_area * self.tip_speed**2)
        lambda_c = V_inf * np.sin(alpha_tpp_rad) / self.tip_speed
        lambda_i = ct_guess / (2 * np.sqrt(mu**2 + lambda_c**2)) if mu > 0.01 else np.sqrt(ct_guess/2)
        inflow_ratio = lambda_c + lambda_i

        # Flapping dynamics based on slide equations
        beta_0 = (self.lock_number / 8) * (pilot_inputs['collective'] / 6 + (1 + mu**2) * self.config['twist_deg'] / 8 - inflow_ratio / 6)
        beta_1c = - (2 * mu * (4/3 * pilot_inputs['collective'] + self.config['twist_deg'] - 2 * inflow_ratio)) / (1 - 0.5 * mu**2)
        beta_1s = - (4/3 * mu * beta_0) / (1 + 0.5 * mu**2)
        
        # Numerical integration over the blade and azimuth
        num_r, num_psi = 15, 36
        r_stations = np.linspace(self.root_cutout, 1.0, num_r)
        psi_stations = np.linspace(0, 2 * np.pi, num_psi, endpoint=False)
        dr = self.radius * (1 - self.root_cutout) / num_r
        
        Thrust, H_force, Y_force, Roll_moment, Pitch_moment, Torque = 0, 0, 0, 0, 0, 0
        stall_detected = False

        for r_norm in r_stations:
            r = r_norm * self.radius
            for psi in psi_stations:
                theta = pilot_inputs['collective'] + r_norm * np.deg2rad(self.config['twist_deg']) + \
                        pilot_inputs['cyclic_lon'] * np.sin(psi) + pilot_inputs['cyclic_lat'] * np.cos(psi)
                
                U_T = self.omega * r + V_inf * np.sin(psi)
                U_P = inflow_ratio * self.tip_speed + \
                      r * (-beta_1s * np.cos(psi) + beta_1c * np.sin(psi)) * self.omega
                
                phi = np.arctan2(U_P, U_T)
                alpha = np.rad2deg(theta - phi)
                cl, cd, stall_angle = get_airfoil_coeffs(alpha)
                
                # Check for stall on the retreating blade tip (180 < psi < 360)
                if self.is_main_rotor and r_norm > 0.8 and np.pi < psi < 2*np.pi and abs(alpha) > stall_angle:
                    stall_detected = True
                
                dL = 0.5 * rho * (U_T**2 + U_P**2) * self.chord * cl * dr
                dD = 0.5 * rho * (U_T**2 + U_P**2) * self.chord * cd * dr
                
                dT = dL * np.cos(phi) - dD * np.sin(phi)
                dQ = (dL * np.sin(phi) + dD * np.cos(phi)) * r
                
                Thrust += dT
                Torque += dQ
                H_force += (dL * np.sin(phi) + dD * np.cos(phi)) * np.sin(psi)
                Y_force += -(dL * np.sin(phi) + dD * np.cos(phi)) * np.cos(psi)
                Roll_moment += dT * r * np.sin(psi)
                Pitch_moment += dT * r * np.cos(psi)
        
        # Average over azimuth and multiply by number of blades
        factor = self.num_blades / num_psi
        return {
            'Thrust': Thrust * factor, 'H_force': H_force * factor, 'Y_force': Y_force * factor,
            'Roll_moment': Roll_moment * factor, 'Pitch_moment': Pitch_moment * factor,
            'Torque': Torque * factor, 'Power_kw': (Torque * factor * self.omega) / 1000,
            'beta_0': beta_0, 'beta_1c': beta_1c, 'beta_1s': beta_1s,
            'stall_detected': stall_detected
        }

def calculate_fuselage_forces(config, V_inf, rho, alpha_f_rad):
    """Calculates fuselage drag and a simplified lift component."""
    area = config['vehicle']['fuselage_flat_plate_area_m2']
    q = 0.5 * rho * V_inf**2
    drag = q * area
    # Simple approximation for fuselage lift based on angle of attack
    lift = q * area * (0.1 * np.sin(2 * alpha_f_rad))
    return drag, lift

