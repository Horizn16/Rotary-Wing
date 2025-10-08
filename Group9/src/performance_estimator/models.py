import numpy as np

def get_airfoil_coeffs(alpha_deg):
    """
    Simplified airfoil model providing Cl and Cd.
    Uses linear lift slope and a parabolic drag polar, with a stall model.
    """
    alpha_rad = np.deg2rad(alpha_deg)
    a = 2 * np.pi  # Lift curve slope
    stall_angle_deg = 15.0
    
    if np.isscalar(alpha_deg):
        if abs(alpha_deg) < stall_angle_deg:
            cl = a * alpha_rad
            cd = 0.01 + 0.015 * cl**2 # Standard parabolic drag polar
        else:
            # Simplified stall model: lift drops, drag rises sharply
            cl = a * np.sin(np.deg2rad(stall_angle_deg)) * np.sign(alpha_deg) * 0.7
            cd = 0.5 + 1.5 * np.sin(alpha_rad - np.deg2rad(stall_angle_deg)*np.sign(alpha_deg))**2
    else:
        cl = np.zeros_like(alpha_deg)
        cd = np.zeros_like(alpha_deg)
        unstalled_mask = np.abs(alpha_deg) < stall_angle_deg
        stalled_mask = ~unstalled_mask

        cl[unstalled_mask] = a * np.deg2rad(alpha_deg[unstalled_mask])
        cd[unstalled_mask] = 0.01 + 0.015 * cl[unstalled_mask]**2
        
        cl[stalled_mask] = a * np.sin(np.deg2rad(stall_angle_deg)) * np.sign(alpha_deg[stalled_mask]) * 0.7
        cd[stalled_mask] = 0.5 + 1.5 * np.sin(np.deg2rad(alpha_deg[stalled_mask]) - np.deg2rad(stall_angle_deg)*np.sign(alpha_deg[stalled_mask]))**2
        
    return cl, cd


class Rotor:
    """
    A class to model a helicopter rotor, calculating its forces, moments, and flapping.
    """
    def __init__(self, config, is_main_rotor=True):
        self.config = config
        self.is_main_rotor = is_main_rotor
        self.radius = config['radius_m']
        self.num_blades = config['num_blades']
        self.chord = config['chord_m']
        self.twist_rad = np.deg2rad(config['twist_deg'])
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
        """
        mu = V_inf / self.tip_speed if self.tip_speed > 0 else 0
        
        thrust_guess = 1.1 * 9.81 * 4050
        ct_guess = thrust_guess / (rho * self.disk_area * self.tip_speed**2) if (rho * self.disk_area * self.tip_speed**2) > 0 else 0
        
        lambda_c = V_inf * np.sin(alpha_tpp_rad) / self.tip_speed if self.tip_speed > 0 else 0
        lambda_i_term = np.sqrt(mu**2 + lambda_c**2)
        lambda_i = ct_guess / (2 * lambda_i_term) if lambda_i_term > 0.01 else np.sqrt(ct_guess/2)
        inflow_ratio = lambda_c + lambda_i

        # --- CORRECTED & STABILIZED FLAPPING EQUATIONS ---
        gamma = self.lock_number
        
        beta_0_num = gamma/2 * ( (1/4)*pilot_inputs['collective']*(1+mu**2) + (1/5)*self.twist_rad - (1/3)*inflow_ratio )
        beta_0_den = (1/3) * (1 + (3/4)*mu**2)
        beta_0 = beta_0_num / beta_0_den if beta_0_den != 0 else 0

        beta_1c_num = -2*mu * ( (2/3)*pilot_inputs['collective'] + (1/2)*self.twist_rad - inflow_ratio )
        beta_1c_den = 1 - (1/2)*mu**2
        beta_1c = beta_1c_num / beta_1c_den if beta_1c_den != 0 else 0
        
        beta_1s_num = (4/3)*mu*beta_0
        beta_1s_den = 1 + (1/2)*mu**2
        beta_1s = -beta_1s_num / beta_1s_den if beta_1s_den != 0 else 0
        
        num_r, num_psi = 15, 36
        r_stations = np.linspace(self.root_cutout, 1.0, num_r)
        psi_stations = np.linspace(0, 2 * np.pi, num_psi, endpoint=False)
        dr = self.radius * (1 - self.root_cutout) / num_r
        
        Thrust, H_force, Y_force, Roll_moment, Pitch_moment, Torque = 0, 0, 0, 0, 0, 0
        max_alpha = 0

        for r_norm in r_stations:
            r = r_norm * self.radius
            for psi in psi_stations:
                if self.is_main_rotor:
                    theta = pilot_inputs['collective'] + r_norm * self.twist_rad + \
                            pilot_inputs['cyclic_lon'] * np.sin(psi) + pilot_inputs['cyclic_lat'] * np.cos(psi)
                else: 
                    theta = pilot_inputs['collective'] + r_norm * self.twist_rad

                U_T = self.omega * r + V_inf * np.sin(psi)
                U_P = inflow_ratio * self.tip_speed + \
                      r * (-beta_1s * np.cos(psi) + beta_1c * np.sin(psi)) * self.omega
                
                phi = np.arctan2(U_P, U_T)
                alpha = np.rad2deg(theta - phi)
                max_alpha = max(max_alpha, abs(alpha))
                
                cl, cd = get_airfoil_coeffs(alpha)
                
                dL = 0.5 * rho * (U_T**2 + U_P**2) * self.chord * cl * dr
                dD = 0.5 * rho * (U_T**2 + U_P**2) * self.chord * cd * dr
                
                dT = dL * np.cos(phi) - dD * np.sin(phi)
                dQ_element = dL * np.sin(phi) + dD * np.cos(phi)
                
                Thrust += dT
                Torque += dQ_element * r
                H_force += dQ_element * np.sin(psi)
                Y_force += -dQ_element * np.cos(psi)
                Roll_moment += dT * r * np.sin(psi)
                Pitch_moment += dT * r * np.cos(psi)
        
        factor = self.num_blades / num_psi
        return {
            'Thrust': Thrust * factor, 'H_force': H_force * factor, 'Y_force': Y_force * factor,
            'Roll_moment': Roll_moment * factor, 'Pitch_moment': Pitch_moment * factor,
            'Torque': Torque * factor, 'Power_kw': (Torque * factor * self.omega) / 1000,
            'beta_0': beta_0, 'beta_1c': beta_1c, 'beta_1s': beta_1s,
            'stall_detected': max_alpha > 15.0
        }

# --- MISSING FUNCTION ADDED BACK ---
def calculate_fuselage_forces(config, V_inf, rho, alpha_f_rad):
    """Calculates fuselage drag and a simplified approximation for fuselage lift."""
    area = config['vehicle']['fuselage_flat_plate_area_m2']
    q = 0.5 * rho * V_inf**2
    drag = q * area
    lift = q * area * (0.1 * np.sin(2 * alpha_f_rad))
    return drag, lift

