import numpy as np

def get_isa_conditions(altitude_m: float):
    """
    Calculates atmospheric conditions based on the International Standard Atmosphere (ISA).
    
    Args:
        altitude_m: Geometric altitude in meters.
        
    Returns:
        A tuple containing (temperature_K, pressure_Pa, density_kg_m3).
    """
    # ISA Sea Level Constants
    T0 = 288.15      # K
    P0 = 101325.0    # Pa
    RHO0 = 1.225     # kg/m^3
    g = 9.80665      # m/s^2
    R = 287.058      # J/(kg*K)
    a = -0.0065      # K/m, temperature lapse rate in troposphere

    altitude_m = min(altitude_m, 11000) # Limit to troposphere

    temp_K = T0 + a * altitude_m
    pressure_Pa = P0 * (temp_K / T0)**(-g / (a * R))
    density_kg_m3 = pressure_Pa / (R * temp_K)
    
    return temp_K, pressure_Pa, density_kg_m3
