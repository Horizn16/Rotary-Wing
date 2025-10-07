import numpy as np

def get_isa_conditions(altitude_m: float):
    """
    Calculates atmospheric conditions based on the International Standard Atmosphere (ISA).
    """
    T0 = 288.15      # K, Sea Level Temperature
    P0 = 101325.0    # Pa, Sea Level Pressure
    RHO0 = 1.225     # kg/m^3, Sea Level Density
    g = 9.80665      # m/s^2, Gravity
    R = 287.058      # J/(kg*K), Specific Gas Constant for Dry Air
    a = -0.0065      # K/m, Temperature Lapse Rate in Troposphere

    # Limit calculations to the troposphere (up to 11 km)
    altitude_m = min(altitude_m, 11000)

    temp_K = T0 + a * altitude_m
    pressure_Pa = P0 * (temp_K / T0)**(-g / (a * R))
    density_kg_m3 = pressure_Pa / (R * temp_K)
    
    return temp_K, pressure_Pa, density_kg_m3

