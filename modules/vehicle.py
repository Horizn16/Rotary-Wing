import numpy as np
import math

RHO_DEFAULT = 1.225

class Airfoil:
    def __init__(self, a0, Cd0, e):
        self.a0 = a0
        self.Cd0 = Cd0
        self.e = e
    def Cl(self, alpha): return self.a0 * alpha
    def Cd(self, alpha): return self.Cd0 + self.e * alpha**2

class Blade:
    """
    Now supports linear taper & linear twist and a root cut-out.
    Provide c_root, c_tip, theta_root(rad), theta_tip(rad), R_root, R_tip.
    """
    def __init__(self, R_root, R_tip, c_root, c_tip, theta_root, theta_tip, airfoil: Airfoil):
        self.R_root = R_root
        self.R_tip  = R_tip
        self.c_root = c_root
        self.c_tip  = c_tip
        self.theta_root = theta_root
        self.theta_tip  = theta_tip
        self.airfoil = airfoil

    def c(self, r):
        # linear taper across [R_root, R_tip]
        mu = (r - self.R_root) / (self.R_tip - self.R_root)
        mu = min(max(mu, 0.0), 1.0)
        return self.c_root + (self.c_tip - self.c_root) * mu

    def theta(self, r):
        # linear twist across [R_root, R_tip]
        mu = (r - self.R_root) / (self.R_tip - self.R_root)
        mu = min(max(mu, 0.0), 1.0)
        return self.theta_root + (self.theta_tip - self.theta_root) * mu

class Rotor:
    def __init__(self, B, blade: Blade):
        self.B = B
        self.blade = blade

    def solidity_local(self, r):
        # local solidity based on circumference annulus
        return (self.B * self.blade.c(r)) / (2 * math.pi * r)
    
    # Prandtl loss at a distance r from the center
    def prandtl_loss(self, r, lambda_v):
        return (2/np.pi) * np.arccos(np.exp((-self.B/2)*((1 - r/self.blade.R_tip)/lambda_v)))