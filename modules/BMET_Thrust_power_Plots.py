import numpy as np
import math
import matplotlib.pyplot as plt

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

def prandtl_F_lambda(B, r, R, lambda_):
    
    lambda_safe = max(abs(lambda_), 1e-8)
    f = 0.5 * B * (1.0 - r / R) / lambda_safe
    f = min(max(f, 1e-8), 50.0)
    exp_arg = math.exp(-f)
    exp_arg = min(max(exp_arg, -1.0), 1.0)
    F = (2.0 / math.pi) * math.acos(exp_arg)
    return max(F, 1e-6)

def solve_annulus_vi_lambda(rotor, r, V, omega, rho, max_iter=200, tol=1e-8, damp=0.6):
    b = rotor.blade
    B = rotor.B
    Ut = omega * r
    c = b.c(r)
    th = b.theta(r)

    # initial guess for vi
    vi = 0.05 * Ut
    for _ in range(max_iter):
        Uax = V + vi
        phi = math.atan2(Uax, Ut)

        lambda_local = (V + vi) / (omega * r) if (omega * r) != 0.0 else 1e-8
        F = prandtl_F_lambda(B, r, b.R_tip, lambda_local)

        U = math.hypot(Ut, Uax)
        q = 0.5 * rho * U*U
        alpha = th - phi
        Cl = b.airfoil.Cl(alpha)
        Cd = b.airfoil.Cd(alpha)

        dT_BE_dr = B * q * c * (Cl*math.cos(phi) - Cd*math.sin(phi))
        dT_MT_dr = 4.0 * math.pi * rho * F * r * Uax * vi

        Rres = dT_BE_dr - dT_MT_dr
        if abs(Rres) < tol * (1.0 + abs(dT_BE_dr)):
            break

        # finite-difference approx for derivative wrt vi (damped)
        dvi = max(1e-4, 0.01*(vi + 1.0))
        vi_p = max(0.0, vi + dvi)
        Uax_p = V + vi_p
        phi_p = math.atan2(Uax_p, Ut)
        lambda_p = (V + vi_p) / (omega * r) if (omega * r) != 0.0 else 1e-8
        F_p = prandtl_F_lambda(B, r, b.R_tip, lambda_p)
        U_p = math.hypot(Ut, Uax_p); q_p = 0.5 * rho * U_p*U_p
        alpha_p = th - phi_p
        Cl_p = b.airfoil.Cl(alpha_p); Cd_p = b.airfoil.Cd(alpha_p)
        dT_BE_dr_p = B * q_p * c * (Cl_p*math.cos(phi_p) - Cd_p*math.sin(phi_p))
        dT_MT_dr_p = 4.0 * math.pi * rho * F_p * r * Uax_p * vi_p
        Rres_p = dT_BE_dr_p - dT_MT_dr_p

        dR_dvi = (Rres_p - Rres) / dvi if abs(Rres_p - Rres) > 1e-16 else 1.0
        step = -Rres / dR_dvi
        vi = max(0.0, vi + damp * step)

    # final local quantities
    Uax = V + vi
    phi = math.atan2(Uax, Ut)
    U = math.hypot(Ut, Uax)
    q = 0.5 * rho * U*U
    alpha = th - phi
    Cl = b.airfoil.Cl(alpha)
    Cd = b.airfoil.Cd(alpha)
    return vi, phi, q, Cl, Cd, U

def solver(rotor: Rotor, V, omega, rho=1.225, n_sections=80):
    b = rotor.blade
    mu = np.linspace(0, 1, n_sections)
    r_nodes = 0.5*(1 - np.cos(np.pi*mu))
    r = b.R_root + (b.R_tip - b.R_root) * r_nodes
    dr = np.gradient(r)

    T = 0.0
    Q = 0.0
    for ri, dri in zip(r, dr):
        c = b.c(ri)
        if c <= 0:
            continue
        vi, phi, q, Cl, Cd, U = solve_annulus_vi_lambda(rotor, ri, V, omega, rho)
        Lp = q * c * Cl
        Dp = q * c * Cd
        dT = rotor.B * (Lp * math.cos(phi) - Dp * math.sin(phi)) * dri
        dQ = rotor.B * (Lp * math.sin(phi) + Dp * math.cos(phi)) * ri * dri
        T += dT
        Q += dQ

    P = Q * omega
    R_tip = b.R_tip

    # Performance coefficients
    CT = 2 * T / (rho * math.pi * omega**2 * R_tip**4)
    CQ = 2 * Q / (rho * math.pi * omega**2 * R_tip**5)
    CP = 2 * Q / (rho * math.pi * omega**3 * R_tip**5)
    
    return T, Q, P, CT, CQ, CP

# ---------------------------
# Test case
# ---------------------------
if __name__ == '__main__':
    airfoil = Airfoil(a0=5.75, Cd0=0.0113, e=1.25)
    rpm = 960
    omega = (np.pi / 30.0) * rpm
    rho = 1.225

    R_root, R_tip = 0.125, 0.762
    c_root_default = 0.0508
    theta_default = np.deg2rad(5)   # pick some mid pitch

    # ------------------------------------------------
    # Sweep 1: Thrust vs No. of Blades / Power vs No. of Blades
    # ------------------------------------------------
    B_values = [2, 3, 4, 5]
    T_B, P_B = [], []

    for B in B_values:
        blade = Blade(R_root, R_tip, c_root_default, c_root_default,
                      theta_default, theta_default, airfoil)
        rotor = Rotor(B, blade)
        T, Q, P, *_ = solver(rotor, V=0, omega=omega, rho=rho, n_sections=200)
        T_B.append(T)
        P_B.append(P)

    plt.figure()
    plt.plot(B_values, T_B, "o-", label="Thrust")
    plt.xlabel("Number of blades B")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust vs Number of blades")

    plt.figure()
    plt.plot(B_values, P_B, "s-", label="Power", color="red")
    plt.xlabel("Number of blades B")
    plt.ylabel("Power [W]")
    plt.title("Power vs Number of blades")

    # ------------------------------------------------
    # Sweep 2: Thrust vs Taper ratio / Power vs Taper ratio
    # ------------------------------------------------
    taper_ratios = np.linspace(0.3, 1.0, 8)   # c_tip / c_root
    T_taper, P_taper = [], []
    B = 4  # fix number of blades

    for tr in taper_ratios:
        c_tip = tr * c_root_default
        blade = Blade(R_root, R_tip, c_root_default, c_tip,
                      theta_default, theta_default, airfoil)
        rotor = Rotor(B, blade)
        T, Q, P, *_ = solver(rotor, V=0, omega=omega, rho=rho, n_sections=200)
        T_taper.append(T)
        P_taper.append(P)

    plt.figure()
    plt.plot(taper_ratios, T_taper, "o-")
    plt.xlabel("Taper ratio (c_tip / c_root)")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust vs Taper Ratio")

    plt.figure()
    plt.plot(taper_ratios, P_taper, "s-", color="red")
    plt.xlabel("Taper ratio (c_tip / c_root)")
    plt.ylabel("Power [W]")
    plt.title("Power vs Taper Ratio")

    # ------------------------------------------------
    # Sweep 3: Thrust vs Twist / Power vs Twist
    # ------------------------------------------------
    twists = np.deg2rad(np.linspace(0, 20, 8))  # twist = theta_tip - theta_root
    T_twist, P_twist = [], []
    B = 4

    for twist in twists:
        blade = Blade(R_root, R_tip, c_root_default, c_root_default,
                      theta_default, theta_default + twist, airfoil)
        rotor = Rotor(B, blade)
        T, Q, P, *_ = solver(rotor, V=0, omega=omega, rho=rho, n_sections=200)
        T_twist.append(T)
        P_twist.append(P)

    plt.figure()
    plt.plot(np.rad2deg(twists), T_twist, "o-")
    plt.xlabel("Twist (θ_tip - θ_root) [deg]")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust vs Twist")

    plt.figure()
    plt.plot(np.rad2deg(twists), P_twist, "s-", color="red")
    plt.xlabel("Twist (θ_tip - θ_root) [deg]")
    plt.ylabel("Power [W]")
    plt.title("Power vs Twist")

    # Show all plots
    plt.show()