#!/usr/bin/env python3
"""
DEEP FOUNDATION DESIGN : USD METHOD
TODO : methology of this app

"""
import os
import re
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from absl import app, flags
from absl.flags import FLAGS


from beam_class import Beam


# Material properties
flags.DEFINE_float("fc", 24, "f'c in MPa")  # 240ksc
flags.DEFINE_float("fy", 425, "yeild for main reinforcement in MPa")
flags.DEFINE_float("fv", 235, "yeild for traverse reinforcement in MPa")
flags.DEFINE_float("Es", 2e5, "Young's Modelus in MPa")

# Try reinf.
flags.DEFINE_integer("main", 16, "initial main bar definition, mm")
flags.DEFINE_integer("trav", 16, "initial traverse bar definition, mm")

# Geometry
flags.DEFINE_float("w", 0, "Width of column cross section, m")
flags.DEFINE_float("h", 0, "Heigth of column cross section, m")

flags.DEFINE_float("B", 0, "Width of footing, m")
flags.DEFINE_float("L", 0, "Long of footing, m")
flags.DEFINE_float("t", 0, "Depth of footing, m")

# Soil properties
flags.DEFINE_float("qa", 300, "Bearing capacity of soil, kN/m2")
flags.DEFINE_float("gamma", 1.8, "Unit weigth of soil, ton/m3")

# Load : dead & live
flags.DEFINE_float("Pu", 0, "Ultimate axial load, kN")

flags.DEFINE_float("Mux", 0, "Mux, kN")
flags.DEFINE_float("Muy", 0, "Muy, kN")
# TODO coding for Mx, My

flags.DEFINE_float("c", 7.5, "Concrete covering, cm")
flags.DEFINE_float("d", 1, "Excavation deep, m")

flags.DEFINE_float("FS", 2.5, "design safety factor")

# flags.DEFINE_float("Qsa", 30, "Pile capacity, tons/pile") # user input instead

from utils import (
    get_valid_integer,
    get_valid_number,
    get_valid_list_input,
    calculate_center_of_gravity,
    display_df,
    toNumpy,
)

from plot_pliecap import plot_pilecap

# Factor and Constants:
CURRENT = os.getcwd()
𝜙b = 0.90
𝜙v = 0.85


# ----------------------------------------
def load_in_pile(
    P, Mx0, My0, N, coordinates, ex, ey, B, L, t, d, gamma, origin, Qsa, designFS
):
    # Calculate Mx and My come from eccentricity
    Mx = Mx0 + P * ey  # kN-m
    My = My0 + P * ex  # kN-m

    # Extract x, y, newX, newY and calculate xi, yi for each coordinate
    processed_coords = [
        {
            "x": float(coord.get("x")),
            "y": float(coord.get("y")),
            "newX": float(coord.get("newX")),
            "newY": float(coord.get("newY")),
        }
        for coord in coordinates
    ]

    for coord in processed_coords:
        coord["xi"] = coord["newX"] - origin["x"]
        coord["yi"] = coord["newY"] - origin["y"]

    # Coefficients: Ix, Iy, Ixy, m, n
    Iy = sum(coord["xi"] ** 2 for coord in processed_coords)
    Ix = sum(coord["yi"] ** 2 for coord in processed_coords)
    Ixy = sum(coord["xi"] * coord["yi"] for coord in processed_coords)

    if Iy == 0 or Ix == 0 or Ixy == 0:
        m = n = 1
    else:
        m = (My * Ix - Mx * Ixy) / (Ix * Iy - Ixy**2)
        n = (Mx * Iy - My * Ixy) / (Ix * Iy - Ixy**2)

    # Self weight and Overburden
    self_weight = -(B * L * t * 2400 * 9.81e-3)  # kN, down
    overburden = -(B * L * d * gamma * 9.81)  # kN, down

    # Load in each pile
    Ri = [
        (-P + self_weight + overburden) / N + m * coord["xi"] + n * coord["yi"]
        for coord in processed_coords
    ]

    # Factor of safety of each pile
    FS = [abs((designFS * Qsa * 9.81) / ri) for ri in Ri]

    print(f"Pu : {P} kN")
    print(f"Mux : {Mx0} kN-m")
    print(f"Muy : {My0} kN-m")
    print(f"Qsa : {Qsa} tons/pile")
    print(f"Soil unit wt. : {gamma} tons/m3")

    print(f"\nSelf-weight : {-self_weight:.2f} kN")
    print(f"Overburden : {-overburden:.2f} kN")

    print(f"\nLoad in each pile: ")
    for i in range(0, len(Ri)):
        print(f"P-{i+1} : {Ri[i]/9.81:.2f} ton")

    print(f"\nFS of each pile: ")
    for i in range(0, len(FS)):
        print(f"P-{i+1} : {np.ceil(FS[i])}")

    return Ri, self_weight, overburden, FS


# Critical for shear, moment and punching shears
def critical_section(d):
    d = d / 100  # m

    # Shear
    vx = FLAGS.w / 2 + d
    vy = FLAGS.h / 2 + d

    # moment
    mx = FLAGS.w / 2
    my = FLAGS.h / 2

    # Punching
    Ap = (FLAGS.w + FLAGS.h + 2 * d) * d  # m2

    return mx, my, vx, vy, Ap  # m ... m2


# Check punching shear
def punching(d, Ru, A_punching):
    d = d / 100
    N = input(
        f"\nEnter PILE NUMBER! againts punching at critical section? ex.1 2 3 4 : "
    )
    N = toNumpy(N).astype(int)  # convert numpy type

    if N.size == 0 or np.any(N, axis=0) == 0:
        return 0
    else:
        """
        N = [1, 4, 7, ...]
        Ru = [R1, R2, R3,...]
        """
        Vup = 0
        for i in range(len(N)):
            Vup = Vup + Ru[N[i] - 1]  # ex.N[0] = 1, R[1-1]

        𝜙Vp = (
            𝜙v * 0.25 * np.sqrt(FLAGS.fc) * A_punching * 1e3
        )  # kN --> f'c = N/mm2, A = mm2

        if 𝜙Vp > abs(Vup):
            print(
                f"𝜙Vp = {𝜙Vp:.2f} kN > Vup ={Vup:.2f} kN --> Punching shear capacity OK"
            )

        else:
            print(
                f"[WARNING!] 𝜙Vp = {𝜙Vp:.2f} kN < Vup ={abs(Vup):.2f} kN Punching shear capacity is OVER!"
            )


# Check beam shear
def shear(B, d, Ru):
    # Calculated shear from Ri of piles
    d = d / 100
    N = input(
        "Enter PILE NUMBER! againts beam shear at critical section?  ex.1 4 5 or 0 if no pile: "
    )
    N = toNumpy(N).astype(int)  # convert numpy type

    if N.size == 0 or np.any(N, axis=0) == 0:
        return 0
    else:
        Vu = 0
        for i in range(len(N)):
            Vu += np.abs(Ru[N[i] - 1])  # ex.N[0] = 1, R[1-1]

        #  shear capacity of concrete section
        𝜙Vn = (
            𝜙v * (1 / 6) * np.sqrt(FLAGS.fc) * B * d * 1e3
        )  # kN --> f'c = N/mm2, Ld = mm2

        if 𝜙Vn > Vu:
            print(f"𝜙Vn = {𝜙Vn:.2f} kN > Vu = {Vu:.2f} kN --> Shear capacity OK")
        else:
            print(f"𝜙Vn = {𝜙Vn:.2f} kN < Vu ={ Vu:.2f} kN --> Shear capacity NOT OK")

        return Vu


# Mu critical at column edge, m
def moment(Ru, moment_arms):
    # Calculate Mu from Ri of piles * arm
    N = input(
        f"\nEnter PILE NUMBER! againts moment at critical section?, ex.1 4 5 or 0 if no pile: "
    )
    N = toNumpy(N).astype(int)  # convert numpy type

    if N.size == 0 or np.any(N, axis=0) == 0:
        return 0
    else:
        moment_arms = abs(moment_arms)
        Mu = 0
        for i in range(len(N)):
            index = N[i] - 1
            Mu += Ru[index] * moment_arms[index]

        print(f"Mu = {Mu:.2f} kN-m")
        return Mu


# Design reinf.
def reinf_design(Mu, instance):

    # Check classification
    instance.classification(Mu)

    # Main bar required
    instance.mainbar_req(Mu)

    # Design main reinf
    no, main_dia, As_main = instance.main_design()

    return no, main_dia, As_main


# ----------------------------------------
def main(_argv):
    print(
        "============================== PILECAP DESIGN : USD METHOD =============================="
    )

    print("[INFO] : PILES INFORMATION")
    # Set the origin coordinate (0, 0)
    origin = {"x": 0, "y": 0}

    # Number of piles
    N = get_valid_integer("How many piles? : ")

    # Pile size (Now only square or rectangle )
    pileSize = get_valid_number("Pile size = ?, unit in cm : ") * 1e-2

    # Pile capacity
    Qsa = get_valid_number("Pile capacity = ? ,tons/pile : ")

    print(f"\n[INFO] : PILES COORDINATES")
    print(
        f"Next to define piles coordinate, please define as graph quater sequence(Q1, Q2, Q3, Q4)"
    )

    # [x1, x2, x3, ...] x-value from C.G. of footing to C.G. of each pile, m
    x_coords = get_valid_list_input(
        f"Define array of x-coordinate of each pile. You've {N} piles in m : ", N
    )

    # [y1, y2, y3, ...] y-value from C.G. of footing to C.G. of each pile, m
    y_coords = get_valid_list_input(
        f"\nDefine array of y-coordinate of each pile. You've {N} piles in m : ", N
    )

    x_deviate = (
        get_valid_list_input(
            f"\nDefine array of x-deviation of each pile. You've {N} piles in cm : ", N
        )
        * 1e-2
    )  # convert cm to m
    y_deviate = (
        get_valid_list_input(
            f"\nDefine array of y-deviation of each pile. You've {N} piles in cm : ", N
        )
        * 1e-2
    )  # convert cm to m

    # Create index of piles
    piles_number = []
    for a, b in enumerate(x_coords):
        piles_number.append(a + 1)

    # Store coordinates of planed piles
    planedPile = [{"x": x, "y": y} for x, y in zip(x_coords, y_coords)]  # m

    # Store coordinates of actual piles
    actualPile = [
        {"x": x + dx, "y": y + dy}
        for x, y, dx, dy in zip(x_coords, y_coords, x_deviate, y_deviate)
    ]  # m

    # Store coordinates of whole piles
    coordinates = [
        {"x": x, "y": y, "newX": x + dx, "newY": y + dy}
        for x, y, dx, dy in zip(x_coords, y_coords, x_deviate, y_deviate)
    ]

    print(f"\n[INFO] : CALCULATE LOAD IN EACH PILE")
    # Calculate new CoG
    weights = [1 for coord in coordinates]
    new_CoG = calculate_center_of_gravity(actualPile, weights)

    # Eccentricity
    ex = new_CoG["x"] - origin["x"]  # m
    ey = new_CoG["y"] - origin["y"]  # m

    # Loads in each pile : +Tension, -Compression
    Ri, self_weight, overburden, FS = load_in_pile(
        FLAGS.Pu,
        FLAGS.Mux,
        FLAGS.Muy,
        N,
        coordinates,
        ex,
        ey,
        FLAGS.B,
        FLAGS.L,
        FLAGS.t,
        FLAGS.d,
        FLAGS.gamma,
        origin,
        Qsa,
        FLAGS.FS,
    )

    Ri = np.array(Ri)
    FS = np.array(FS)

    # Effective depth
    d1 = FLAGS.c + FLAGS.main / 5 + FLAGS.trav / 10  # cm
    d = 100 * FLAGS.t - d1  # cm

    # Calculate critical section for M, V, Vp
    mx, my, vx, vy, Ap = critical_section(d)

    # Display pilecap
    plot_pilecap(
        FLAGS.w,
        FLAGS.h,
        FLAGS.B,
        FLAGS.L,
        d,
        mx,
        my,
        vx,
        vy,
        origin,
        piles_number,
        pileSize,
        planedPile,
        actualPile,
    )

    # check punching shear capacity
    # for symmetry footing each R in Rn, Ru is equal
    punching(d, Ri, Ap)

    print(f"\n[INFO] : DESIGN REINFORCEMENT")
    W = FLAGS.w
    D = FLAGS.h
    B = FLAGS.B
    L = FLAGS.L

    # Display rebar df
    table = os.path.join(CURRENT, "data/Deform_Bar.csv")
    df = pd.read_csv(table)
    display_df(df)

    print(f"\n==================== X-X Axis ====================")

    # check shear capacity
    Vu = shear(B, d, -1 * Ri)

    # #check moment capacity
    Mu = moment(-1 * Ri, x_coords)

    # Instanciate
    beam = Beam(fc=FLAGS.fc, fy=FLAGS.fy, fv=FLAGS.fv, c=FLAGS.c)

    beam.section_properties(FLAGS.main, FLAGS.trav, L * 100, FLAGS.t * 100)
    d, d1 = beam.eff_depth()
    beam.capacity()

    # Design reinf.
    no, main_dia, As_main = reinf_design(Mu, beam)

    print(f"\n==================== Y-Y Axis ====================")
    # Swapp variable
    W, D = D, W
    B, L = L, B

    # check shear capacity
    Vu = shear(B, d, -1 * Ri)

    # check moment capacity
    Mu = moment(-1 * Ri, x_coords)

    # Instanciate
    beam.section_properties(FLAGS.main, FLAGS.trav, L * 100, FLAGS.t * 100)
    d, d1 = beam.eff_depth()
    beam.capacity()

    # Design reinf.
    no, main_dia, As_main = reinf_design(Mu, beam)

    return


if __name__ == "__main__":
    app.run(main)

"""
-run script
    cd <path to project directory>
    conda activate <your conda env name>

    ex.2-piles
    python app/pilecap_design.py --w=0.2 --h=0.2 --B=0.5 --L=0.5 --t=0.3 --Pu=80 --Mux=0 --Muy=2

    ex.4-piles
    python app/pilecap_design.py --w=0.4 --h=0.6 --B=2.1 --L=1.5 --t=0.75 --Pu=2300 --Mux=60 --Muy=90

    .75 -.75 -.75 .75
    .45 .45 -.45 -.45
    2 -5 3 3
    3 1 -4 -3
"""
