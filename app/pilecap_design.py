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

from beam import Beam
from rc.plot_shape import rectangleShape, addShape, add_Hline, add_Vline
from rc.utils import Utils

from beam import Beam

from absl import app, flags
from absl.flags import FLAGS


# Factor and Constants:
CURR = os.getcwd()
df = pd.read_csv(os.path.join(CURR, "sections", "Deform_Bar.csv"))
𝜙b = 0.90
𝜙v = 0.85

𝜙 = {"6": 6, "9": 9, "12": 12, "16": 16, "20": 20, "25": 25, "28": 28, "32": 32}  # mm
A = {
    "6": 0.2827,
    "9": 0.636,
    "12": 1.131,
    "16": 2.01,
    "20": 3.146,
    "25": 4.908,
    "28": 6.157,
    "32": 6.313,
}  # cm2

ut = Utils()

# try initial rebar
𝜙1 = 16  # main
𝜙2 = 9  # traverse
As = A[str(𝜙1)]
Av = 2 * A[str(𝜙2)]

# Material properties
flags.DEFINE_float("fc", 24, "f'c in MPa")  # 240ksc
flags.DEFINE_float("fy", 425, "yeild for main reinforcement in MPa")
flags.DEFINE_float("fv", 235, "yeild for traverse reinforcement in MPa")
flags.DEFINE_float("Es", 2e5, "Young's Modelus in MPa")

# Geometry
flags.DEFINE_float("W", 0, "Width of column cross section, m")
flags.DEFINE_float("H", 0, "Heigth of column cross section, m")

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
flags.DEFINE_float("h", 1, "Excavation deep, m")

flags.DEFINE_float("FS", 2.5, "design safety factor")

# flags.DEFINE_float("Qsa", 30, "Pile capacity, tons/pile") # user input instead


# ----------------------------------------
# Capture coordinates input
def toNumpy(x):
    x = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", x)
    return np.array([float(n) for n in x])


def load_in_pile(
    P, Mx0, My0, N, coordinates, ex, ey, B, L, t, h, gamma, origin, Qsa, designFS
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

    print(*coordinates)
    print(*processed_coords)

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
    overburden = -(B * L * h * gamma * 9.81)  # kN, down

    # Load in each pile
    Ri = [
        (-P + self_weight + overburden) / N + m * coord["xi"] + n * coord["yi"]
        for coord in processed_coords
    ]

    # Factor of safety of each pile
    FS = [abs((designFS * Qsa * 9.81) / ri) for ri in Ri]

    print(f"\nPu : {P} kN")
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
    vx = FLAGS.W / 2 + d
    vy = FLAGS.H / 2 + d

    # moment
    mx = FLAGS.W / 2
    my = FLAGS.H / 2

    # Punching
    Ap = (FLAGS.W + FLAGS.H + 2 * d) * d  # m2

    return mx, my, vx, vy, Ap  # m ... m2


# Check punching shear
def punching(d, Ru, A_punching):
    d = d / 100
    N = input("Enter PILE NUMBER! againts punching at critical section? ex.1 2 3 4 : ")
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

        if 𝜙Vp > Vup:
            print(
                f"𝜙Vp = {𝜙Vp:.2f} kN > Vup ={Vup:.2f} kN --> Punching shear capacity OK"
            )
        else:
            print(
                f"𝜙Vp = {𝜙Vp:.2f} kN < Vup ={Vup:.2f} kN --> Punching shear capacity NOT OK"
            )


# Check beam shear
def shear(B, d, Ru):
    # Calculated shear from Ri of piles
    d = d / 100
    N = input(
        "Enter PILE NUMBER! againts shear at critical section?  ex.1 4 5 or 0 if no pile: "
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
        "Enter PILE NUMBER! againts moment at critical section?, ex.1 4 5 or 0 if no pile: "
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


def plot_figures():
    fig = go.Figure()
    fig.update_layout(showlegend=False, title="Foundation Layout")
    return fig


# ----------------------------------------
def main(_argv):
    print(
        "========================================================================================================"
    )
    print('PILECAP DESIGN : USD METHOD')

    # ----------------------------------------------------------------
    # Input piles coordinates and calculate load in each pile
    # ----------------------------------------------------------------

    # Set the origin coordinate value, you can change.
    origin = {"x": 0, "y": 0}

    # Input number of piles
    N = int(input("How many piles? : "))  # TODO error raise if not digit

    # Input pile size
    pileSize = float(
        input("Pile size = ?, unit in meters : ")
    )  # TODO error raise if not digit

    # Input pile capacity
    Qsa = float(
        input("Pile capacity = ? ,tons/pile : ")
    )  # TODO error raise if not digit

    # Input pile coordinate --> Quater path : Q1, Q2, Q3, Q4
    while True:
        # [x1, x2, x3, ...] x-value from C.G. of footing to C.G. of each pile, m
        xCoords = input(
            f"Define array of x-coordinate from C.G. of footing to C.G. of each pile. You've {N} piles, m : "
        )
        xCoords = toNumpy(xCoords)  # m

        # [y1, y2, y3, ...] y-value from C.G. of footing to C.G. of each pile, m
        yCoords = input(
            f"Define array of y-coordinate from C.G. of footing to C.G. of each pile. You've {N} piles, m : "
        )
        yCoords = toNumpy(yCoords)  # m

        deviateX = input(
            f"Define array of x-deviation of each pile. You've {N} piles, cm : "
        )
        deviateX = toNumpy(deviateX) * 1e-2  # m

        deviateY = input(
            f"Define array of y-deviation of each pile. You've {N} piles, cm : "
        )
        deviateY = toNumpy(deviateY) * 1e-2  # m

        if (
            len(xCoords) == N
            and len(yCoords) == N
            and len(deviateX) == N
            and len(deviateY) == N
        ):
            break
        else:
            print("Try again!")

    # Create index of piles
    piles_number = []
    for a, b in enumerate(xCoords):
        piles_number.append(a + 1)

    # Store coordinates of planed piles
    planedPile = [{"x": x, "y": y} for x, y in zip(xCoords, yCoords)]  # m

    # Store coordinates of actual piles
    actualPile = [
        {"x": x + dx, "y": y + dy}
        for x, y, dx, dy in zip(xCoords, yCoords, deviateX, deviateY)
    ]  # m

    # Store coordinates of whole piles
    coordinates = [
        {"x": x, "y": y, "newX": x + dx, "newY": y + dy}
        for x, y, dx, dy in zip(xCoords, yCoords, deviateX, deviateY)
    ]

    # Calculate new CoG
    weights = [1 for coord in coordinates]
    new_CoG = ut.calculate_center_of_gravity(actualPile, weights)

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
        FLAGS.h,
        FLAGS.gamma,
        origin,
        Qsa,
        FLAGS.FS,
    )

    Ri = np.array(Ri)
    FS = np.array(FS)

    # Effective depth
    d1 = FLAGS.c + 𝜙2 / 10 + 𝜙1 / 10 / 2  # cm
    d = 100 * FLAGS.t - d1  # cm

    print("--------------------------------------------------------------")
    mx, my, vx, vy, Ap = critical_section(d)

    # ----------------------------------------------------------------
    # Plot pile cap
    # ----------------------------------------------------------------
    fig = plot_figures()

    # Column and pilecap shapes
    column_shapes = rectangleShape(origin, FLAGS.W, FLAGS.H)
    footing_shapes = rectangleShape(origin, FLAGS.B, FLAGS.L)
    addShape(fig, column_shapes, color="#b3bcc9")
    addShape(fig, footing_shapes, color="#b3bcc9")

    # Planed piles shapes
    for coord, label in zip(planedPile, piles_number):
        CL = {"x": coord["x"], "y": coord["y"]}
        planedPileShape = rectangleShape(CL, pileSize, pileSize)
        addShape(fig, planedPileShape, color="#1a66a1")

        # Annotations
        fig.add_annotation(
            text=str(label),
            x=coord["x"],
            y=coord["y"],
            showarrow=False,
            font=dict(color="red", size=11),
        )

    # Actual piles shapes
    for coord, label in zip(actualPile, piles_number):
        CL = {"x": coord["x"], "y": coord["y"]}
        actualPileShape = rectangleShape(CL, pileSize, pileSize)
        addShape(fig, actualPileShape, color="#eb9234")

    # Critical line of M
    add_Vline(fig, mx, [-FLAGS.L / 2, FLAGS.L / 2], color="green")
    add_Hline(fig, my, [-FLAGS.B / 2, FLAGS.B / 2], color="green")

    # Critical line of V
    add_Vline(fig, vx, [-FLAGS.L / 2, FLAGS.L / 2], color="blue")
    add_Hline(fig, vy, [-FLAGS.B / 2, FLAGS.B / 2], color="blue")

    # Ctrical bound of punching
    punching_shapes = rectangleShape(origin, FLAGS.W + d * 1e-2, FLAGS.H + d * 1e-2)
    addShape(fig, punching_shapes, line_type="dash", fill_option=False, color="#eb9234")

    fig.show()

    # ----------------------------------------------------------------
    # Design reinforcement
    # ----------------------------------------------------------------
    W = FLAGS.W
    D = FLAGS.H
    B = FLAGS.B
    L = FLAGS.L

    # check punching shear capacity
    # for symmetry footing each R in Rn, Ru is equal
    punching(d, Ri, Ap)

    # X-X axis
    print(f"\nX-X Axis :")

    # check shear capacity
    Vu = shear(B, d, -1 * Ri)
    print("--------------------------------------------------------------")

    # #check moment capacity
    Mu = moment(-1 * Ri, xCoords)
    print("--------------------------------------------------------------")

    beam = Beam(FLAGS.fc, FLAGS.fy, FLAGS.fv, FLAGS.c)

    beam.initial(𝜙1, 𝜙2, B * 100, FLAGS.t * 100, L, Mu, Vu=0)

    β1 = beam.beta()
    d, d1 = beam.eff_depth()
    pmin, pmax1, p = beam.percent_reinf()

    # Calculate 𝜙Mn
    𝜙Mn1 = beam.capacity(d)

    # Check classification
    classify = beam.classification(Mu, 𝜙Mn1)

    # Main bar required
    data = beam.mainbar_req(d, d1, 𝜙Mn1, Mu, classify)

    # Design main reinf
    beam.db()
    dia_main, As_main = beam.main_call(data)

    # ----------------------------------------------------------------
    # Y-Y axis
    print("===================================================================")
    ## swap variable for y-y axis
    print(f"\nY-Y Axis :")
    W, D = D, W
    B, L = L, B

    # check shear capacity
    Vu = shear(B, d, -1 * Ri)
    print("--------------------------------------------------------------")

    # #check moment capacity
    Mu = moment(-1 * Ri, xCoords)
    print("--------------------------------------------------------------")

    beam = Beam(FLAGS.fc, FLAGS.fy, FLAGS.fv, FLAGS.c)
    beam.initial(𝜙1, 𝜙2, B * 100, FLAGS.t * 100, L, Mu, Vu=0)

    β1 = beam.beta()
    d, d1 = beam.eff_depth()
    pmin, pmax1, p = beam.percent_reinf()

    # Calculate 𝜙Mn
    𝜙Mn1 = beam.capacity(d)

    # Check classification
    classify = beam.classification(Mu, 𝜙Mn1)

    # Main bar required
    data = beam.mainbar_req(d, d1, 𝜙Mn1, Mu, classify)

    # Design main reinf
    beam.db()
    beam.main_call( data)


if __name__ == "__main__":
    app.run(main)

# TODO bandwidth for reinf.layout, round pile
"""
-Please see FLAGS definition for unit informations
-Make sure you are in the project directory run python in terminal(Mac) or command line(Windows)
-run script
    cd <path to project directory>
    conda activate <your conda env name>

    ex.2-piles
    python app/pilecap_design.py --W=0.2 --H=0.2 --B=0.5 --L=0.5 --t=0.3 --Pu=80 --Mux=0 --Muy=2

    ex.4-piles
    python app/pilecap_design.py --W=0.4 --H=0.6 --B=2.1 --L=1.5 --t=0.75 --Pu=2300 --Mux=60 --Muy=90

    .75 -.75 -.75 .75
    .45 .45 -.45 -.45
    2 -5 3 3
    3 1 -4 -3
"""
