### ISOLATED FOOTING DESIGN : USD METHOD
###  Adopt from : https://www.scribd.com/document/400417851/191061 by รศ.อมร พิมานมาศ และคณะ'

import numpy as np
import pandas as pd

from absl import app, flags, logging
from absl.flags import FLAGS

from beam import Beam

## FLAGS definition
flags.DEFINE_float("fc", 23.5, "240ksc, MPa")
flags.DEFINE_integer("fy", 392, "SD40 main bar, MPa")
flags.DEFINE_integer("fv", 235, "SR24 traverse, MPa")

flags.DEFINE_integer("main", 12, "initial main bar definition, mm")
flags.DEFINE_integer("trav", 9, "initial traverse bar definition, mm")

flags.DEFINE_float("c", 7.5, "concrete covering, cm")
flags.DEFINE_float("B", 0, "footing size B x L x t, m")
flags.DEFINE_float("L", 0, "footing size B x L x t, m")
flags.DEFINE_float("t", 0, "footing size B x L x t, m")

flags.DEFINE_float("bc", 0, "column section bc x hc, m")
flags.DEFINE_float("hc", 0, "column section bc x hc, m")

flags.DEFINE_float("Pu", 0, "Axial Load, kN")
flags.DEFINE_float("Mux", 0, "Moment, kN-m")
flags.DEFINE_float("Muy", 0, "Moment, kN-m")

flags.DEFINE_float("qa", 300, "Bearing capacity of soil(kN/m2)")
flags.DEFINE_float("gamma", 18, "Unit wt of soil(kN/m3)")

##===============================================================================
# CONSTANCE
𝜙b = 0.90
𝜙v = 0.85
Es = 200000  # Mpa

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


# Cal.critical length
def initial(d):
    xv = FLAGS.B / 2 + FLAGS.bc + d / 100  # Vu critical d from column edge
    x1 = (
        FLAGS.B / 2 - FLAGS.bc / 2 - d / 200
    )  # d/2 from the periphery of the column left, m
    x2 = (
        FLAGS.B / 2 + FLAGS.bc / 2 + d / 200
    )  # d/2 from the periphery of the column right, m
    Ap = (x2 - x1) * (FLAGS.hc + d / 100)  # punching bearing area, m2
    p = 2 * (FLAGS.bc + d / 100) + 2 * (FLAGS.hc + d / 100)  # punching perimeter, m
    xm = FLAGS.B / 2 + FLAGS.bc / 2  # Mu critical at column edge, m
    return xv, x1, x2, Ap, p, xm


# calculate soil capacity depend on footing area
# case axial load only --> qu = uniform load
# with external moment --> qu = trapazoid load


# SLS qu
def q(d, P, M):
    # q = P/A + Mc/I + gamma*V
    I = (FLAGS.B * FLAGS.t**3) / 12
    q1 = (
        P / (FLAGS.B * FLAGS.L)
        - M * (FLAGS.B / 2) / I
        + FLAGS.gamma * FLAGS.B * FLAGS.L * d / 100
    )  # kN/m2
    q2 = (
        P / (FLAGS.B * FLAGS.L)
        + M * (FLAGS.B / 2) / I
        + FLAGS.gamma * FLAGS.B * FLAGS.L * d / 100
    )  # kN/m2
    if q1 < FLAGS.qa and q2 < FLAGS.qa:
        print(
            f"q1 = {q1:.2f} kN/m2, q2 = {q2:.2f} kN/m2 < q_allow = {FLAGS.qa:.2f} kN/m2---> OK"
        )
    else:
        print(
            f"q1 = {q1:.2f} kN/m2, q2 = {q2:.2f} kN/m2 > q_allow = {FLAGS.qa:.2f} kN/m2 ---> NOT OK"
        )


# ULS qu
def qu(d, Mu):
    # q = P/A + Mc/I + gamma*V
    I = (FLAGS.B * FLAGS.t**3) / 12
    qu1 = (
        FLAGS.Pu / (FLAGS.B * FLAGS.L)
        - Mu * (FLAGS.B / 2) / I
        + FLAGS.gamma * FLAGS.B * FLAGS.L * d / 100
    )  # kN/m2
    qu2 = (
        FLAGS.Pu / (FLAGS.B * FLAGS.L)
        + Mu * (FLAGS.B / 2) / I
        + FLAGS.gamma * FLAGS.B * FLAGS.L * d / 100
    )  # kN/m2
    print(f"qu1 = {qu1:.2f} kN/m2, qu2 = {qu2:.2f} kN/m2")
    return qu1, qu2


# create qu equation at x point for any critical case
def ωux(qu1, qu2, x):
    ωux = qu1 + (qu2 - qu1) * x / FLAGS.B  # linear equation
    return ωux  # kN/m2


# check shear
def shear(
    x,
    d,
    qu1,
    qu2,
):
    ωu = ωux(qu1, qu2, x)  # qu at Vu-critical plane, kN/m2
    Vu = (1 / 2) * (ωu + qu2) * (FLAGS.B - x) * FLAGS.L  # kN
    𝜙Vn = (
        𝜙v * (1 / 6) * np.sqrt(FLAGS.fc) * (FLAGS.B - x) * FLAGS.L * 1000
    )  # kN --> f'c = N/mm2, BL = mm2

    #     vn = (Vu/𝜙v)/(1000*(B-x)*L) #N/mm2
    #     v_allow = np.sqrt(fc)/6
    Vu = np.abs(Vu)
    𝜙Vn = np.abs(𝜙Vn)

    if 𝜙Vn > Vu:
        print(f"𝜙Vn = {𝜙Vn:.2f} N/mm2 > Vu ={Vu:.2f} N/mm2 --> Shear capacity OK")
    else:
        print(f"𝜙Vn = {𝜙Vn:.2f} N/mm2 < Vu ={Vu:.2f} N/mm2 --> Shear capacity NOT OK")


# check punching shear
def punching(x1, x2, Ap, p, d, qu1, qu2):
    """x1 --> d/2 from the periphery of the column left, m
    x2 --> d/2 from the periphery of the column right, m
    Ap --> punching bearing area, m2
    p --> punching perimeter, m
    """
    ωu1 = ωux(qu1, qu2, x1)  # qu at center plane, kN/m2
    ωu2 = ωux(qu1, qu2, x2)  # qu at Vu'-critical plane, kN/m2

    Vup = (1 / 2) * (qu1 + qu2) * FLAGS.B * FLAGS.L - (1 / 2) * (ωu2 + ωu1) * Ap  # kN
    𝜙Vp = 𝜙v * 0.25 * np.sqrt(FLAGS.fc) * p * d * 10  # kN --> f'c = N/mm2, pd = mm2

    #     vnp = (Vup/𝜙v)/(p*d)/10 #N/mm2
    #     v_allow = min(1+np.sqrt(fc)/(6*B/L), (2+40*d*p/100)*np.sqrt(fc)/12) #N/mm2
    if 𝜙Vp > Vup:
        print(
            f"𝜙Vp = {𝜙Vp:.2f} N/mm2 > Vup ={Vup:.2f} N/mm2 --> Punching shear capacity OK"
        )
    else:
        print(
            f"𝜙Vp = {𝜙Vp:.2f} N/mm2 < Vup ={Vup:.2f} N/mm2 --> Punching shear capacity NOT OK"
        )

# Calculate moment
def moment(x, qu1, qu2):  # Mu critical at column edge, m
    wu = ωux(qu1, qu2, x)  # qu at Mu-critical plane, kN/m2

    # case axial load only --> qu = uniform load
    if wu == qu1:
        Mu = wu * (FLAGS.B - x) * FLAGS.L * (FLAGS.B - x) / 2
        print(f"Mu = {Mu:.2f} kN-m")
        return np.abs(Mu)

    # case with external moment --> qu = trapazoid load
    else:
        Mu = (wu * (FLAGS.B - x) * FLAGS.L * (FLAGS.B - x) / 2) + (1 / 2) * (
            wu + qu2
        ) * (FLAGS.B - x) * FLAGS.L * (2 / 3) * (FLAGS.B - x)
        print(f"Mu = {Mu:.2f} kN-m")
        return np.abs(Mu)

# Design 
def calculate(d, d1, beam):
    b = FLAGS.B * 100  # cm, define for calling method mainbar_req() in beam.py
    h = FLAGS.t * 100  # cm, define for calling method mainbar_req() in beam.py

    qu1, qu2 = qu(d, FLAGS.Mux)
    print("--------------------------------------------------------------")

    xv, x1, x2, Ap, p, xm = initial(d)

    shear(xv, d, qu1, qu2)  # criticald at d from column edge
    print("--------------------------------------------------------------")

    punching(x1, x2, Ap, p, d, qu1, qu2)  # critical at d/2 from column edge
    print("--------------------------------------------------------------")

    Mu = moment(xm, qu1, qu2)  # critical at column edge, m

    # ----------------------------------------------------------------
    ## Design reinforcements
    beam.initial(FLAGS.main, FLAGS.trav, b, h, FLAGS.L * 100, Mu, Vu=0)

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

# Make report
def call(beam, P, M):
    print("ISOLATED FOOTING DESIGN : USD METHOD")
    print(
        "Code adopt from book: https://www.scribd.com/document/400417851/191061 by รศ.อมร พิมานมาศ และคณะ"
    )
    print(
        "========================================================================================================"
    )

    # TODO
    # Prepare for drawing
    𝜙𝜙 = []
    n = []

    print(f"Footing dimension: B = {FLAGS.B} m, L = {FLAGS.L} m, depth = {FLAGS.t} m")
    print(
        f"Pu = {FLAGS.Pu:.2f} kN/m2, Mux = {FLAGS.Mux:.2f} kN-m, Muy = {FLAGS.Muy:.2f} kN-m"
    )

    print(f"\nCALCULATED")
    # Check qu
    b = FLAGS.B * 100  # cm, define for calling method mainbar_req() in beam.py
    h = FLAGS.t * 100  # cm, define for calling method mainbar_req() in beam.py

    d1 = (
        FLAGS.c + FLAGS.trav / 10 + FLAGS.main / 10 / 2
    )  # Effective depth of Compression Steel
    d = h - d1  # Effective depth of Tension Steel
    q(d, P, M)

    # -----------------------------
    print(f"\nX-X AXIS")
    calculate(d, d1, beam)
    # 𝜙𝜙.append(dia) # for drawing
    # n.append(N) # for drawing

    # -----------------------------
    print(f"\nY-Y AXIS")
    FLAGS.B, FLAGS.L = FLAGS.L, FLAGS.B  # swap variable
    FLAGS.bc, FLAGS.hc = FLAGS.hc, FLAGS.bc  # swap variable
    FLAGS.Mux, FLAGS.Muy = FLAGS.Muy, FLAGS.Mux  # swap variable
    calculate(d, d1, beam)
    # 𝜙𝜙.append(dia) # for drawing
    # n.append(N) # for drawing


# -----------------------------
def main(_argv):
    # Load
    wt = FLAGS.B * FLAGS.L * FLAGS.t * 2400 * 9.8e-3  # self wt., kN
    Pu = FLAGS.Pu + 1.4 * wt  # kN

    P = Pu / 1.4
    M = max(FLAGS.Mux, FLAGS.Muy) / 1.4

    # Instantiate
    beam = Beam(fc=FLAGS.fc, fy=FLAGS.fy, fv=FLAGS.fv, c=FLAGS.c)

    call(beam, P, M)

    # for test flags it's  from parent class(beam.py) or child class()
    # print(f"Test flags 2: {FLAGS.fc}")
    # print(f"Test flags 2: {FLAGS.c}")


if __name__ == "__main__":
    app.run(main)

"""
How to used?
-Please see FLAGS definition for unit informations
-Make sure you are in the project directory run python in terminal(Mac) or command line(Windows)
-run script
    % cd <path to project directory>
    % conda activate <your conda env name>
    % python rc/isolated_footing.py --B=1 --L=1 --t=.35 --bc=.2 --hc=.2 --Pu=11 --Mux=4.5 
    % python rc/isolated_footing.py --B=2.1 --L=1.5 --t=.75 --bc=.4 --hc=.6 --Pu=250 --Mux=10 --Muy=5 --fc=28 --fy=395 --qa=350 --gamma=20
"""
