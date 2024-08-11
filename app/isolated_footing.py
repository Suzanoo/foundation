"""
ISOLATED FOOTING DESIGN : USD METHOD


"""

import os
import numpy as np
import pandas as pd

from absl import app, flags
from absl.flags import FLAGS

from beam_class import Beam

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

from utils import display_df

##===============================================================================
# CONSTANCE
𝜙b = 0.90
𝜙v = 0.85
Es = 200000  # Mpa

CURRENT = os.getcwd()


# Cal.critical length
def critical_section(d):
    """
    Calculate critical position of Mu, Vu and Vp
    d: eff.depth in cm
    """
    # Vu critical d from column edge, m
    xv = FLAGS.B / 2 + FLAGS.bc + d / 100  # Vu critical d from column edge

    # bottom-left of punching block, m
    x1 = FLAGS.B / 2 - FLAGS.bc / 2 - d / 200

    # top-right of punching block, m
    x2 = FLAGS.B / 2 + FLAGS.bc / 2 + d / 200

    # punching bearing area, m2
    Ap = (x2 - x1) * (FLAGS.hc + d / 100)

    # punching perimeter, m
    p = 2 * (FLAGS.bc + d / 100) + 2 * (FLAGS.hc + d / 100)

    # Mu critical at column edge, m
    xm = FLAGS.B / 2 + FLAGS.bc / 2
    return xv, x1, x2, Ap, p, xm


# SLS qu
def q(d, P, M):
    """
    calculate soil capacity depend on footing area on SLS
    -case axial load only --> qu = uniform load
    -with external moment --> qu = trapazoid load
    d: eff.depth in cm
    q: stress -->  P/A + Mc/I + gamma*V, kN/m2
    """
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
        return True
    else:
        print(
            f"q1 = {q1:.2f} kN/m2, q2 = {q2:.2f} kN/m2 > q_allow = {FLAGS.qa:.2f} kN/m2 ---> NOT OK"
        )
        return False


# ULS qu
def qu(d, Mu):
    """
    calculate soil capacity depend on footing area on ULS
    -case axial load only --> qu = uniform load
    -with external moment --> qu = trapazoid load
    d: eff.depth in cm
    q: stress -->  P/A + Mc/I + gamma*V, kN/m2
    """
    I = (FLAGS.B * FLAGS.t**3) / 12
    qu1 = (
        FLAGS.Pu / (FLAGS.B * FLAGS.L)
        - Mu * (FLAGS.B / 2) / I
        + FLAGS.gamma * FLAGS.B * FLAGS.L * d / 100
    )
    qu2 = (
        FLAGS.Pu / (FLAGS.B * FLAGS.L)
        + Mu * (FLAGS.B / 2) / I
        + FLAGS.gamma * FLAGS.B * FLAGS.L * d / 100
    )
    print(f"qu1 = {qu1:.2f} kN/m2, qu2 = {qu2:.2f} kN/m2")
    return qu1, qu2


# Create qu equation at x point for any critical case
def ωux(qu1, qu2, x):
    ωux = qu1 + (qu2 - qu1) * x / FLAGS.B  # linear equation
    return ωux  # kN/m2


# Check shear
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


# Check punching shear
def punching(x1, x2, Ap, p, d, qu1, qu2):
    """
    x1 --> d/2 from the periphery of the column left, m
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
def calculate_moment_magnitude(d, Mu):
    # Calculate stress magnitude
    qu1, qu2 = qu(d, Mu)

    # Calculate critical section
    xv, x1, x2, Ap, p, xm = critical_section(d)

    # Check beam shear capacity
    shear(xv, d, qu1, qu2)

    # Check punching shear capacity
    punching(x1, x2, Ap, p, d, qu1, qu2)

    # Calculate moment
    Mu = moment(xm, qu1, qu2)
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


# -----------------------------
def main(_argv):
    print(
        "============================== ISOLATED FOOTING DESIGN : USD METHOD =============================="
    )
    print(
        "Credit: https://www.scribd.com/document/400417851/191061 by รศ.อมร พิมานมาศ และคณะ"
    )

    print(f"\n[INFORMATION]")
    print("Footing dimension: B = {FLAGS.B} m, L = {FLAGS.L} m, depth = {FLAGS.t} m")
    print(
        f"Pu = {FLAGS.Pu:.2f} kN/m2, Mux = {FLAGS.Mux:.2f} kN-m, Muy = {FLAGS.Muy:.2f} kN-m"
    )

    # Load
    wt = FLAGS.B * FLAGS.L * FLAGS.t * 2400 * 9.8e-3  # self wt., kN
    Pu = FLAGS.Pu + 1.4 * wt  # kN

    P = Pu / 1.4
    Mmax = max(FLAGS.Mux, FLAGS.Muy) / 1.4

    # Display rebar df
    table = os.path.join(CURRENT, "data/Deform_Bar.csv")
    df = pd.read_csv(table)
    display_df(df)

    print(f"\n==================== X-X AXIS ====================")
    # instanciate
    beam = Beam(fc=FLAGS.fc, fy=FLAGS.fy, fv=FLAGS.fv, c=FLAGS.c)
    beam.section_properties(FLAGS.main, FLAGS.trav, FLAGS.L * 100, FLAGS.t * 100)
    d, d1 = beam.eff_depth()
    beam.capacity()

    # Calculate soil capacity againts loads
    soil_capacity = q(d, P, Mmax)
    if soil_capacity == False:
        return

    # Calculate ultimate moment
    Mu = calculate_moment_magnitude(d, FLAGS.Mux)

    # Design reinf.
    no, main_dia, As_main = reinf_design(Mu, beam)

    print(f"\n==================== Y-Y AXIS ====================")
    FLAGS.B, FLAGS.L = FLAGS.L, FLAGS.B  # swap variable
    FLAGS.bc, FLAGS.hc = FLAGS.hc, FLAGS.bc  # swap variable

    beam.section_properties(FLAGS.main, FLAGS.trav, FLAGS.L * 100, FLAGS.t * 100)
    d, d1 = beam.eff_depth()
    beam.capacity()

    # Calculate soil capacity againts loads
    soil_capacity = q(d, P, Mmax)
    if soil_capacity == False:
        return

    # Calculate ultimate moment
    Mu = calculate_moment_magnitude(d, FLAGS.Muy)

    # Design reinf.
    no, main_dia, As_main = reinf_design(Mu, beam)


if __name__ == "__main__":
    app.run(main)

"""
    % cd <path to project directory>
    % conda activate <your conda env name>
    % python app/isolated_footing.py --B=1.5 --L=1 --t=.35 --bc=.2 --hc=.2 --Pu=11 --Mux=2 
    % python app/isolated_footing.py --B=2.1 --L=1.5 --t=.75 --bc=.4 --hc=.6 --Pu=250 --Mux=10 --Muy=5 --fc=28 --fy=395 --qa=350 --gamma=20
"""
