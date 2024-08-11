# OOP for Beam
import numpy as np

from shear import ShearReinforcement
from rebar import Rebar


class Beam:
    # 1) Initial variables
    def __init__(self, fc, fy, fv, c):
        self.fc = fc  # MPa
        self.fy = fy  # MPa SD30
        self.fv = fv  # Mpa SR24
        self.Es = 2e5  # MPa
        self.c = c  # cm
        self.𝜙b = 0.9
        self.𝜙v = 0.85

        self.rebar = Rebar()

    # 2) β1
    def beta(self):  # β1
        if self.fc <= 30:  # MPa
            self.β1 = 0.85
        elif self.fc > 30 and self.fc < 55:  # MPa
            self.β1 = 0.85 - 0.05 * (self.fc - 30) / 7
        else:
            self.β1 = 0.65

    # 3) Effective depth
    def eff_depth(self):  # mm, mm, cm
        self.d1 = (
            self.c + self.dia_traverse / 10 + self.dia_main / 10 / 2
        )  # Effective depth of Compression Steel
        self.d = self.h - self.d1  # Effective depth of Tension Steel
        print(f"\nEffective Depth : \nd = {self.d:.2f} cm, d' = {self.d1:.2f} cm")
        return self.d, self.d1

    # 4) Percent Reinforcement
    def percent_reinf(self):
        self.pmin = max(np.sqrt(self.fc) / (4 * self.fy), 1.4 / self.fy)
        self.pb = (0.85 * self.fc / self.fy) * self.β1 * (600 / (600 + self.fy))
        self.pmax1 = 0.75 * self.pb
        self.p = 0.50 * self.pb  # conservative!!!
        print(f"\n% Reinforcement : \npmin = {self.pmin:.4f}, pmax = {self.pmax1:.4f}")

    # 5) Section capacity
    def capacity(self):
        As = self.p * self.b * self.d  # cm2
        a = As * self.fy / (0.85 * self.fc * self.b)  # cm.
        self.𝜙Mn1 = self.𝜙b * As * self.fy * (self.d - a / 2) * 1e-3  # kN-m

        print(f"\nSection capacity : \n𝜙Mn = {self.𝜙Mn1:.2f} kN-m")

    # 6) Classification
    def classification(self, Mu):
        𝜙Mn2 = Mu - self.𝜙Mn1  # kN-m
        print(f"\nClassification")
        if 𝜙Mn2 > 0:
            self.section_classification = "double_reinforcement"
            print("double_reinforcement")
            print(f"Mu = {Mu:.2f}, 𝜙Mn1 = {self.𝜙Mn1:.2f}, 𝜙Mn2 = {𝜙Mn2:.2f} :kg.m")

        else:
            self.section_classification = "singly_reinforcement"
            𝜙Mn2 = 0
            print("singly_reinforcement")

    # 7) Singly reinforcement req'd
    def singly_reinf(self, Mu):  # cm, cm, kN-m
        Ru = abs(Mu) * 1000 / (self.b * self.d**2)  # MPa
        p_req = (
            0.85
            * (self.fc / self.fy)
            * (1 - (1 - 2 * (Ru / self.𝜙b) / (0.85 * self.fc)) ** 0.5)
        )

        if p_req > self.pmin:
            As_major = p_req * self.b * self.d  # As_minor = 0
        else:
            As_major = self.pmin * self.b * self.d  # As_minor = 0

        print(
            f"Ru = {Ru:.2f}, p_req = {p_req:.4f}, pmin = {self.pmin:.4f}, pmax = {self.pmax1:.4f}"
        )
        print(f"As_major = {As_major:.2f} cm2, As_minor = 0 cm2")
        return As_major

    # 8) Double reinforcement req'd
    def double_reinf(self, Mu):
        𝜙Mn2 = abs(Mu) - self.𝜙Mn1
        As1 = self.pmax1 * self.b * self.d  # cm2
        As2 = (𝜙Mn2 * 1000 / self.𝜙b) / (self.fy * (self.d - self.d1))  # cm2
        As_major = As1 + As2  # cm2

        p1 = (As1 + As2) / self.b * self.d
        p2 = As2 / self.b * self.d

        if p1 - p2 > 0.85 * self.fc * self.d1 * self.β1 * (600 / (600 - self.fy)) / (
            self.fy * self.d
        ):
            fs = self.fy
            As_minor = As2
            print(f"fs' = {fs:.2f} --> yeild : OK")

            # check p
            if self.pmax1 + p2 > 1.4 / self.fy and self.pmax1 + p2 < p1:
                print("pmin < p < pmax ---> OK")
            else:
                print("p ---< Out of range")
        else:
            fs = 600 * (1 - (self.d1 / self.d) * (600 + self.fy) / 600)
            print(f"fs' = {fs:.2f} --> fs' not yeild ")

            a = self.β1 * self.d * (600 / (600 + self.fy))  # Eq 2.28d
            As_minor = (As1 * self.fy - 0.85 * self.fc * a * self.b) / fs

        print(f"As_major = {As_major:.2f} cm2, As_minor = {As_minor:.2f} cm2")
        return fs, As_major, As_minor

    # 9) Calculated main bar req'd
    def mainbar_req(self, Mu):
        if self.section_classification == "singly_reinforcement":
            As_major = self.singly_reinf(Mu)
            self.data = [As_major]
        else:
            fs, As_major, As_minor = self.double_reinf(Mu)
            self.data = [fs, As_major, As_minor]

    # 10) Calculate traverse spacing
    def traverse_trial(self, Vu):

        dia, As = self.rebar.rebar_selected()

        traverse = ShearReinforcement(self.fc, self.fv, self.fy)

        Av = 2 * As  # cm2

        s_req, s_max = traverse.beamTraverse(self.b, self.d, Av, Vu)

        s = float(
            input(
                f"s_req = {s_req:.2f} cm, s_max = {s_max:.2f} cm, Please select spacing : "
            )
        )
        print(f"Traverse:  ø-{dia} mm @ {s} cm")
        return int(dia), Av, s

    # 11) Design main reinforcements
    def main_trial(self):
        """
        data = return from function main_bar_req()
        if double reinf. --> data = [fs, As_major, As_minor]
        if single reinf. --> data = [As_major]
        """

        while True:
            # Double Reinforcement
            # Decision for compression steel if it's not yeild
            if (len(self.data) != 1) and (
                self.data[0] < self.fy
            ):  # data = [fs, As_major, As_minor]
                ask = input("fs' not yeild --> Break : Y/N : ").upper()
                if ask == "Y":
                    break
                else:
                    pass

            elif (len(self.data) != 1) and (self.data[0] >= self.fy):
                t = ["As_major", "As_minor"]
                for i in range(
                    1, len(self.data) - 1
                ):  # data = [fs, As_major, As_minor]
                    print(
                        f"{t[i-1]} reinf : As-req = {self.data[i]:.2f} cm2 : Please provide As : "
                    )
                    N, dia, As_assign = self.rebar.rebar_design(self.data[i])

            ##Singly reinforcement
            else:
                N, dia, As_assign = self.rebar.rebar_design(self.data[0])

            return N, dia, As_assign

    # 12) Initialize the section properties
    def section_properties(self, dia_main, dia_traverse, b, h):
        self.dia_main = dia_main  # mm
        self.dia_traverse = dia_traverse  # mm
        self.b = b  # cm
        self.h = h  # cm
        self.beta()
        self.percent_reinf()

    # 13) main reinf
    def main_design(self):
        print(f"\n[INFO] Main Reinforcement)")
        while True:
            N, dia, As_assign = self.main_trial()

            ask = input("Try again! : Y/N : ").upper()
            if ask == "Y":
                pass
            else:
                break
        return N, dia, As_assign

    # 14) traverse
    def traverse_design(self, Vu):
        print(f"\n[INFO] Traverse)")
        while True:
            dia, Av, s = self.traverse_trial(self.d, Vu)
            ask = input("Try again! : Y/N : ").upper()
            if ask == "Y":
                pass
            else:
                break
        return dia, Av, s
