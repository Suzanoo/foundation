#!/usr/bin/env python3

import os
import json

from rc.utils import Utils

ut = Utils()

def rectangle(w, h, CoG, color):
    x = CoG['x']
    y = CoG['y']
    x0 = x - w/2
    x1 = x+w/2
    y0 = y - h/2
    y1 = y+h/2
    return {
        "type": "rect",
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "line": {
            "color": color,
            "width": 2,
        },
    }

def load_in_pile(
    P, Mx0, My0, N, coordinates, ex, ey, B, L, t, h, gamma, origin, Qsa, designFS
):
    """
    Calculates the load in each pile and the factor of safety of each pile.

    :param P: Load (kN)
    :param Mx0: Initial moment around X-axis (kN-m)
    :param My0: Initial moment around Y-axis (kN-m)
    :param N: Number of piles
    :param coordinates: List of coordinates of piles
    :param ex: Deviation along X-X axis (m)
    :param ey: Deviation along Y-Y axis (m)
    :param B: Pilecap width (m)
    :param L: Pilecap length (m)
    :param t: Pilecap depth (m)
    :param h: Height of overburden (m)
    :param gamma: Soil density (t/m3)
    :param Origin: Origin coordinates
    :param Qsa: Allowable load capacity of a single pile (kN)
    :param designFS: Design factor of safety
    :return: Tuple containing the list of loads in each pile, self weight, overburden and factor of safety of each pile
    """


    # Calculate Mx and My come from eccentricity
    Mx = Mx0 + P * ey * 1e-3  # kN-m
    My = My0 + P * ex * 1e-3  # kN-m


    # Extract x, y, newX, newY and calculate xi, yi for each coordinate
    processed_coords = [
        {
            "x": float(coord.get("x")),
            "y": float(coord.get("y")),
            "newX": float(coord.get("newX", coord.get("x"))),
            "newY": float(coord.get("newY", coord.get("y"))),
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

    m = (My * Ix - Mx * Ixy) / (Ix * Iy - Ixy**2)
    n = (Mx * Iy - My * Ixy) / (Ix * Iy - Ixy**2)


    # Self weight and Overburden
    self_weight = -(B * L * t * 2400 * 9.81e-3)  # kN
    overburden = -(B * L * h * gamma * 9.81)  # kN


    # Load in each pile
    Ri = [
        (-P + self_weight + overburden) / N + m * coord["xi"] + n * coord["yi"]
        for coord in processed_coords
    ]

    # Factor of safety of each pile
    FS = [abs((designFS * Qsa) / ri) for ri in Ri]

    return Ri, self_weight, overburden, FS


## ----------------------------------------------------------------
## Input
cwd = os.getcwd()
file1 = os.path.join(cwd, "data/pileInfo.json")
file2 = os.path.join(cwd, "data/loadInfo.json")

# Open the file for reading
with open(file1, "r") as file:
    # Load JSON content from the file
    pileInfo = json.load(file)

with open(file2, "r") as file:
    # Load JSON content from the file
    loadInfo = json.load(file)

## Capture inputs
P = loadInfo["P"]  # kN
Mx0 = loadInfo["Mx0"]  # kN-m
My0 = loadInfo["My0"]  # kN-m
h = loadInfo["overBurdent"]  # m
gamma = loadInfo["gamma"]  # kN/m3
Qsa = loadInfo["Qsa"] * 9.81e-3  # kN
designFS = loadInfo["FS"]


N = pileInfo["pileQuantities"]
coordinates = pileInfo["coordinates"]
B = pileInfo["footingWidth"] / 1000  # m
L = pileInfo["footingLength"] / 1000  # m
t = pileInfo["footingDepth"] / 1000  # m

W = pileInfo["columnWidth"] / 1000
D = pileInfo["columnLength"] / 1000

origin = pileInfo["origin"]  # (x0, y0)

# calculate new CoG
weights = [1 for coord in coordinates]
new_CoG = ut.calculate_center_of_gravity(coordinates, weights)

# Eccentricity
ex = new_CoG["x"] - origin["x"]
ey = new_CoG["y"] - origin["y"]

# Load in piles
Ri, self_weight, overburden, FS = load_in_pile(
    P, Mx0, My0, N, coordinates, ex, ey, B, L, t, h, gamma, origin, Qsa, designFS
)

for i in range(len(Ri)):
    print(f"Load in R-{i+1} : {Ri[i]:.2f} kN")

# ----------------------------------------------------------------
'''
TODO
- display pilecap
- calculate FS

'''
footing_shape = rectangle(B, L, origin, "blue")
column_shape = rectangle(W, D, origin, "orange")


# ----------------------------------------------------------------
# ~ python rc/pile_deviation.py
