class SoilLayer:
    """
    Represents a single layer of soil.

    Attributes:
        thickness (float): Thickness of the soil layer in meters.
        initial_void_ratio (float): Initial void ratio of the soil.
        compression_index (float): Compression index (C_c), unitless.
        secondary_compression_index (float): Secondary compression index (C_alpha), unitless.
        initial_stress (float): Initial effective stress in the soil layer in kPa.
        modulus_of_elasticity (float): Modulus of elasticity (E_s) of the soil in kPa.
        poisson_ratio (float): Poisson's ratio (ν), unitless.
    """

    def __init__(
        self,
        thickness,
        initial_void_ratio,
        compression_index,
        secondary_compression_index,
        initial_stress,
        modulus_of_elasticity,
        poisson_ratio,
    ):
        self.thickness = thickness
        self.initial_void_ratio = initial_void_ratio
        self.compression_index = compression_index
        self.secondary_compression_index = secondary_compression_index
        self.initial_stress = initial_stress
        self.modulus_of_elasticity = modulus_of_elasticity
        self.poisson_ratio = poisson_ratio

    def __str__(self):
        STR = f"""
            Soil thickness: {self.thickness} m
            Initial void ratio: {self.initial_void_ratio}
            Compression Index: {self.compression_index}
            Secondary Compression Index: {self.secondary_compression_index}
            Initial Stress: {self.initial_stress} kPa
            Modulus of Elasticity: {self.modulus_of_elasticity} kPa
            Poisson Ratio: {self.poisson_ratio}
        """
        return STR


class Load:
    """
    Represents the load applied to the embankment.

    Attributes:
        magnitude (float): Magnitude of the load in kPa.
        width (float): Width of the load area in meters.
        length (float): Length of the load area in meters.
    """

    def __init__(self, magnitude, width, length):
        self.magnitude = magnitude
        self.width = width
        self.length = length

    def __str__(self) -> str:
        STR = f"""
            Load magnitude: {self.magnitude} kPa
            Width: {self.width} m
            Length: {self.length} m
        """
        return STR


class SettlementCalculator:
    """
    Calculates the settlement of an embankment based on soil layers and applied load.

    Attributes:
        soil_layers (list of SoilLayer): List of SoilLayer objects representing the soil profile.
        load (Load): Load object representing the applied load on the embankment.
    """

    def __init__(self, soil_layers, load):
        self.soil_layers = soil_layers
        self.load = load

    def calculate_immediate_settlement(self):
        """
        Elastic Settlement (Immediate or Short-Term):

        Terzaghi’s Elastic Settlement Formula
        Si = qB(1-ν^2)Es/Is

        Si: Immediate settlement
        q: Applied pressure
        ν: Poisson's ratio
        Es: Modulus of elasticity of soil
        Is: nfluence factor, which depends on the shape of the loaded area.
        """
        settlement = 0  # m
        for layer in self.soil_layers:
            I_s = 1.0  # Influence factor, typically determined based on the geometry of the load
            settlement += (
                (self.load.magnitude * self.load.width * (1 - layer.poisson_ratio**2))
                / (layer.modulus_of_elasticity)
                * I_s
            )

        print(
            f"Elastic Settlement (Immediate or Short-Term) : Si = qB(1-ν^2)Es/Is : = {settlement:.2f} m"
        )

        return settlement

    def calculate_primary_consolidation_settlement(self):
        """
        Primary Settlement (Time-Dependent)

        Terzaghi’s Consolidation Equation:
        Sc = Cc * H * log((σ'0 + Δσ') / σ'0) / (1 + e0)

        Sc: Consolidation settlement
        Cc: Compression index (from lab tests)
        H: Thickness of the compressible soil layer
        σ'0: Initial effective stress
        Δσ': Increase in effective stress
        e0: Initial void ratio
        """
        settlement = 0  # m
        for layer in self.soil_layers:
            delta_stress = self.load.magnitude / self.load.width
            settlement += (
                layer.compression_index
                * layer.thickness
                * (
                    np.log10(
                        (layer.initial_stress + delta_stress) / layer.initial_stress
                    )
                )
            ) / (1 + layer.initial_void_ratio)

        print(
            f"Primary Settlement (Time-Dependent) : Sc = Cc * H * log((σ'0 + Δσ') / σ'0) / (1 + e0) = {settlement:.2f} m"
        )

        return settlement

    def calculate_secondary_settlement(self, time, time_primary):
        """
        Secondary Settlement (Creep)
        Args:
            time (float): Time elapsed since the load application in days.
            time_primary (float): Time corresponding to the end of primary consolidation in days.

        Returns:
            float: Secondary settlement in meters.

        Ss = Cα * H * log( t / tp ) / ( 1 + ep)

        Cα: Secondary compression index
        t: Time elapsed
        tp: Time corresponding to end of primary consolidation
        ep: Void ratio at tp
        """
        settlement = 0
        for layer in self.soil_layers:
            settlement += (
                layer.secondary_compression_index
                * layer.thickness
                * np.log10(time / time_primary)
            ) / (1 + layer.initial_void_ratio)

        print(
            f"Secondary Settlement (Creep) : Ss = Cα * H * log( t / tp ) / ( 1 + ep) = {settlement:.2f} m"
        )

        return settlement

    def total_settlement(self, time, time_primary):
        """
        Calculates the total settlement of the embankment, including immediate, primary, and secondary settlement.

        Args:
            time (float): Time elapsed since the load application in days.
            time_primary (float): Time corresponding to the end of primary consolidation in days.

        Returns:
            float: Total settlement in meters.
        """
        immediate = self.calculate_immediate_settlement()
        primary = self.calculate_primary_consolidation_settlement()
        secondary = self.calculate_secondary_settlement(time, time_primary)
        return immediate + primary + secondary


class Embankment:
    def __init__(self, name, soil_layers, load):
        """
        Represents an embankment and handles the prediction of settlement.

        Attributes:
            name (str): Name of the embankment.
            settlement_calculator (SettlementCalculator): Calculator for predicting settlement.
        """
        self.name = name
        self.settlement_calculator = SettlementCalculator(soil_layers, load)

    def predict_settlement(self, time, time_primary):
        """
        Predicts the total settlement of the embankment over time.

        Args:
            time (float): Time elapsed since the load application in days.
            time_primary (float): Time corresponding to the end of primary consolidation in days.

        Returns:
            float: Predicted total settlement in meters.
        """
        return self.settlement_calculator.total_settlement(time, time_primary)


# ----------------------------------------------------------------
import numpy as np

# Define soil layers
layer1 = SoilLayer(
    thickness=1.5,
    initial_void_ratio=0.8,
    compression_index=0.3,
    secondary_compression_index=0.05,
    initial_stress=100,
    modulus_of_elasticity=10000,
    poisson_ratio=0.3,
)
layer2 = SoilLayer(
    thickness=1,
    initial_void_ratio=0.7,
    compression_index=0.25,
    secondary_compression_index=0.04,
    initial_stress=120,
    modulus_of_elasticity=8000,
    poisson_ratio=0.35,
)

print("[INFO] : Embankment Layers")
print(layer1)
print(layer2)

# Define load
load = Load(magnitude=20, width=60, length=200)
print("[INFO] : Loads")
print(load)

# Create embankment
embankment = Embankment(
    name="Example Embankment", soil_layers=[layer1, layer2], load=load
)

# Predict settlement
t = 2000  #
tp = 100
print("[INFO] : Predict settlement")
predicted_settlement = embankment.predict_settlement(time=t, time_primary=tp)
print(
    f"\nTime elapsed since the load application = {t} days, \nTime corresponding to the end of primary consolidation = {tp} days"
)
print(f"Predicted Settlement : {predicted_settlement:.2f} m")


"""
python app/settlement.py
"""
