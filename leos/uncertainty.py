"""
leos.uncertainty
----------------
Defines structured value-error wrappers for handling physical tracking dimensions
with explicit uncertainty boundaries.
"""

class UncertainQuantity:
    def __init__(self, value, uncertainty):
        self.value = value
        self.uncertainty = uncertainty

    def __repr__(self):
        return f"{self.value} \u00b1 {self.uncertainty}"
