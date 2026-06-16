import numpy as np
from astropy import units as u

class UncertainQuantity:
    __array_ufunc__ = None
    def __init__(self, value, uncertainty, unit=None):
        if isinstance(value, u.Quantity):
            self.value = value
            self.uncertainty = uncertainty if isinstance(uncertainty, u.Quantity) else uncertainty * value.unit
        else:
            unit = unit or u.dimensionless_unscaled
            self.value       = value * unit
            self.uncertainty = uncertainty * unit
        self.uncertainty = np.abs(self.uncertainty.to(self.value.unit))
    def _format_rounded(self):
        val_raw = self.value.value
        sig_raw = self.uncertainty.to(self.value.unit).value
        unit_str = f" {self.value.unit}" if self.value.unit != u.dimensionless_unscaled else ""
        if np.isscalar(val_raw):
            if sig_raw == 0 or not np.isfinite(sig_raw): return f"{val_raw}{unit_str} ± {sig_raw}{unit_str}"
            decimals = max(0, int(1 - np.floor(np.log10(sig_raw))))
            return f"{val_raw:.{decimals}f}{unit_str} ± {sig_raw:.{decimals}f}{unit_str}"
        else:
            val_raw, sig_raw = np.asarray(val_raw), np.asarray(sig_raw)
            rounded_vals, rounded_sigs = [], []
            for v, s in zip(val_raw.flat, sig_raw.flat):
                if s == 0 or not np.isfinite(s):
                    rounded_vals.append(str(v)); rounded_sigs.append(str(s))
                else:
                    decimals = max(0, int(1 - np.floor(np.log10(s))))
                    rounded_vals.append(f"{v:.{decimals}f}"); rounded_sigs.append(f"{s:.{decimals}f}")
            return f"{np.array(rounded_vals).reshape(val_raw.shape)}{unit_str} ± {np.array(rounded_sigs).reshape(sig_raw.shape)}{unit_str}"
    def __str__(self): return self._format_rounded()
    def __repr__(self): return f"UncertainQuantity({self._format_rounded()})"
    def relative_uncertainty(self):
        val = self.value.value
        sig = self.uncertainty.to(self.value.unit).value
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_val = np.where(val != 0, sig / np.abs(val), np.inf)
            rel_sig = np.where(np.isfinite(rel_val), rel_val**2, 0.0)
        return UncertainQuantity(rel_val * u.dimensionless_unscaled, rel_sig * u.dimensionless_unscaled)
    def to_unit(self, new_unit): return UncertainQuantity(self.value.to(new_unit), self.uncertainty.to(new_unit))
    def __add__(self, other):
        if isinstance(other, UncertainQuantity): return UncertainQuantity(self.value + other.value, np.sqrt(self.uncertainty**2 + other.uncertainty**2))
        return UncertainQuantity(self.value + other, self.uncertainty)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, UncertainQuantity): return UncertainQuantity(self.value - other.value, np.sqrt(self.uncertainty**2 + other.uncertainty**2))
        return UncertainQuantity(self.value - other, self.uncertainty)
    def __rsub__(self, other): return UncertainQuantity(other - self.value, self.uncertainty)
    def __neg__(self): return UncertainQuantity(-self.value, self.uncertainty)
    def __mul__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value * other.value
            sig = np.abs(val) * np.sqrt(self._safe_rel(self.uncertainty, self.value)**2 + self._safe_rel(other.uncertainty, other.value)**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value * other, self.uncertainty * other)
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value / other.value
            sig = np.abs(val) * np.sqrt(self._safe_rel(self.uncertainty, self.value)**2 + self._safe_rel(other.uncertainty, other.value)**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value / other, self.uncertainty / other)
    def __rtruediv__(self, other):
        val = other / self.value
        return UncertainQuantity(val, np.abs(val) * self._safe_rel(self.uncertainty, self.value))
    @staticmethod
    def _safe_rel(sigma, value):
        v = value.value if isinstance(value, u.Quantity) else value
        s = sigma.value if isinstance(sigma, u.Quantity) else sigma
        with np.errstate(divide="ignore", invalid="ignore"): return np.where(v != 0, s / np.abs(v), 0.0)
    def __pow__(self, n):
        if isinstance(n, UncertainQuantity): raise TypeError("UncertainQuantity ** UncertainQuantity is not supported.")
        return UncertainQuantity(self.value ** n, np.abs(n) * np.abs(self.value) ** (n - 1) * self.uncertainty)
    def sqrt(self): return self ** 0.5
    def exp(self):
        if self.value.unit != u.dimensionless_unscaled: raise ValueError("exp() requires a dimensionless quantity.")
        val = np.exp(self.value.value)
        return UncertainQuantity(val, val * self.uncertainty.value, u.dimensionless_unscaled)
    def log(self):
        if self.value.unit != u.dimensionless_unscaled: raise ValueError("log() requires a dimensionless quantity.")
        x = self.value.value
        if np.any(np.asarray(x) <= 0): raise ValueError("log() requires a positive value.")
        return UncertainQuantity(np.log(x), self.uncertainty.value / np.abs(x), u.dimensionless_unscaled)
    def _validate_and_extract(self, is_inverse=False):
        if is_inverse:
            try: return self.value.to_value(u.dimensionless_unscaled), self.uncertainty.to_value(u.dimensionless_unscaled)
            except: raise ValueError(f"Inverse trig requires dimensionless inputs, not '{self.value.unit}'.")
        else:
            if self.value.unit == u.dimensionless_unscaled:
                return (self.value.value if isinstance(self.value, u.Quantity) else self.value), (self.uncertainty.value if isinstance(self.uncertainty, u.Quantity) else self.uncertainty)
            try: return self.value.to_value(u.rad), self.uncertainty.to_value(u.rad)
            except: raise ValueError(f"Trig requires angular or dimensionless units, not '{self.value.unit}'.")
    def sin(self): x, sig_x = self._validate_and_extract(); return UncertainQuantity(np.sin(x), np.abs(np.cos(x)) * sig_x, u.dimensionless_unscaled)
    def cos(self): x, sig_x = self._validate_and_extract(); return UncertainQuantity(np.cos(x), np.abs(np.sin(x)) * sig_x, u.dimensionless_unscaled)
    def tan(self): x, sig_x = self._validate_and_extract(); return UncertainQuantity(np.tan(x), sig_x / (np.cos(x) ** 2), u.dimensionless_unscaled)
    def sec(self): x, sig_x = self._validate_and_extract(); val = 1.0 / np.cos(x); return UncertainQuantity(val, np.abs(val * np.tan(x)) * sig_x, u.dimensionless_unscaled)
    def csc(self): x, sig_x = self._validate_and_extract(); val = 1.0 / np.sin(x); return UncertainQuantity(val, np.abs(val * (1.0 / np.tan(x))) * sig_x, u.dimensionless_unscaled)
    def cot(self): x, sig_x = self._validate_and_extract(); return UncertainQuantity(1.0 / np.tan(x), sig_x / (np.sin(x) ** 2), u.dimensionless_unscaled)
    def asin(self): x, sig_x = self._validate_and_extract(is_inverse=True); return UncertainQuantity(np.arcsin(x) * u.rad, (sig_x / np.sqrt(1.0 - x**2)) * u.rad)
    def acos(self): x, sig_x = self._validate_and_extract(is_inverse=True); return UncertainQuantity(np.arccos(x) * u.rad, (sig_x / np.sqrt(1.0 - x**2)) * u.rad)
    def atan(self): x, sig_x = self._validate_and_extract(is_inverse=True); return UncertainQuantity(np.arctan(x) * u.rad, (sig_x / (1.0 + x**2)) * u.rad)
    def asec(self): x, sig_x = self._validate_and_extract(is_inverse=True); return UncertainQuantity(np.arccos(1.0 / x) * u.rad, (sig_x / (np.abs(x) * np.sqrt(x**2 - 1.0))) * u.rad)
    def acsc(self): x, sig_x = self._validate_and_extract(is_inverse=True); return UncertainQuantity(np.arcsin(1.0 / x) * u.rad, (sig_x / (np.abs(x) * np.sqrt(x**2 - 1.0))) * u.rad)
    def acot(self): x, sig_x = self._validate_and_extract(is_inverse=True); return UncertainQuantity(np.arctan(1.0 / x) * u.rad, (sig_x / (1.0 + x**2)) * u.rad)
def propagate(func, *args, cov=None, h_rel=1e-6):
    values, sigmas, is_uq = [], [], []
    for a in args:
        if isinstance(a, UncertainQuantity):
            values.append(a.value); sigmas.append(a.uncertainty); is_uq.append(True)
        else:
            values.append(a); sigmas.append(None); is_uq.append(False)
    central = func(*values)
    derivs = []
    for i, (val, flag) in enumerate(zip(values, is_uq)):
        if not flag: continue
        if isinstance(val, u.Quantity): h = np.where(val.value != 0, np.abs(val.value) * h_rel, h_rel) * val.unit
        else: h = np.where(np.asarray(val) != 0, np.abs(np.asarray(val)) * h_rel, h_rel)
        args_plus, args_minus = list(values), list(values)
        args_plus[i] = val + h; args_minus[i] = val - h
        derivs.append((func(*args_plus) - func(*args_minus)) / (2.0 * h))
    if not derivs: return UncertainQuantity(central, central * 0.0)
    first_term = derivs[0] * sigmas[0]
    target_unit = first_term.unit if isinstance(first_term, u.Quantity) else 1
    var_numeric = 0.0
    for deriv, sig in zip(derivs, sigmas):
        if sig is not None:
            d_num = deriv.to_value(target_unit / sig.unit) if isinstance(deriv, u.Quantity) else deriv
            s_num = sig.to_value(sig.unit) if isinstance(sig, u.Quantity) else sig
            var_numeric += (d_num * s_num) ** 2
    if cov is not None:
        cov = np.asarray(cov); num_uq = len(derivs)
        for i in range(num_uq):
            for j in range(i + 1, num_uq):
                if cov[i, j] != 0:
                    d_i = derivs[i].to_value(target_unit / sigmas[i].unit) if isinstance(derivs[i], u.Quantity) else derivs[i]
                    d_j = derivs[j].to_value(target_unit / sigmas[j].unit) if isinstance(derivs[j], u.Quantity) else derivs[j]
                    var_numeric += 2.0 * d_i * d_j * cov[i, j]
    sigma_f = np.sqrt(var_numeric) * target_unit if target_unit != 1 else np.sqrt(var_numeric)
    return UncertainQuantity(central, sigma_f)
