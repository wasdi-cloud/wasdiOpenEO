import numpy as np
import wasdi


__all__ = [
    "e",
    "pi",
    "nan",
    "constant",
    "divide",
    "subtract",
    "multiply",
    "add",
    "_sum",
    "_min",
    "_max",
    "median",
    "mean",
    "sd",
    "variance",
    "floor",
    "ceil",
    "_int",
    "_round",
    "exp",
    "log",
    "ln",
    "cos",
    "arccos",
    "cosh",
    "arcosh",
    "sin",
    "arcsin",
    "sinh",
    "arsinh",
    "tan",
    "arctan",
    "tanh",
    "artanh",
    "arctan2",
    "linear_scale_range",
    "mod",
    "absolute",
    "sgn",
    "sqrt",
    "power",
    "clip",
    "quantiles",
    "product",
    "normalized_difference",
    "ndvi",
]


def e():
    wasdi.wasdiLog("openEOMath.e")
    return np.e

def pi():
    wasdi.wasdiLog("openEOMath.pi")
    return np.pi

def nan():
    wasdi.wasdiLog("openEOMath.nan")
    return np.nan


def constant(x):
    wasdi.wasdiLog("openEOMath.constant")
    return x


def divide(x, y):
    wasdi.wasdiLog("openEOMath.divide")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        result = x / y
        return result
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.divide: " + str(oError))
        return None

def subtract(x, y):
    wasdi.wasdiLog("openEOMath.subtract")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        result = x - y
        return result

    except Exception as oError:
        wasdi.wasdiLog("openEOMath.subtract: " + str(oError))
        return None



def multiply(x, y):
    wasdi.wasdiLog("openEOMath.multiply")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None
    try:
        result = x * y
        return result

    except Exception as oError:
        wasdi.wasdiLog("openEOMath.multiply: " + str(oError))
        return None

def add(x, y):
    wasdi.wasdiLog("openEOMath.add")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        result = x + y
        return result
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.add: " + str(oError))
        return None



def _min(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath._min")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if ignore_nodata:
            return np.nanmin(data, axis=axis, keepdims=keepdims)
        else:
            return np.min(data, axis=axis, keepdims=keepdims)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath._min: " + str(oError))
        return None



def _max(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath._max")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if ignore_nodata:
            return np.nanmax(data, axis=axis, keepdims=keepdims)
        else:
            return np.max(data, axis=axis, keepdims=keepdims)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath._max: " + str(oError))
        return None

def median(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath.median")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if ignore_nodata:
            return np.nanmedian(data, axis=axis, keepdims=keepdims)
        else:
            return np.median(data, axis=axis, keepdims=keepdims)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.median: " + str(oError))
        return None

def mean(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath.mean")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if ignore_nodata:
            return np.nanmean(data, axis=axis, keepdims=keepdims)
        else:
            return np.mean(data, axis=axis, keepdims=keepdims)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.mean: " + str(oError))
        return None

def sd(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath.sd")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if ignore_nodata:
            return np.nanstd(data, axis=axis, ddof=1, keepdims=keepdims)
        else:
            return np.std(data, axis=axis, ddof=1, keepdims=keepdims)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.sd: " + str(oError))
        return None

def variance(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath.variance")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if ignore_nodata:
            return np.nanvar(data, axis=axis, ddof=1, keepdims=keepdims)
        else:
            return np.var(data, axis=axis, ddof=1, keepdims=keepdims)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.variance: " + str(oError))
        return None

def floor(x):
    wasdi.wasdiLog("openEOMath.floor")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None
    try:
        return np.floor(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.floor: " + str(oError))
        return None



def ceil(x):
    wasdi.wasdiLog("openEOMath.ceil")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.ceil(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.ceil: " + str(oError))
        return None

def _int(x):
    wasdi.wasdiLog("openEOMath._int")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.trunc(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath._int: " + str(oError))
        return None

def _round(x, p=0):
    wasdi.wasdiLog("openEOMath._round")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.around(x, decimals=p)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath._round: " + str(oError))
        return None

def exp(p):
    wasdi.wasdiLog("openEOMath.exp")

    if p is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None
    try:
        return np.exp(p)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.exp: " + str(oError))
        return None

def log(x, base):
    wasdi.wasdiLog("openEOMath.log")

    if x is None or base is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.log(x) / np.log(base)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.log: " + str(oError))
        return None

def ln(x):
    wasdi.wasdiLog("openEOMath.ln")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.log(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.ln: " + str(oError))
        return None

def cos(x):
    wasdi.wasdiLog("openEOMath.cos")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.cos(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.cos: " + str(oError))
        return None


def arccos(x):
    wasdi.wasdiLog("openEOMath.arccos")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arccos(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.arccos: " + str(oError))
        return None

def cosh(x):
    wasdi.wasdiLog("openEOMath.cosh")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.cosh(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.and: " + str(oError))
        return None

def arcosh(x):
    wasdi.wasdiLog("openEOMath.arcosh")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arccosh(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.arcosh: " + str(oError))
        return None


def sin(x):
    wasdi.wasdiLog("openEOMath.sin")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.sin(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.sin: " + str(oError))
        return None

def arcsin(x):
    wasdi.wasdiLog("openEOMath.arcsin")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arcsin(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.arcsin: " + str(oError))
        return None

def sinh(x):
    wasdi.wasdiLog("openEOMath.sinh")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.sinh(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.sinh: " + str(oError))
        return None

def arsinh(x):
    wasdi.wasdiLog("openEOMath.arsinh")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arcsinh(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.arsinh: " + str(oError))
        return None

def tan(x):
    wasdi.wasdiLog("openEOMath.tan")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.tan(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.tan: " + str(oError))
        return None

def arctan(x):
    wasdi.wasdiLog("openEOMath.arctan")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arctan(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.arctan: " + str(oError))
        return None


def tanh(x):
    wasdi.wasdiLog("openEOMath.tanh")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.tanh(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.tanh: " + str(oError))
        return None

def artanh(x):
    wasdi.wasdiLog("openEOMath.artanh")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arctanh(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.artanh: " + str(oError))
        return None

def arctan2(y, x):
    wasdi.wasdiLog("openEOMath.arctan2")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.arctan2(y, x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.arctan2: " + str(oError))
        return None

def linear_scale_range(x, inputMin, inputMax, outputMin=0.0, outputMax=1.0):
    wasdi.wasdiLog("openEOMath.linear_scale_range")

    if x is None or inputMin is None or inputMax is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        lsr = ((x - inputMin) / (inputMax - inputMin)) * (outputMax - outputMin) + outputMin
        return lsr
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.linear_scale_range: " + str(oError))
        return None


def mod(x, y):
    wasdi.wasdiLog("openEOMath.mod")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.mod(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.mod: " + str(oError))
        return None


def absolute(x):
    wasdi.wasdiLog("openEOMath.absolute")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.abs(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.absolute: " + str(oError))
        return None


def sgn(x):
    wasdi.wasdiLog("openEOMath.sgn")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.sign(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.sgn: " + str(oError))
        return None


def sqrt(x):
    wasdi.wasdiLog("openEOMath.sqrt")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.sqrt(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.sqrt: " + str(oError))
        return None

def power(base, p):
    wasdi.wasdiLog("openEOMath.power")

    if base is None or p is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        oExp = base ** p
        return oExp
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.power: " + str(oError))
        return None

def clip(x, min, max):
    wasdi.wasdiLog("openEOMath.clip")

    if x is None or min is None or max is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.clip(x, min, max)
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.clip: " + str(oError))
        return None


def quantiles(data, probabilities=None, q=None, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath.quantiles")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(probabilities, list):
            probabilities = np.array(probabilities)

        if q is not None:
            probabilities = np.arange(1.0 / q, 1, 1.0 / q)

        if data.size == 0:
            return np.array([np.nan] * len(probabilities))

        if ignore_nodata:
            result = np.nanquantile(
                data, q=probabilities, method="linear", axis=axis, keepdims=keepdims
            )
        else:
            result = np.quantile(
                data, q=probabilities, method="linear", axis=axis, keepdims=keepdims
            )

        return result
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.quantiles: " + str(oError))
        return None

def _sum(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath._sum")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if len(data) == 0:
            return nan(data=data)

        if ignore_nodata:
            result = np.nansum(data, axis=axis, keepdims=keepdims)
        else:
            result = np.sum(data, axis=axis, keepdims=keepdims)
        return result
    except Exception as oError:
        wasdi.wasdiLog("openEOMath._sum: " + str(oError))
        return None

def product(data, ignore_nodata=True, axis=None, keepdims=False):
    wasdi.wasdiLog("openEOMath.product")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if len(data) == 0:
            return nan(data=data)

        if ignore_nodata:
            result = np.nanprod(data, axis=axis, keepdims=keepdims)
        else:
            result = np.prod(data, axis=axis, keepdims=keepdims)
        return result
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.product: " + str(oError))
        return None


def normalized_difference(x, y):
    wasdi.wasdiLog("openEOMath.normalized_difference")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        nd = (x - y) / (x + y)
        return nd
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.normalized_difference: " + str(oError))
        return None

def ndvi(data, nir="nir", red="red", target_band=None):
    wasdi.wasdiLog("openEOMath.ndvi")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:

        r = np.nan
        n = np.nan
        if "bands" in data.dims:
            if red == "red":
                if "B04" in data["bands"].values:
                    r = data.sel(bands="B04")
            elif red == "rededge":
                if "B05" in data["bands"].values:
                    r = data.sel(bands="B05")
                elif "B06" in data["bands"].values:
                    r = data.sel(bands="B06")
                elif "B07" in data["bands"].values:
                    r = data.sel(bands="B07")
            if nir == "nir":
                n = data.sel(bands="B08")
            elif nir == "nir08":
                if "B8a" in data["bands"].values:
                    n = data.sel(bands="B8a")
                elif "B8A" in data["bands"].values:
                    n = data.sel(bands="B8A")
                elif "B05" in data["bands"].values:
                    n = data.sel(bands="B05")
            elif nir == "nir09":
                if "B09" in data["bands"].values:
                    n = data.sel(bands="B09")
            if red in data["bands"].values:
                r = data.sel(bands=red)
            if nir in data["bands"].values:
                n = data.sel(bands=nir)
        nd = normalized_difference(n, r)
        if target_band is not None:
            nd = nd.assign_coords(bands=target_band)
        # TODO: Remove this once we have the .openeo accessor
        nd.attrs = data.attrs
        return nd
    except Exception as oError:
        wasdi.wasdiLog("openEOMath.ndvi: " + str(oError))
        return None


e.implementation = e
pi.implementation = pi
nan.implementation = nan
constant.implementation = constant
divide.implementation = divide
subtract.implementation = subtract
multiply.implementation = multiply
add.implementation = add
_sum.implementation = _sum
_min.implementation = _min
_max.implementation = _max
median.implementation = median
mean.implementation = mean
sd.implementation = sd
variance.implementation = variance
floor.implementation = floor
ceil.implementation = ceil
_int.implementation = _int
_round.implementation = _round
exp.implementation = exp
log.implementation = log
ln.implementation = ln
cos.implementation = cos
arccos.implementation = arccos
cosh.implementation = cosh
arcosh.implementation = arcosh
sin.implementation = sin
arcsin.implementation = arcsin
sinh.implementation = sinh
arsinh.implementation = arsinh
tan.implementation = tan
arctan.implementation = arctan
tanh.implementation = tanh
artanh.implementation = artanh
arctan2.implementation = arctan2
linear_scale_range.implementation = linear_scale_range
mod.implementation = mod
absolute.implementation = absolute
sgn.implementation = sgn
sqrt.implementation = sqrt
power.implementation = power
clip.implementation = clip
quantiles.implementation = quantiles
product.implementation = product
normalized_difference.implementation = normalized_difference
ndvi.implementation = ndvi
