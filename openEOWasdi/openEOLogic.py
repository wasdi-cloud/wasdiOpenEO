import numpy as np
import xarray as xr
import wasdi

__all__ = [
    "_and",
    "_all",
    "_any",
    "between",
    "eq",
    "gt",
    "gte",
    "_if",
    "is_infinite",
    "is_nan",
    "is_nodata",
    "is_valid",
    "lt",
    "lte",
    "neq",
    "_not",
    "_or",
    "xor"
]

def _and(x, y):
    wasdi.wasdiLog("openEOMLogic.and")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, bool) and isinstance(y, bool):
            return x and y
        else:
            return np.logical_and(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.and: " + str(oError))
        return None

def _all(x):
    wasdi.wasdiLog("openEOMLogic.all")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.all(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.all: " + str(oError))
        return None

def _any(x):
    wasdi.wasdiLog("openEOMLogic.any")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.any(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.any: " + str(oError))
        return None

def between(x, min, max, exclude_max=False):
    wasdi.wasdiLog("openEOMLogic.between")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if exclude_max:
            return np.logical_and(np.greater_equal(x, min), np.less(x, max))
        else:
            return np.logical_and(np.greater_equal(x, min), np.less_equal(x, max))
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.between: " + str(oError))
        return None

def eq(x, y):
    wasdi.wasdiLog("openEOMLogic.eq")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, type(y)):
            return x == y
        else:
            return False

    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.eq: " + str(oError))
        return None

def gt(x, y):
    wasdi.wasdiLog("openEOMLogic.gt")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, str) and isinstance(y,str):
            return x>y
        elif _internalIsNumber(x) and _internalIsNumber(y):
            return float(x)>float(y)
        else:
            return np.greater(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.gt: " + str(oError))
        return None

def gte(x, y):
    wasdi.wasdiLog("openEOMLogic.gte")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, str) and isinstance(y,str):
            return x>=y
        elif _internalIsNumber(x) and _internalIsNumber(y):
            return float(x)>=float(y)
        else:
            return np.greater_equal(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.gte: " + str(oError))
        return None


def _if(value, accept, reject=None):
    wasdi.wasdiLog("openEOMLogic.if")
    try:
        if value:
            return accept
        else:
            return reject

    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic._if: " + str(oError))
        return None

def is_infinite(x):
    wasdi.wasdiLog("openEOMLogic.is_infinite")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.isinf(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.is_infinite: " + str(oError))
        return None

def is_nan(x):
    wasdi.wasdiLog("openEOMLogic.is_nan")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.isnan(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.is_nan: " + str(oError))
        return None

def is_nodata(x):
    wasdi.wasdiLog("openEOMLogic.is_nodata")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return not np.isfinite(x)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.is_nodata: " + str(oError))
        return None

def is_valid(x):
    wasdi.wasdiLog("openEOMLogic.is_valid")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if _internalIsNumber(x):
            return np.isfinite(x)
        return False
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.is_valid: " + str(oError))
        return None

def lt(x, y):
    wasdi.wasdiLog("openEOMLogic.lt")
    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, str) and isinstance(y,str):
            return x < y
        elif _internalIsNumber(x) and _internalIsNumber(y):
            return float(x)<float(y)
        else:
            return np.less(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.lt: " + str(oError))
        return None

def lte(x, y):
    wasdi.wasdiLog("openEOMLogic.lte")
    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, str) and isinstance(y,str):
            return x<=y
        elif _internalIsNumber(x) and _internalIsNumber(y):
            return float(x)<=float(y)
        else:
            return np.less_equal(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.lte: " + str(oError))
        return None

def neq(x, y, delta = None, case_sensitive = None):
    wasdi.wasdiLog("openEOMLogic.neq")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if not isinstance(x,type(y)):
            return False

        if _internalIsNumber(x) and _internalIsNumber(y):
            if delta is None:
                return x!=y
            else:
                fDiff = float(x)-float(y)
                return fDiff>delta
        elif isinstance(x, str) and isinstance(y, str):
            if case_sensitive is True:
                x= x.upper()
                y = y.upper()

            return x!=y

        return np.not_equal(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.neq: " + str(oError))
        return False

def _not(x):
    wasdi.wasdiLog("openEOMLogic._not")

    if x is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if isinstance(x, bool):
            return not x
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic._not: " + str(oError))

    return None

def _or(x, y):
    wasdi.wasdiLog("openEOMLogic._or")

    try:
        if isinstance(x, bool):
            if x:
                return True

        if isinstance(y, bool):
            if y:
                return True

        if x is None or y is None:
            return None
        else:
            return False

    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic._or: " + str(oError))
        return None

def xor(x, y):
    wasdi.wasdiLog("openEOMLogic.xor")

    if x is None or y is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return np.logical_xor(x, y)
    except Exception as oError:
        wasdi.wasdiLog("openEOMLogic.xor: " + str(oError))
        return None

def _internalIsNumber(x):
    if x is None:
        return False
    if isinstance(x, int) or isinstance(x, float):
        return True
    return False