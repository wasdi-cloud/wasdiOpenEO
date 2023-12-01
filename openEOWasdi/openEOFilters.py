from typing import Callable
import wasdi
import numpy as np
import xarray as xr
import pandas as pd
from openeo_pg_parser_networkx.pg_schema import TemporalInterval

__all__ = ["filter_labels", "filter_temporal", "filter_bands","filter_bbox","filter_spatial","mask","resample_spatial","resample_cube_spatial"]


def filter_temporal(data: xr.DataArray, extent: TemporalInterval, dimension: str = None) -> xr.DataArray:
    wasdi.wasdiLog("openFilters.filter_temporal")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        aoNumpyArrayDimension = data.time.data.ravel()
        asStringArray = []

        sStartTime = extent[0]

        if sStartTime is not None:
            if type(sStartTime) is str:
                sStartTime = pd.to_datetime(sStartTime, format='%Y-%m-%d')

            sStartTime = sStartTime.to_numpy()

        sEndTime = extent[1]
        if sEndTime is not None:
            if type(sEndTime) is str:
                sEndTime = pd.to_datetime(sEndTime, format='%Y-%m-%d')

            sEndTime = sEndTime.to_numpy() - np.timedelta64(1, "ms")

        for sDate in aoNumpyArrayDimension:
            if sStartTime <= sDate < sEndTime:
                asStringArray.append(sDate)

        oFilteredDataSet = data.loc[dict(time=slice(sStartTime, sEndTime))]

        return oFilteredDataSet
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.filter_temporal: " + str(oError))
        return data


def filter_labels(data: xr.DataArray, condition: Callable, dimension: str) -> xr.DataArray:
    wasdi.wasdiLog("openFilters.filter_labels")

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        if dimension not in data.dims:
            wasdi.wasdiLog(f"Provided dimension ({dimension}) not found in data.dims: {data.dims}")

        labels = data[dimension].values
        label_mask = condition(x=labels)
        label = labels[label_mask]
        data = data.sel(**{dimension: label})
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.filter_labels: " + str(oError))
        return data

def filter_bands(data: xr.DataArray, bands=None, wavelengths=None):
    if wavelengths is None:
        wavelengths = []
    if bands is None:
        bands = []

    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.filter_bands: " + str(oError))
        return data

def filter_bbox(data: xr.DataArray, extent):
    if data is None or extent is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.filter_bands: " + str(oError))
        return data

def filter_spatial(data: xr.DataArray, geometries):
    if data is None or geometries is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.filter_spatial: " + str(oError))
        return data

def mask(data: xr.DataArray, mask: xr.DataArray, replacement=None):
    if data is None or mask is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.mask: " + str(oError))
        return data

def resample_spatial(data: xr.DataArray, resolution=0, projection=None, method="near", align="upper-left"):
    if data is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.resample_spatial: " + str(oError))
        return data

def resample_cube_spatial(data: xr.DataArray, target: xr.DataArray, method="near"):
    if data is None or target is None:
        wasdi.wasdiLog("One operand is none, result is none")
        return None

    try:
        return data
    except Exception as oError:
        wasdi.wasdiLog("openEOFilters.resample_cube_spatial: " + str(oError))
        return data


filter_labels.implementation = filter_labels
filter_temporal.implementation = filter_temporal
filter_bands.implementation = filter_bands
filter_bbox.implementation = filter_bbox
filter_spatial.implementation = filter_spatial
mask.implementation = mask
resample_spatial.implementation = resample_spatial
resample_cube_spatial.implementation = resample_cube_spatial
