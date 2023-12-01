from typing import Callable, Optional, Union
import wasdi
import xarray as xr
import numpy as np
from collections import namedtuple
import rasterio
from openeo_pg_parser_networkx.pg_schema import TemporalInterval, TemporalIntervals

__all__ = ["apply", "apply_dimension", "create_raster_cube", "drop_dimension", "dimension_labels", "merge_cubes", "aggregate_temporal", "aggregate_temporal_period", "rename_labels", "reduce_dimension"]


def apply(
    data: xr.DataArray, process: Callable, context: Optional[dict] = None
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.apply")

    positional_parameters = {"x": 0}
    named_parameters = {"context": context}
    result = xr.apply_ufunc(
        process,
        data,
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
        },
    )
    return result

def apply_dimension(
    data: xr.DataArray,
    process: Callable,
    dimension: str,
    target_dimension: Optional[str] = None,
    context: Optional[dict] = None,
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.apply_dimension")

    if context is None:
        context = {}

    dimension = translateDimension(dimension)

    if dimension not in data.dims:
        wasdi.wasdiLog("Provided dimension not found in data.dims")
        return None

    if target_dimension is None:
        target_dimension = dimension

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    result = xr.apply_ufunc(
        process,
        reordered_data,
        input_core_dims=[[dimension]],
        output_core_dims=[[dimension]],
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
            "axis": reordered_data.get_axis_num(dimension),
            "keepdims": True,
        },
        exclude_dims={dimension},
    )

    reordered_result = result.transpose(*data.dims, ...).rename(
        {dimension: target_dimension}
    )

    if len(data[dimension]) == len(reordered_result[target_dimension]):
        reordered_result.rio.write_crs(data.rio.crs, inplace=True)

    return reordered_result


def drop_dimension(data: xr.DataArray, name: str) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.drop_dimension")
    name = translateDimension(name)

    if name not in data.dims:
        wasdi.wasdiLog(f"Provided dimension ({name}) not found in data.dims: {data.dims}")

    if len(data[name]) > 1:
        wasdi.wasdiLog(f"The number of dimension labels exceeds one, which requires a reducer. Dimension ({name}) has {len(data[name])} labels.")

    return data.drop(name)


def create_raster_cube() -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.create_raster_cube")
    return xr.DataArray()


def dimension_labels(data: xr.DataArray, dimension: str) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.dimension_labels")

    dimension = translateDimension(dimension)

    if dimension not in data.dims:
        wasdi.wasdiLog(f"Provided dimension ({dimension}) not found in data.dims: {data.dims}")

    return data.coords[dimension]

Overlap = namedtuple("Overlap", ["only_in_cube1", "only_in_cube2", "in_both"])
NEW_DIM_NAME = "__cubes__"
NEW_DIM_COORDS = ["cube1", "cube2"]

def merge_cubes(
    cube1: xr.DataArray,
    cube2: xr.DataArray,
    overlap_resolver: Callable = None,
    context: Optional[dict] = None,
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.merge_cubes")

    if context is None:
        context = {}
    if not isinstance(cube1, type(cube2)):
        raise Exception(
            f"Provided cubes have incompatible types. cube1: {type(cube1)}, cube2: {type(cube2)}"
        )

    # Key: dimension name
    # Value: (labels in cube1 not in cube2, labels in cube2 not in cube1)
    overlap_per_shared_dim = {
        dim: Overlap(
            only_in_cube1=np.setdiff1d(cube1[dim].data, cube2[dim].data),
            only_in_cube2=np.setdiff1d(cube2[dim].data, cube1[dim].data),
            in_both=np.intersect1d(cube1[dim].data, cube2[dim].data),
        )
        for dim in set(cube1.dims).intersection(set(cube2.dims))
    }

    differing_dims = set(cube1.dims).symmetric_difference(set(cube2.dims))

    if len(differing_dims) == 0:
        # Check whether all of the shared dims have exactly the same labels
        dims_have_no_label_diff = all(
            [
                len(overlap.only_in_cube1) == 0 and len(overlap.only_in_cube2) == 0
                for overlap in overlap_per_shared_dim.values()
            ]
        )
        if dims_have_no_label_diff:
            # Example 3: All dimensions and their labels are equal
            concat_both_cubes = xr.concat([cube1, cube2], dim=NEW_DIM_NAME).reindex(
                {NEW_DIM_NAME: NEW_DIM_COORDS}
            )

            # Need to rechunk here to ensure that the cube dimension isn't chunked and the chunks for the other dimensions are not too large.
            concat_both_cubes_rechunked = concat_both_cubes.chunk(
                {NEW_DIM_NAME: -1}
                | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
            )
            if overlap_resolver is None:
                # Example 3.1: Concat along new "cubes" dimension
                merged_cube = concat_both_cubes_rechunked
            else:
                # Example 3.2: Elementwise operation
                positional_parameters = {"data": 0}
                named_parameters = {"context": context}

                merged_cube = concat_both_cubes_rechunked.reduce(
                    overlap_resolver,
                    dim=NEW_DIM_NAME,
                    keep_attrs=True,
                    positional_parameters=positional_parameters,
                    named_parameters=named_parameters,
                )
        else:
            # Example 1 & 2
            dims_requiring_resolve = [
                dim
                for dim, overlap in overlap_per_shared_dim.items()
                if len(overlap.in_both) > 0
                and (len(overlap.only_in_cube1) > 0 or len(overlap.only_in_cube2) > 0)
            ]

            if len(dims_requiring_resolve) == 0:
                # Example 1: No overlap on any dimensions, can just combine by coords
                merged_cube = xr.combine_by_coords([cube1, cube2], combine_attrs='override')
            elif len(dims_requiring_resolve) == 1:
                # Example 2: Overlap on one dimension, resolve these pixels with overlap resolver
                # and combine the rest by coords

                if overlap_resolver is None or not callable(overlap_resolver):
                    wasdi.wasdiLog("Overlapping data cubes, but no overlap resolver has been specified.")

                overlapping_dim = dims_requiring_resolve[0]

                stacked_conflicts = xr.concat(
                    [
                        cube1.sel(
                            **{
                                overlapping_dim: overlap_per_shared_dim[
                                    overlapping_dim
                                ].in_both
                            }
                        ),
                        cube2.sel(
                            **{
                                overlapping_dim: overlap_per_shared_dim[
                                    overlapping_dim
                                ].in_both
                            }
                        ),
                    ],
                    dim=NEW_DIM_NAME,
                ).reindex({NEW_DIM_NAME: NEW_DIM_COORDS})

                # Need to rechunk here to ensure that the cube dimension isn't chunked and the chunks for the other dimensions are not too large.
                stacked_conflicts_rechunked = stacked_conflicts.chunk(
                    {NEW_DIM_NAME: -1}
                    | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
                )

                positional_parameters = {"data": 0}
                named_parameters = {"context": context}

                merge_conflicts = stacked_conflicts_rechunked.reduce(
                    overlap_resolver,
                    dim=NEW_DIM_NAME,
                    keep_attrs=True,
                    positional_parameters=positional_parameters,
                    named_parameters=named_parameters,
                )

                rest_of_cube_1 = cube1.sel(
                    **{
                        overlapping_dim: overlap_per_shared_dim[
                            overlapping_dim
                        ].only_in_cube1
                    }
                )
                rest_of_cube_2 = cube2.sel(
                    **{
                        overlapping_dim: overlap_per_shared_dim[
                            overlapping_dim
                        ].only_in_cube2
                    }
                )
                merged_cube = xr.combine_by_coords(
                    [merge_conflicts, rest_of_cube_1, rest_of_cube_2]
                )

            else:
                raise ValueError(
                    "More than one overlapping dimension, merge not possible."
                )

    elif len(differing_dims) <= 2:
        if overlap_resolver is None or not callable(overlap_resolver):
            wasdi.wasdiLog("Overlapping data cubes, but no overlap resolver has been specified.")

        # Example 4: broadcast lower dimension cube to higher-dimension cube
        lower_dim_cube = cube1 if len(cube1.dims) < len(cube2.dims) else cube2
        higher_dim_cube = cube1 if len(cube1.dims) >= len(cube2.dims) else cube2
        lower_dim_cube_broadcast = lower_dim_cube.broadcast_like(higher_dim_cube)

        # Stack both cubes and use overlap resolver to resolve each pixel
        both_stacked = xr.concat(
            [higher_dim_cube, lower_dim_cube_broadcast], dim=NEW_DIM_NAME
        ).reindex({NEW_DIM_NAME: NEW_DIM_COORDS})

        # Need to rechunk here to ensure that the cube dimension isn't chunked and the chunks for the other dimensions are not too large.
        both_stacked_rechunked = both_stacked.chunk(
            {NEW_DIM_NAME: -1}
            | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
        )

        positional_parameters = {"data": 0}
        named_parameters = {"context": context}
        merged_cube = both_stacked_rechunked.reduce(
            overlap_resolver,
            dim=NEW_DIM_NAME,
            keep_attrs=True,
            positional_parameters=positional_parameters,
            named_parameters=named_parameters,
        )
    else:
        raise ValueError("Number of differing dimensions is >2, merge not possible.")

    return merged_cube

def translateDimension(dimension: str) -> str:
    if dimension == 't':
        dimension = "time"
    elif dimension == "bands":
        dimension = "band"

    return dimension
def reduce_dimension(
    data: xr.DataArray,
    reducer: Callable,
    dimension: str,
    context: Optional[dict] = None,
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubes.reduce_dimension")
    dimension = translateDimension(dimension)

    if dimension not in data.dims:
        wasdi.wasdiLog(f"Provided dimension ({dimension}) not found in data.dims: {data.dims}")

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    oNewDataset = data

    if type(data) is xr.Dataset:
        oVariable = data.variables["spatial_ref"]
        oNewDataset = data.drop("spatial_ref")

    reduced_data = oNewDataset.reduce(
        reducer,
        dim=dimension,
        keep_attrs=True,
        numeric_only=True,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )

    if type(data) is xr.Dataset:
        reduced_data = reduced_data.assign({"spatial_ref": oVariable})

    return reduced_data


def reduce_spatial(
    data: xr.DataArray, reducer: Callable, context: Optional[dict] = None
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.reduceSpatial")
    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    spatial_dims = data.openeo.spatial_dims if data.openeo.spatial_dims else None
    return data.reduce(
        reducer,
        dimension=spatial_dims,
        keep_attrs=True,
        context=context,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )


def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    wasdi.wasdiLog("openEOCubs.geometry_mask")
    return rasterio.features.geometry_mask(
        [geom.to_crs(geobox.crs) for geom in geoms],
        out_shape=geobox.shape,
        transform=geobox.affine,
        all_touched=all_touched,
        invert=invert,
    )


def aggregate_temporal(
    data: xr.DataArray,
    intervals: Union[TemporalIntervals, list[TemporalInterval], list[Optional[str]]],
    reducer: Callable,
    labels: Optional[list] = None,
    dimension: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.aggregate_temporal")
    temporal_dims = "time"

    if dimension is not None:
        if dimension not in data.dims:
            wasdi.wasdiLog(f"A dimension with the specified name: {dimension} does not exist.")
        applicable_temporal_dimension = dimension
    else:
        if not temporal_dims:
            wasdi.wasdiLog(f"No temporal dimension detected on dataset. Available dimensions: {data.dims}")
        if len(temporal_dims) > 1:
            wasdi.wasdiLog(f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified.")
        applicable_temporal_dimension = temporal_dims[0]

    aggregated_data = data.groupby_bins(
        group=applicable_temporal_dimension, labels=labels
    )

    raise NotImplementedError("aggregate_temporal is currently not implemented")


def aggregate_temporal_period(
    data: xr.DataArray,
    reducer: Callable,
    period: str,
    dimension: Optional[str] = None,
    **kwargs,
) -> xr.DataArray:
    wasdi.wasdiLog("openEOCubs.aggregate_temporal_period")
    temporal_dims = "time"

    if dimension is not None:
        if dimension not in data.dims:
            wasdi.wasdiLog(f"A dimension with the specified name: {dimension} does not exist.")
        applicable_temporal_dimension = dimension
    else:
        if not temporal_dims:
            wasdi.wasdiLog(f"No temporal dimension detected on dataset. Available dimensions: {data.dims}")
        if len(temporal_dims) > 1:
            wasdi.wasdiLog(f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified.")
        applicable_temporal_dimension = temporal_dims[0]

    periods_to_frequency = {
        "hour": "H",
        "day": "D",
        "week": "W",
        "month": "M",
        "season": "QS-DEC",
        "year": "AS",
    }

    if period in periods_to_frequency.keys():
        frequency = periods_to_frequency[period]
    else:
        wasdi.wasdiLog(f"The provided period '{period})' is not implemented yet. The available ones are {list(periods_to_frequency.keys())}.")

    resampled_data = data.resample({applicable_temporal_dimension: frequency})

    return resampled_data.reduce(reducer, keep_attrs=True)

def rename_labels(data: xr.DataArray, dimension: str, target):
    wasdi.wasdiLog("openEOCubs.rename_labels")
    dimension = translateDimension(dimension)
    data[dimension] = (dimension, target)

    return data


apply.implementation = apply
apply_dimension.implementation = apply_dimension
create_raster_cube.implementation = create_raster_cube
drop_dimension.implementation = drop_dimension
dimension_labels.implementation = dimension_labels
merge_cubes.implementation = merge_cubes
aggregate_temporal.implementation = aggregate_temporal
aggregate_temporal_period.implementation = aggregate_temporal_period
rename_labels.implementation = rename_labels
reduce_dimension.implementation = reduce_dimension
