import wasdi
from openEOWasdi.openEOCubes import *
from openEOWasdi.openEOCollections import *
from openEOWasdi.openEOFilters import *
from openEOWasdi.openEOMath import *
from openEOWasdi.openEOLogic import *
import json
from openeo_pg_parser_networkx import OpenEOProcessGraph
from openeo_pg_parser_networkx import ProcessRegistry
import process
import urllib.parse
from osgeo import gdal

def run():
    wasdi.wasdiLog('WASDI openEO App v.1.0.5')
    gdal.UseExceptions()

    aoPayload = {"inputs": wasdi.getParametersDict()}

    try:
        wasdi.wasdiLog("Prepare the OpenEO Graph")
        #wasdi.wasdiLog(aoPayload)

        sJsonText = json.dumps(wasdi.getParametersDict()["process"])

        try:
            oParsedGraph = OpenEOProcessGraph.from_json(sJsonText)
        except:
            wasdi.wasdiLog("The Graph is not a JSON, try to decode it")

            #wasdi.wasdiLog("INPUT process:")
            #wasdi.wasdiLog(wasdi.getParametersDict()["process_encoded"])

            sDecodedJson = urllib.parse.unquote(wasdi.getParametersDict()["process_encoded"])
            oParsedGraph = OpenEOProcessGraph.from_json(sDecodedJson)


        wasdi.wasdiLog("OpenEO Graph parsed")

        aoProcRegistry = ProcessRegistry(wrap_funcs=[process.process])
        aoProcRegistry["apply"] = apply
        aoProcRegistry["apply_dimension"] = apply_dimension
        aoProcRegistry["create_raster_cube"] = create_raster_cube
        aoProcRegistry["drop_dimension"] = drop_dimension
        aoProcRegistry["dimension_labels"] = dimension_labels
        aoProcRegistry["merge_cubes"] = merge_cubes
        aoProcRegistry["aggregate_temporal"] = aggregate_temporal
        aoProcRegistry["aggregate_temporal_period"] = aggregate_temporal_period
        aoProcRegistry["rename_labels"] = rename_labels
        aoProcRegistry["reduce_dimension"] = reduce_dimension

        aoProcRegistry["filter_temporal"] = filter_temporal
        aoProcRegistry["filter_labels"] = filter_labels

        aoProcRegistry["filter_bands"] = filter_bands
        aoProcRegistry["filter_bbox"] = filter_bbox
        aoProcRegistry["filter_spatial"] = filter_spatial
        aoProcRegistry["mask"] = mask
        aoProcRegistry["resample_spatial"] = resample_spatial
        aoProcRegistry["resample_cube_spatial"] = resample_cube_spatial

        aoProcRegistry["e"] = e
        aoProcRegistry["pi"] = pi
        aoProcRegistry["nan"] = nan
        aoProcRegistry["constant"] = constant
        aoProcRegistry["divide"] = divide
        aoProcRegistry["subtract"] = subtract
        aoProcRegistry["multiply"] = multiply
        aoProcRegistry["add"] = add
        aoProcRegistry["sum"] = _sum
        aoProcRegistry["min"] = _min
        aoProcRegistry["max"] = _max
        aoProcRegistry["median"] = median
        aoProcRegistry["mean"] = mean
        aoProcRegistry["sd"] = sd
        aoProcRegistry["variance"] = variance
        aoProcRegistry["floor"] = floor
        aoProcRegistry["ceil"] = ceil
        aoProcRegistry["int"] = _int
        aoProcRegistry["round"] = _round
        aoProcRegistry["exp"] = exp
        aoProcRegistry["log"] = log
        aoProcRegistry["ln"] = ln
        aoProcRegistry["cos"] = cos
        aoProcRegistry["arccos"] = arccos
        aoProcRegistry["cosh"] = cosh
        aoProcRegistry["arcosh"] = arcosh
        aoProcRegistry["sin"] = sin
        aoProcRegistry["arcsin"] = arcsin
        aoProcRegistry["sinh"] = sinh
        aoProcRegistry["arsinh"] = arsinh
        aoProcRegistry["tan"] = tan
        aoProcRegistry["arctan"] = arctan
        aoProcRegistry["tanh"] = tanh
        aoProcRegistry["artanh"] = artanh
        aoProcRegistry["arctan2"] = arctan2
        aoProcRegistry["linear_scale_range"] = linear_scale_range
        aoProcRegistry["mod"] = mod
        aoProcRegistry["absolute"] = absolute
        aoProcRegistry["sgn"] = sgn
        aoProcRegistry["sqrt"] = sqrt
        aoProcRegistry["power"] = power
        aoProcRegistry["clip"] = clip
        aoProcRegistry["quantiles"] = quantiles
        aoProcRegistry["product"] = product
        aoProcRegistry["normalized_difference"] = normalized_difference
        aoProcRegistry["ndvi"] = ndvi

        aoProcRegistry["and"] = _and
        aoProcRegistry["all"] = _all
        aoProcRegistry["any"] = _any
        aoProcRegistry["between"] = between
        aoProcRegistry["eq"] = eq
        aoProcRegistry["gt"] = gt
        aoProcRegistry["gte"] = gte
        aoProcRegistry["if"] = _if
        aoProcRegistry["is_infinite"] = is_infinite
        aoProcRegistry["is_nan"] = is_nan
        aoProcRegistry["is_nodata"] = is_nodata
        aoProcRegistry["is_valid"] = is_valid
        aoProcRegistry["lt"] = lt
        aoProcRegistry["lte"] = lte
        aoProcRegistry["neq"] = neq
        aoProcRegistry["not"] = _not
        aoProcRegistry["or"] = _or
        aoProcRegistry["xor"] = xor

        aoProcRegistry["load_collection"] = load_collection
        aoProcRegistry["save_result"] = save_result

        oCallable = oParsedGraph.to_callable(process_registry=aoProcRegistry)

        wasdi.wasdiLog("OpenEO Graph callable done")

        oCallable()

    except Exception as oError:
        wasdi.wasdiLog("wasdiOpenEO -> GRAPH ERROR: " + str(oError))

    wasdi.setPayload(aoPayload)
    wasdi.wasdiLog("Bye")
    wasdi.updateStatus("DONE", 100)

if __name__ == '__main__':
    wasdi.init("./config.json")
    run()