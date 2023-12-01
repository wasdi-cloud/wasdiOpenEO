import os.path

import wasdi
from osgeo import gdal
import xarray as xr
import zipfile
import rioxarray
import pandas as pd
from pathlib import Path

__all__ = [
    "load_collection",
    "save_result"
]

m_bOpitmized = True
m_bDelete = False

def load_collection(id, bands, spatial_extent, temporal_extent):

    sEndDate = temporal_extent.end.__root__.strftime("%Y-%m-%d")
    sStartDate = temporal_extent.start.__root__.strftime("%Y-%m-%d")

    wasdi.wasdiLog("Loading collection " + id + " from " + sStartDate + " to " + sEndDate)

    if id == "Sentinel-2":
        return load_collection_s2(id,bands,spatial_extent,temporal_extent)

    return None

def load_collection_s2(id, bands, spatial_extent, temporal_extent):

    sNetCdfFilePath = wasdi.getPath("cube.nc")

    if m_bOpitmized:
        if os.path.exists(sNetCdfFilePath):
            oDataSet = xr.open_dataset(sNetCdfFilePath)
            return oDataSet

    # Get Start and End Dates
    sEndDate = temporal_extent.end.__root__.strftime("%Y-%m-%d")
    sStartDate = temporal_extent.start.__root__.strftime("%Y-%m-%d")

    # We assume S2L1 at the moment. So bands are called like B01 and not B1.
    # This code just fixes the band names
    for iBand in range(len(bands)):
        sBand = bands[iBand]
        if len(sBand)==2:
            sBand = sBand[0]+"0"+sBand[1]
            bands[iBand] = sBand

    # We search the EO Images
    aoWasdiImages = wasdi.searchEOImages(sPlatform=id, sProductType="S2MSI1C", sDateFrom=sStartDate, sDateTo=sEndDate, fULLat=spatial_extent.north, fULLon=spatial_extent.west, fLRLat=spatial_extent.south, fLRLon=spatial_extent.east)

    wasdi.wasdiLog("Found " + str(len(aoWasdiImages)) + " Images")

    # And get a list of the WASDI available images
    asFiles = wasdi.getProductsByActiveWorkspace()

    # Here we will save the images still to import
    aoImagesToImport = []
    # Here the names of the zip files
    asZipFiles = []
    # And here the full path of the zip files
    asZipFilesFullPath = []

    # We check the images already available in the workspace
    for oImage in aoWasdiImages:
        sFileName = oImage["fileName"]
        if not sFileName.endswith(".zip"):
            sFileName = sFileName + ".zip"

        asZipFiles.append(sFileName)

        if sFileName not in asFiles:
            aoImagesToImport.append(oImage)

    # If there are files to import lets do it
    if len(aoImagesToImport)>0:
        wasdi.wasdiLog("Importing " + str(len(aoImagesToImport)) + " Images")
        wasdi.importProductList(aoImagesToImport)

    # Here we understand the different  involved
    asDates = list(set([sS2Tile.split(sep='_')[2].split(sep='T')[0] for sS2Tile in asZipFiles]))
    asDates.sort()

    # Here we obtain the array with full path of our files
    for sFileName in asZipFiles:
        if m_bOpitmized:
            asZipFilesFullPath.append(wasdi.getPath(sFileName))
        else:
            if wasdi.fileExistsOnWasdi(sFileName):
                asZipFilesFullPath.append(wasdi.getPath(sFileName))

    aoFilesCubeIndex = {}
    asLocalFilesToDelete = []

    # For each Date
    for sDate in asDates:
        # For each Band
        for sBand in bands:

            wasdi.wasdiLog("Working on date " + sDate + " Band " + sBand)

            # We take a list of images for each band
            if sBand not in aoFilesCubeIndex.keys():
                # Initialize the dictionary with this band
                aoFilesCubeIndex[sBand] = []

            # Initialize an array with the name of the jp2 files we need
            asJp2BandNames = [sBand + '.jp2']

            # Extract a list of the EO Images of this day
            asDayFiles = [sFile for sFile in asZipFilesFullPath if sDate + "T" in sFile]

            # Name of the daily mosaic we are going to produce
            sDailyMosaic = sDate + "_" + sBand + "_mosaic.vrt"

            aoAllBandsZipFiles = []

            # For all the S2 Zip Files of this day
            for sZipName in asDayFiles:
                with zipfile.ZipFile(sZipName, 'r') as sZipFile:
                    asNameList = sZipFile.namelist()
                    asBandsInS2File = [sName for sName in asNameList for sBand in asJp2BandNames if sBand in sName]

                asJp2BandsFillPath = ['/vsizip/' + sZipName + '/' + sBand for sBand in asBandsInS2File]

                for sJp2Band in asJp2BandsFillPath:
                    oBandFile = {}
                    oBandFile["zip"] = Path(sZipName).name
                    oBandFile["jp2"] = sJp2Band

                    # We create an array with the full path of this band inside the zip
                    aoAllBandsZipFiles.append(oBandFile)

            # Reproject all the bands in a WGS84 VRT File
            asReprojectedFiles = []
            for oBandFile in aoAllBandsZipFiles:
                sJp2File = oBandFile["jp2"]
                sZipName = oBandFile["zip"]
                # We reproject in WGS84
                sReprojectedFilePath = wasdi.getPath("reprojected_" + sDate + "_" +sBand + "_" + sZipName.split("_")[5] + ".vrt")
                oWarpOptions = gdal.WarpOptions(dstSRS="EPSG:4326", outputBounds=[spatial_extent.west, spatial_extent.south,
                                                                                  spatial_extent.east,
                                                                                  spatial_extent.north])
                if not os.path.exists(sReprojectedFilePath):
                    gdal.Warp(sReprojectedFilePath, sJp2File, options=oWarpOptions)
                asReprojectedFiles.append(sReprojectedFilePath)
                asLocalFilesToDelete.append(sReprojectedFilePath)

            # Daily mosaic
            sVrtMosaicFilePath = wasdi.getPath(sDailyMosaic)
            if not os.path.exists(sVrtMosaicFilePath):
                gdal.BuildVRT(sVrtMosaicFilePath, asReprojectedFiles, separate=False, srcNodata=0, VRTNodata=0)
            asLocalFilesToDelete.append(sVrtMosaicFilePath)

            # Convert in tiff
            sFileDailyFileForBand = sDate + "_" + sBand + ".tif"
            sFileDailyFileForBandPath = wasdi.getPath(sFileDailyFileForBand)
            oGdalTranslateOptions = gdal.TranslateOptions(format="GTiff")

            if not os.path.exists(sFileDailyFileForBandPath):
                # Now we need to translate it
                gdal.Translate(sFileDailyFileForBandPath, sVrtMosaicFilePath, options=oGdalTranslateOptions)

            aoFilesCubeIndex[sBand].append(sFileDailyFileForBandPath)
            asLocalFilesToDelete.append(sFileDailyFileForBandPath)

    time = xr.Variable('time', pd.DatetimeIndex(asDates))
    aoBands = xr.Variable('band', bands)

    wasdi.wasdiLog("Generating data cube")

    aoDatasets = []

    for sBand in bands:
        aoFilesCubeIndex[sBand].sort()
        oChunks = {'x': 5490, 'y': 5490}
        oDataSet = xr.concat([rioxarray.open_rasterio(sFile, chunks=oChunks) for sFile in aoFilesCubeIndex[sBand]], dim=time)
        aoDatasets.append(oDataSet)

    if len(aoDatasets)>1:
        oDataSet = xr.concat(aoDatasets, dim=aoBands)
    elif len(aoDatasets) == 1:
        oDataSet["band"] = ("band", bands)

    wasdi.wasdiLog("Saving data cube")
    oDataSet.to_netcdf(wasdi.getPath("cube.nc"))
    wasdi.addFileToWASDI("cube.nc")
    # Reopen it to have a dataset instead of a data array
    oDataSet = xr.open_dataset(wasdi.getPath("cube.nc"))

    if m_bDelete:
        for sFile in asLocalFilesToDelete:
            try:
                os.remove(sFile)
            except Exception as oError:
                wasdi.wasdiLog("Error deleting intermediate file " + str(oError))

    if m_bDelete:
        for sFile in asZipFiles:
            wasdi.deleteProduct(sFile)

    return oDataSet


def save_result(data: xr.DataArray, format, options):
    wasdi.wasdiLog("openCollections.save_result")

    bReturn = False

    sFinalCubePath = wasdi.getPath("final.nc")
    data.to_netcdf(sFinalCubePath)

    sFormat = "GTiff"

    if type(format) == str:
        sFormat = format
    else:
        try:
            sFormat = format.__root__
        except:
            wasdi.wasdiLog("openCollections.save_result: error reading format, assuming GTiff")

    if sFormat == "GTiff":
        sFinalTiffPath = wasdi.getPath("final.tif")
        oGdalTranslateOptions = gdal.TranslateOptions(format="GTiff")
        gdal.Translate(sFinalTiffPath, sFinalCubePath, options=oGdalTranslateOptions)
        bReturn = True
        wasdi.addFileToWASDI("final.tif")
    elif sFormat == "NetCDF":
        bReturn = True
        wasdi.addFileToWASDI("final.nc")

    return bReturn

load_collection.implementation = load_collection
save_result.implementation = save_result
