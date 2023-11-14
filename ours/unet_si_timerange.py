import functools
import os
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import gc
import logging
from tensorflow.keras.backend import clear_session
import xarray
from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect
from typing import Dict
import traceback

## Needed because joblib hijacks root logger
logging.basicConfig(level=logging.INFO)


@functools.lru_cache(maxsize=25)


class WaterSegmentation():
    
    
    def __init__(self, logger=None):
        if logger is None:
            self.log = logging.getLogger(__name__)
        else: self.log=logger
        self.models=None
    
    def processWindow(self, model, data, patch_size=120):
        """ preprocess the data and apply the model"""
            
        ## read the xarray data as a numpy array
        data = data.values.copy()
            
        ## apply model
        prediction = np.squeeze(
            model.predict(data.reshape(1, patch_size * patch_size, -1)).reshape((patch_size, patch_size)))

        clear_session() # to avoid memory leakage executors
        gc.collect()

        return prediction
    

allbands_s1 = ["VV", "VH", "R"]
allbands_s2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AWEInsh", "AWEIsh", "NDWI", "MNDWI"]
allbands_s2_toscale = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AWEInsh", "AWEIsh"]
sc_s2 = 1e-4


def get_sen_ranges(scaling="narrow"):
    ranges = dict()
    for band in ["B01", "B02", "B03", "B04"]:
        ranges[band] = [-0.05, 0.2] if scaling == "narrow" else [0, 0.2] if scaling == "adjusted" else [0, 1]
    ranges["B05"] = [0, 0.3] if scaling in ["narrow", "adjusted"] else [0, 1]
    for band in ["B06", "B07"]:
        ranges[band] = [0, 0.6] if scaling in ["narrow", "adjusted"] else [0, 1]
    for band in ["B08", "B8A", "B09"]:
        ranges[band] = [0, 0.5] if scaling == "narrow" else [0, 0.7] if scaling == "adjusted" else [0, 1]
    for band in ["B11", "B12"]:
        ranges[band] = [0, 0.3] if scaling == "narrow" else [0, 0.4] if scaling == "adjusted" else [0, 1]
    ranges["SCL"] = [0, 11]
    for band in ["AWEIsh", "AWEInsh"]:
        ranges[band] = [-0.7, 0.7] if scaling == "narrow" else [-0.8, 0.8] if scaling == "adjusted" else [-1, 1]
    for band in ["MNDWI", "NDWI"]:
        ranges[band] = [-1, 1] if scaling == "narrow" else [-0.8, 0.8] if scaling == "adjusted" else [-1, 1]
    ranges["VV"] = [-30, 5]  if scaling == "narrow" else [-20, 0] if scaling == "adjusted" else [-1, 1]
    ranges["VH"] = [-40, 0] if scaling == "narrow" else [-28, -8] if scaling == "adjusted" else [-1, 1]
    ranges["R"] = [0, 20]
    return ranges


def scale_values(values, val_min, val_max, scaled_min=-1, scaled_max=1):
    values[values > val_max] = val_max
    values[values < val_min] = val_min
    return ((scaled_max - scaled_min) * (values - val_min) / (val_max - val_min) + scaled_min)


def scale_sentinel(array, band, min_scaled=-1, max_scaled=1, scaling="narrow"):
    ranges = get_sen_ranges(scaling=scaling)
    return scale_values(array, *ranges[band], scaled_min=min_scaled, scaled_max=max_scaled)


def to_db(image):
    return 10 * np.log10(image)


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """ main function """

    try:
        ## get path to model, falling back to default value
        if len(context) > 0:
            path_model = context.get("path_model", "/data/users/Public/landuytl/water-detection/Models/PS120_2100-SY-FW0.9_MOP0.005_MOT0.80_S0_2100_S2-4B_D4_F32_W-E_WC-23456_D-UC_A0_best.h5")
            inspect(message="path_model from context", data=[path_model, os.path.exists(path_model)])
        else: # for local debugging
            path_model = "/projects/TAP/CORSA/water-detection/Models/unet/PS120_2100-SY-FW0.9_MOP0.005_MOT0.80_S0_2100_S2-4B_D4_F32_W-E_WC-23456_D-UC_A0/UNet_best.h5"
            inspect(message="path_model set to default", data=[path_model, os.path.exists(path_model)])

        ## get the array
        cubearray:xarray.DataArray = cube.get_array()    
        inspect(message="cubearray", data=cubearray) # bands, y, x

        ## normalize data
        inputarray = cubearray.transpose("bands", "y", "x")
        bands = inputarray.bands.values
        for i_b, band in enumerate(bands):
            if band in allbands_s2_toscale:
                inputarray[i_b,:,:] = inputarray[i_b,:,:] * sc_s2
            elif band in allbands_s1:
                inputarray[i_b,:,:] = to_db(inputarray[i_b,:,:])
            inputarray[i_b,:,:] = scale_sentinel(inputarray[i_b,:,:].values, band)

        ## transpose to format accepted by model & remove t-dimension
        inputarray = inputarray.transpose("y", "x", "bands")
        inspect(message="inputarray", data=inputarray)

        ## load in the pretrained U-Net keras models and do inference!
        s = WaterSegmentation()
        model = load_model(path_model)
        result = s.processWindow(model, inputarray, patch_size=120)
        inspect(message="result", data=result)

        ## transform your numpy array predictions into an xarray
        result = result.astype(np.float64)
        result_xarray = xarray.DataArray(result, dims=["y", "x"], coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]))
        result_xarray = result_xarray \
            .expand_dims("bands",0).assign_coords(bands=["prediction"])
        inspect(message="result_xarray", data=result_xarray)

        ## openEO assumes a fixed data order, so transpose your result and store it in a XarrayDataCube that you return
        result_xarray = result_xarray.transpose("bands", "y", "x")
        inspect(message="result_xarray_final", data=result_xarray)
        return XarrayDataCube(result_xarray)
    except Exception as e:
        inspect(message="Error in UDF!",
                data=traceback.format_exc())
        return XarrayDataCube(xarray.DataArray(data=np.ones((1, 120, 120)) * -1, dims=["bands", "y", "x"])) # return dummy data