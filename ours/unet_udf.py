import torch
import numpy as np
import xarray
from openeo.udf import XarrayDataCube
import segmentation_models_pytorch as smp
import logging
import functools
import xarray
from openeo.udf.debug import inspect
import gc


## Needed because joblib hijacks root logger
logging.basicConfig(level=logging.INFO)


@functools.lru_cache(maxsize=25)

class BVMSegmentation():
    
    
    def __init__(self, logger=None):
        if logger is None:
            self.log = logging.getLogger(__name__)
        else: self.log=logger
        self.models=None
    
    def processWindow(self, model, data, patch_size=120):
        """ preprocess the data and apply the model"""
            
        ## read the xarray data as a numpy array
        data = data.values.copy()

        # Apply model to the data
        tensor_data = torch.from_numpy(data).float()
        with torch.no_grad():
            result = model(tensor_data)
            output = torch.softmax(result, dim=0)
            predicted = torch.argmax(output, dim=0)

        gc.collect()
        return predicted.numpy()
    

# adjust as necessary
def preprocess_data(data):
    # Add preprocessing steps here
    # Example: Reshape, normalize, etc.
    return data

def postprocess_data(result):
    # Add postprocessing steps here
    # Example: Reshape, scale back, etc.
    return result

# Load PyTorch model
def load_model(saved_weight, backbone, enc, pretr, in_chan, clas):
    architecture = getattr(smp, backbone)
    model = architecture(
        encoder_name=enc,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretr,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_chan,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=clas,  # model output channels (number of classes in your dataset)
    )
    chkpt = torch.load(saved_weight)
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()
    return model



# Main entry point for the UDF
def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    which_model = context.get("model_backbone")
    encoder = context.get("encoder")
    pretrain = context.get("pretrain")
    in_channel = context.get("in_channel")
    classes = context.get("classes")
    model_path = context.get("model_path")  # Update this path

    model = load_model(model_path, which_model, encoder, pretrain, in_channel, classes)

    # Extract data array from the cube
    cubearray: xarray.DataArray = cube.get_array()
    # Discard cloud for now 
    cubearray_wo_SCL = cubearray.drop_sel({"bands":"SCL"})
    cubearray_final = cubearray_wo_SCL.drop_vars("t")
    print("test")
    print(cubearray_final.dims) # bands, y, x

    inputarray = cubearray_final.transpose("bands", "y", "x")
    bands = inputarray.bands.values
    inspect(message="bands", data=bands)

    # normalization of sat images (without cloud channel)
    norm_lo = np.percentile(inputarray, 1, axis=[1, 2])
    norm_hi = np.percentile(inputarray, 99, axis=[1, 2])
    c = norm_lo[:, None, None]
    d = norm_hi[:, None, None]
    inputarray = (inputarray - c) * (1.0 - 0.0) / (d - c) + 0.0
    inputarray = np.clip(inputarray, 0, 1)

    # Apply the model
    s = BVMSegmentation()
    result = s.processWindow(model, inputarray, patch_size=256)
    inspect(message="result", data=result)

    ## transform your numpy array predictions into an xarray
    result = result.astype(np.float64)
    result_xarray = xarray.DataArray(result, dims=["y", "x"], coords=dict(x=cubearray_final.coords["x"], y=cubearray_final.coords["y"]))
    result_xarray = result_xarray \
        .expand_dims("bands",0).assign_coords(bands=["prediction"])
    inspect(message="result_xarray", data=result_xarray)

    # Postprocess the results if necessary
    # postprocessed_result = postprocess_data(result_array)

    # Wrap the result into an xarray DataArray and then into an XarrayDataCube
    return XarrayDataCube(result_xarray)

