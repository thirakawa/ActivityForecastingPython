# Activity Forecasting Python

This repository contains python implementation of activity forecasting proposed by Kitani et al. [1].


## Execution environment
* Language: Python (2.7.14)
* Modules: Numpy, OpenCV


***
## Training

### Training (inverse optimal control)
    python ioc.py
### Inference (optimal control)
    python oc.py
    
### Optional: Convert XML file to Numpy array format.
If you want to use the original XML files of [1], please convert to numpy array format (*.npy).
Copy to "convert_xml2npy.py" to the directory storing XML files, and execute the following:

    python convert_xml2npy.py





## Reference
1. K. Kitani, et al., "Activity Forecasting," In Proc. of European Conference on Computer Vision (ECCV), pp. 201-214, 2012.
