# Anomaly detection on PCB using segmentation network
A prototype created to detect anomaly on a PCI board using Efficient network with UNET architecture. 

## Installing / Getting started
```shell
pip install -r requirements.txt
```

### Initial Configuration
Prepare a dataset with original image with masked image of the defect located (have to change in the code)

    ├── dataset                   # Contains the images
    │   ├── image                 # Original images
    │   ├── image-masked          # Masked the location of defect in the original images
    ├── anomaly-pci               
    │   ├── efficientUNET_v1.h5   # pre-trained model created for the prototype
    │   ├── requirements.txt      # All the library used in this project
    |   ├── unet.py               # For training a new model using Efficient network with UNET architecture
    └── ...

## Developing
This project is all written in python 3.
```shell
git clone https://github.com/ttansuwan/anomaly-pci
cd anomaly-pci
# recommend to use virtual-env
python -m venv anomaly-pci-env
source anomaly-pci-env/bin/activate
pip install -r requirements.txt
```
## Features
- Pre-trained model is included if wanted to conduct any experiment (warning: it does not perform that well due to lack of data)
- Can create your own model using unet.py

## Links
- Credits:
    - https://github.com/qubvel/segmentation_models
    - https://towardsdatascience.com understanding-semantic-segmentation-with-unet-6be4f42d4b47
