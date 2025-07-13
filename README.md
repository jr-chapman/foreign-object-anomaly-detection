# Framework to Locate Sensitive Information in Medical Images 
This framework uses an anomaly detection model to identify pixels in chest radiographs that contain foreign objects. 

## Abstract 
Healthcare institutions gather sensitive patient data during diagnosis and treatment. With increased digitalisation, this data is shared for research purposes or stored digitally, however this risks the exposure of highly personal information. Anonymisation of medical data is critical to protect patient privacy and comply with legal regulations, though it must not compromise clinical relevance or diagnostic utility of the information. While anonymisation of textual medical data has been studied, methods for identifying areas and pixels within medical images that may compromise privacy remain underexplored.

This thesis addresses the challenge of medical image anonymisation by focusing on sensitive information in the form of foreign objects, such as implants or medical devices, which can serve as unique patient identifiers. Image processing techniques and a trained anomaly detection model are applied to locate foreign objects at pixel level. Traditional pixel-wise segmentation techniques depend on annotated training data which are unavailable for this specific task. Therefore, this framework employs unsupervised anomaly detection trained using the publicly available PadChest dataset. 

A system to detect and locate pixels containing foreign objects is implemented, including comprehensive data preprocessing, model training, output post-processing and extensive evaluation. Using precise pixel-wise localisation, this model facilitates subsequent removal of sensitive areas contributing to anonymisation while preserving medical integrity of the image. The resulting model achieves an AUC-ROC of 0.788 and furthers medical image anonymisation in an age of increasing digital data usage and sharing.

## Installation Guidelines 
This section provides instructions for the installation and setup of the designed framework for the detection of pixels containing foreign objects in medical images. As this implementation is based on a pre-trained model, please also consider [Lily Georgescu's](https://github.com/lilygeorgescu/MAE-medical-anomaly-detection.git) repository with the original masked autoencoder for anomaly detection. Follow the steps below to ensure correct installation and successful execution of the framework.

### Step 1: Clone the GitHub Repository 

First the GitHub repository must be cloned to the local machine. Create a local folder where the implementation should be saved. Open a terminal and navigate to the created directory. Then execute the following command: 

```
git clone https://github.com/jr-chapman/foreign_object_anomaly_detection.git
```

### Step 2: Create the Framework Environment

As the framework requires several Python packages with specific versions and dependencies, it is recommended that the implementation is run in its own Conda environment. The framework requires Python 3.10.16.
- See [here](https://www.python.org/downloads/) for assistance installing Python
- See [here](https://www.anaconda.com/download) for assistance installing Anaconda

Open the cloned repository in an IDE of your choice and use the following commands to set up the Conda environment. All the Python packages, their dependencies and versions required to execute this framework are installed with the Conda environment through the [_environment.yml_](https://github.com/jr-chapman/foreign-object-anomaly-detection/blob/main/environment.yml) file. 
```
conda env create -f environment.yml
conda activate foreign-object-anomaly-detection
```
If you wish to rename the Conda environment, change the name specified in the [_environment.yml_](https://github.com/jr-chapman/foreign-object-anomaly-detection/blob/main/environment.yml) file. 

### Step 3: Download the PadChest Dataset 
The PadChest dataset published by Bustos et al. can be downloaded from this link: [https://bimcv.cipf.es/bimcv-projects/padchest/](https://bimcv.cipf.es/bimcv-projects/padchest/).
To download the dataset, click the button _Download Complete Dataset_. You will first have to fill in a request form and agree to the terms of use. Then you will receive with the link to download the full dataset. Download folders 1-50 and 54 to your local machine without changing the original folder structure or names. Unzip the folders and load them into a local directory. The directory name will be needed for execution. 

- If you wish to verify download was successful, the file _Verify_Zips_ImageCounts.csv.xlsx_ provides an overview of the separate files and number of contained images to check you have the whole dataset.

Finally, download the file _PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv_ which contains all dataset metadata. Create a folder _dataset_ in the root directory of this repository and save the CSV file here. 

### Step 4: Download the Pre-Trained Model
This implementation is based on the pre-trained MAE anomaly detection model by Lily Georgescu. To download the pre-trained model, navigate to the repository at this link: [https://github.com/lilygeorgescu/MAE-medical-anomaly-detection](https://github.com/lilygeorgescu/MAE-medical-anomaly-detection) and scroll to the section _Results and trained models_. Here, install the pre-trained model for the BraTS2020 dataset over the Google Drive link. A direct link to the Google Drive is provided [here](https://drive.google.com/file/d/1QxFHy8nYeaj5OPQExmcbf9PQNzMOhoCy/view).

Once downloaded, copy and save the _brats\_pretrained.pth_ file to the root directory of the repository. 

### Step 5: Execute the Model 

The following section provides some example calls for model training, testing and inference. GPU acceleration is recommended for execution. The framework is run using command line arguments and the minimum arguments required for successful execution are shown here. If you wish to tune other model parameters, please see the [_args.py_](https://github.com/jr-chapman/foreign-object-anomaly-detection/blob/main/config/args.py) file in the _config_ directory of the repository for all available arguments. 

#### Model Training 
To train the model, run the following command from the command line. Replace _<dataset_path>_ with the path to the directory where the PadChest dataset images are saved. As the original images are rather large, please beware that they may need resizing before use depending on your available computational resources. 

```
python main.py --train --data_path=<dataset_path> --model_path=./brats_pretrained.pth 
```

#### Model Validation and Testing
For execution of validation and testing, use the command below. Provide the path to the pre-trained model you wish to use for validation and testing in _<trained_model_checkpoint>_ and the dataset subset in _<validation_subset>_. The dataset subsets are: _validation_negative, testing_negative, validation_positive, testing_positive_
```
python main.py --validate --model_path=<trained_model_checkpoint> --data_path=<dataset_path> --validation_set=<validation_subset> --err=ssim,anomaly
```

Training and validation or testing can also be called together using the following arguments. In this case, the previously trained model will directly be used for validation. 
```
python main.py --train --data_path=<dataset_path> --lr=0.001 --model_path=./brats_pretrained.pth --validate --validation_set=<validation_subset> --err=ssim,anomaly
```

#### Model Inference 
To run inference, use the arguments below. Running inference means no dataset splits will be performed and the whole dataset will be passed through the pipeline with the provided trained model. Replace _<trained_model_checkpoint>_ with the path to the trained model. 

```
python main.py --inference --data_path=<dataset_path> --model_path=<trained_model_checkpoint>  
```

By following these steps, the setup and execution of the framework should be successful. If you encounter any difficulties, please contact the repository maintainer for assistance. 

