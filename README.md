# Framework to Locate Sensitive Information in Medical Images 
This framework uses an anomaly detection model to identify pixels in chest radiographs that contain foreign objects. 

## Abstract 
Healthcare institutions gather sensitive patient data during diagnosis and treatment. With increased digitalisation, this data is shared for research purposes or stored digitally, however this risks the exposure of highly personal information. Anonymisation of medical data is critical to protect patient privacy and comply with legal regulations, though it must not compromise clinical relevance or diagnostic utility of the information. While anonymisation of textual medical data has been studied, methods for identifying areas and pixels within medical images that may compromise privacy remain underexplored.

This thesis addresses the challenge of medical image anonymisation by focusing on sensitive information in the form of foreign objects, such as implants or medical devices, which can serve as unique patient identifiers. Image processing techniques and a trained anomaly detection model are applied to locate foreign objects at pixel level. Traditional pixel-wise segmentation techniques depend on annotated training data which are unavailable for this specific task. Therefore, this framework employs unsupervised anomaly detection trained using the publicly available PadChest dataset. 

A system to detect and locate pixels containing foreign objects is implemented, including comprehensive data preprocessing, model training, output post-processing and extensive evaluation. Using precise pixel-wise localisation, this model facilitates subsequent removal of sensitive areas contributing to anonymisation while preserving medical integrity of the image. The resulting model achieves an AUC-ROC of 0.788 and furthers medical image anonymisation in an age of increasing digital data usage and sharing.

## Installation Guidelines 
This section provides instructions for the installation and setup of the designed framework for the detection of pixels containing foreign objects in medical images. As this implementation is based on a pre-trained model, please also consider [Lily Georgescu's](https://github.com/lilygeorgescu/MAE-medical-anomaly-detection.git) repository with the original masked autoencoder for anomaly detection. The steps listed below can be followed to ensure correct installation and successful execution of the framework. 

### Step 1: Clone the GitHub Repository 

First the GitHub repository must be cloned to the local machine. Create a local folder where the implementation should be saved. Open a terminal and navigate to the created directory. Then execute the following command: 

```
git clone https://github.com/jr-chapman/foreign_object_anomaly_detection.git
```

### Step 2: Create the Framework Environment

As the framework requires several Python packages with specific versions and dependencies, it is recommended that the implementation is run in its own Conda environment. The framework requires Python 3.10.16.
- See [here](https://www.python.org/downloads/) for assistance installing Python
- See [here](https://www.anaconda.com/download) for assistance installing Anaconda

Use the following commands to set up a Conda environment. Replace _<my_environment>_ with your desired environment name. 
```
conda create –name <my_environment>
conda activate <my_environment>
```

### Step 3: Install the Required Packages 
All the Python packages, their dependencies and versions are listed in the _requirements.txt_ file. To install them, run the following command: 
```
pip install -r requirements.txt 
```

### Step 4: Download the PadChest Dataset 
The PadChest dataset published by Bustos et al. can be downloaded from this link: [https://bimcv.cipf.es/bimcv-projects/padchest/](https://bimcv.cipf.es/bimcv-projects/padchest/).
To download the dataset, click the button _Download Complete Dataset_. You will first have to fill in a request form and agree to the terms of use. Then you will be provided with the link to the full dataset download. Download folders 1-50 and 54 to your local machine without changing the original folder structure or names. Unzip the folders and load them into a local directory. The directory name will be needed for execution. 

- If you wish to verify download was successful, the file _Verify_Zips_ImageCounts.csv.xlsx_ provides an overview of the separate files and number of contained images to check you have the whole dataset.

Finally, download the file _PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv_ which contains all dataset metadata. This folder should be saved to the _dataset_ directory in the cloned repository, so it is accessible during execution.

### Step 5: Execute the Model 

The following section provides you with some example calls for model execution for training, testing and inference. GPU acceleration is recommended for execution. The framework is executed using command line arguments and the minimum arguments required for successful execution are shown here. If you wish to tune other model parameters, please the _args.py_ file in the _config_ directory of the repository for all available arguments. 

- If you do not wish to train your own model, a pre-trained model for this task is available in (checkpoint.pth)[checkpoint.pth]. 
- The pre-trained weights from [Georgescu](https://github.com/lilygeorgescu/MAE-medical-anomaly-detection.git) used for continued training are in the file (brats_pretrained.pth)[checkpoint.pth], no further action is required to use them. 

#### Model Training 
To train the model, call the following command from the command line. Replace _<dataset_path>_ with the path to the directory where you saved the PadChest dataset images. As the original images are rather large, please beware that they may need resizing before use depending on your available computational resources. 

```
python main.py --train --data_path=_<dataset_path>_ --lr=0.001 --model_path=../brats_pretrained.pth 
```

#### Model Evaluation or Testing
For execution of evaluation or testing, use the following command. Provide the path to the pre-trained model you wish to use for evaluation or testing in _<checkpoint>_ and the dataset subset in _<subset>_. The dataset subsets are: _validation_negative, testing_negative, validation_positive, testing_positive_
```
python3 main.py --evaluate --model_path=<checkpoint> --data_path=<dataset_path> --evaluation_set=<subset> --err=ssim,anomaly
```

Training and evaluation or testing can also be called together using the following arguments. In this case, the previously trained model will directly be used for evaluation. 
```
python main.py --train --data_path=<dataset_path> --lr=0.001 --model_path=../brats_pretrained.pth --evaluate --evaluation_set=<subset> --err=ssim,anomaly
```

#### Model Inference 
To run inference, use the arguments below. Running inference means no dataset splits will be performed and the whole dataset will be passed through the pipeline with the provided trained model. Replace _<trained_model_checkpoint>_ with the path to the trained model. 

```
python main.py --inference --data_path=<dataset_path> --model_path=<trained_model_checkpoint>  
```

By following these steps, the setup and execution of the framework should be successful. If you encounter any difficulties, please contact the repository maintainer for assistance. 

