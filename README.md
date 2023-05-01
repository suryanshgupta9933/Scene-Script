# 👁‍🗨 Scene-Script

## Introduction
The goal of this project is to create a pipeline that can take an image as input and output the text in the image. The pipeline consists of a VIT model for feature extraction and a Transformer model for text generation. The pipeline is deployed using Nvidia's PyTrition and Streamlit app.

## Dataset
The dataset used for the model is Flickr30k. The dataset consists of 31,783 images collected from Flickr. Each image is paired with 5 captions. The dataset can be downloaded from [here](https://www.kaggle.com/hsankesara/flickr-image-dataset).

## Installation
Clone the repository using the following command:
```
git clone https://github.com/smackiaa/Scene-Script.git
```
Now, install the required packages using the following command:
```
pip install -r requirements.txt
```

## Usage
### Preprocessing
Preprocess the captions text file to remove unnecessary data using the following command:
```
python preprocess.py --caption_file <path/to/captions/file>
```
- `caption_file` is the path to the captions file.

> **Note:** The preprocessed captions file is saved in the same directory as the original captions file. It overwrites the original captions file. So, either use a copy of the original captions file or rename the original captions file before running the above command or use the one given in the repository in the `data` directory.

### Model Configuration
- The model configuration json file contains the model parameters.
- There are three different model configurations given in the `configs` directory.
- The smallest one being 92 million parameters and the largest one being 112 million parameters.
- The default model configuration is set to 97 million parameters which is a good balance between the model size and the performance.

### Hyperparameters
- The model configuration json file also contains the training parameters.
- The `learning_rate` is set to 1e-4.
- The `batch_size` is set to 128.
- The `num_epochs` is set to 3.

> **Note:** Change the batch size and the number of epochs according to the GPU memory and the time available.

### Training
To train the model, run the following command:
```
python train.py --model_config <model-config> --images_dir <path/to/images/directory> --caption_file <path/to/captions/file>
```
- `model_config` is the path to the model configuration file. The model configuration file is a json file that contains the model parameters and the training parameters. 

> **Note:** Default is set to 97 million parameters.

- `images_dir` is the path to the directory containing the images.
- `caption_file` is the path to the captions file.

### Inference
To run inference on a single image, run the following command:
```
python inference.py --model_config <model-config> --model_weights <path/to/model/weights> --image_path <path/to/image>
```
- `model_config` is the path to the model configuration file. The model configuration file is a json file that contains the model parameters.

> **Note:** Default is set to 97 million parameters.

- `model_weights` is the path to the model weights file.

> **Note:** Default is set to the 'weights/scene-script.pth' file.

- `image_path` is the path to the image.
