# Getting Started with Convolutional Occupancy Networks

Convolutional Occupancy Networks (CON) is a method for completing noisy point clouds 
that are given as input. The algorithm takes noisy point clouds as input, applies point
feature extraction to them, projects the extracted point features onto single or
multiple cubical planes or a 3D grid. Then, it applies a NIF model conditioned on input
cartesian coordinate and plane/grid features queried at given coordinates. The refined
mesh model can be generated though several methods (marching cubes, MISE, etc.) applied 
to NIF model results. For more information, please refer to the 
[paper](https://arxiv.org/abs/2003.04618).

## Training

To train a CON model, you need to prepare a dataset containing the set of noisy point
clouds and the occupancy information of the shape that is to be reconstructed. This 
information can be obtained through the object's mesh, however the usage of meshes is 
not supported yet. In this tutorial the ShapeNet dataset will be used.

#### Dataset Preparation

The authors of CON provide the download script at the following
[link](https://github.com/autonomousvision/occupancy_networks#preprocessed-data). 
Download the dataset and extract the content to a folder. 

#### Running the Training Script

From the root directory, run the following command:

```
python3 examples/con/train.py --data-directory ${PATH_TO_DATA_DIRECTORY} --save-directory ${PATH_TO_SAVE_DIRECTORY}
```

By default the Convolutional Occupancy Networks training scripts uses the grid mode. To 
change the operation mode or the values of other hyperparameters, please check the 
`parse_arguments` function of the training script.

#### TensorBoard Visualization

It is often useful to check the intermediate values of various variables during 
training. We report the loss values between the predicted occupancies and ground-truth
occupancies. To visualize the output, run the following command:

```
tensorboard --logdir=saved_models
```

## Evaluation

From the root directory, run the following:

```
python3 examples/con/evaluate.py --data-directory=${PATH_TO_DATA_DIRECTORY} --model-file=${PATH_TO_MODEL_FILE} --output-file ${PATH_TO_OUTPUT_FILE}
```

Once the evaluation has finished, the results will be saved to an output JSON file.