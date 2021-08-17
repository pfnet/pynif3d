# Getting Started with IDR

Implicit Differentiable Renderer (abbreviated as IDR) is a neural network architecture
that simultaneously learns unknown geometry, camera parameters, and a neural
renderer that approximates the light reflected from the surface towards the camera. For
more information, please refer to the [project page](https://lioryariv.github.io/idr).

## Training

The example in this tutorial uses the DTU MVS dataset, which can be downloaded
from [here](https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip?dl=1) for inspection
purposes or will automatically be downloaded by the training script. This dataset
contains images and camera poses corresponding to scans for several scenes. In this
tutorial, an IDR model that can be used to reconstruct a specific scene,
passed as input argument to the training script, will be trained.

#### Running the Training Script

From the root directory, run the following:

```
python3 examples/idr/train.py --data-directory=${PATH_TO_DATA_DIRECTORY} --scan-id ${SCAN_ID}
```

To change the values of other hyperparameters, please check the `parse_arguments`
function of the training script.

#### TensorBoard Visualization

It is often useful to check the intermediate values of various variables during
training. We report the MSE and PSNR values corresponding to the reconstructed RGB 
values at the sampled points along the rays and the ground-truth RGB values.

From the root directory, run the following:

```
tensorboard --logdir=${PATH_TO_OUTPUT}
```

Replace `${PATH_TO_OUTPUT}` with the path to the directory that contains the
checkpoints.

## Evaluation

From the root directory, run the following:

```
python3 examples/idr/evaluate.py --data-directory ${PATH_TO_DATA_DIRECTORY} --model-file=${PATH_TO_MODEL_FILE} --scan-id ${SCAN_ID} --output-file ${PATH_TO_OUTPUT_FILE}
```

Once the evaluation has finished, the results (average MSE/PSNR per test dataset) will
be saved to an output JSON file.

## Synthesizing Novel Viewpoints

Novel viewpoints of the scene can be synthesized using a trained model. From the root
directory, run the following:

```
python3 examples/idr/demo.py --data-directory ${PATH_TO_DATA_DIRECTORY} --model-file ${PATH_TO_MODEL_FILE} --scan-id ${SCAN_ID} --save-directory ${PATH_TO_SAVE_DIRECTORY}
```

This script will synthesize viewpoints from given camera poses and save the rendered
images to an output directory.