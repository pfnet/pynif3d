# Getting Started with NeRF

Neural Radiance Fields (abbreviated as NeRF) is a method that optimizes a neural
implicit representation of a complex scene using a sparse set of input views (images)
and their ground-truth camera poses. Using the trained model, novel/unseen views of the
same scene can be synthesized. For more information, please refer to the
[project page](https://www.matthewtancik.com/nerf).

## Training

The example in this tutorial uses the Blender synthetic dataset, which can be downloaded
from [here](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)
for inspection purposes or will automatically be downloaded by the training script. This
dataset contains images and camera poses corresponding to viewpoints of several scenes,
stored in separate directories ("chair", "drums", "ficus" etc.). In this tutorial, a
NeRF model that can be used to reconstruct the `lego` scene, will be trained.

#### Running the Training Script

From the root directory, run the following:

```
python3 examples/nerf/train.py --data-directory ${PATH_TO_DATA_DIRECTORY} --save-directory ${PATH_TO_SAVE_DIRECTORY}
```

To change the values of other hyperparameters, please check the `parse_arguments`
function of the training script.

#### TensorBoard Visualization

It is often useful to check the intermediate values of various variables during
training. We report the MSE and PSNR values between the reconstructed RGB values at the
sampled points along the rays, and the ground-truth RGB values.

From the root directory, run the following:

```
tensorboard --logdir=${PATH_TO_OUTPUT}
```

## Evaluation

From the root directory, run the following:

```
python3 examples/nerf/evaluate.py --data-directory ${PATH_TO_DATA_DIRECTORY} --model-file ${PATH_TO_MODEL_FILE}
```

Once the evaluation has finished, the results (average MSE/PSNR per test dataset) will
be saved to an output JSON file.

## Synthesizing Novel Viewpoints

Novel viewpoints of the scene can be synthesized using a trained model. From the root
directory, run the following:

```
python3 examples/nerf/demo.py --data-directory ${PATH_TO_DATA_DIRECTORY} --model-file ${PATH_TO_MODEL_FILE} --save-directory ${PATH_TO_SAVE_DIRECTORY}
```

This script will synthesize viewpoints from given camera poses and create a video in the
output (save) directory.
