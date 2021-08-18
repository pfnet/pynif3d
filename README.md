# PyNIF3D

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pfnet/pynif3d/blob/master/LICENSE)
[![Read the Docs](https://readthedocs.org/projects/pynif3d/badge/?version=latest)](https://pynif3d.readthedocs.io/en/latest/)

PyNIF3D is an open-source PyTorch-based library for research on neural implicit
functions (NIF)-based 3D geometry representation. It aims to accelerate research by 
providing a modular design that allows for easy extension and combination of NIF-related
components, as well as readily available paper implementations and dataset loaders.

As of August 2021, the following implementations are supported:

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)
- [Convolutional Occupancy Networks (Peng et al., 2020)](https://arxiv.org/abs/2003.04618)
- [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance (Yariv et al., 2020)](https://arxiv.org/abs/2003.09852)

## Installation

To get started with PyNIF3D, you can use `pip` to install a copy of this repository on
your local machine or build the provided Dockerfile.

### Local Installation

```
pip install --user "https://github.com/pfnet/pynif3d.git"
```

The following packages need to be installed in order to ensure the proper functioning of
all the PyNIF3D features:

- torch_scatter>=1.3.0
- torchsearchsorted>=1.0

A [script](https://github.com/pfnet/pynif3d/blob/main/post_install.bash) has been
provided to take care of the installation steps for you. Please download it to a
directory of choice and run:

```
bash post_install.bash
```

### Docker Build

#### Enabling CUDA Support

Please make sure the following dependencies are installed in order to build the Docker 
image with CUDA support:

- nvidia-docker
- nvidia-container-runtime

Then register the `nvidia` runtime by adding the following to `/etc/docker/daemon.json`:
```
{
    "runtimes": {
        "nvidia": {
            [...]
        }
    },
    "default-runtime": "nvidia"
}
```

Restart the Docker daemon:
```
sudo systemctl restart docker
```

You should now be able to build a Docker image with CUDA support.

#### Building Dockerfile

```
git clone https://github.com/pfnet/pynif3d.git
cd pynif3d && nvidia-docker build -t pynif3d .
```

#### Running the Container

```
nvidia-docker run -it pynif3d bash
```


## Tutorials

Get started with PyNIF3D using the examples provided below:

<table style="text-align: center;">
  <tr>
    <td>
        <img src="https://camo.githubusercontent.com/88a39df6c735d3b11571504bcacf9c6a322c743b463e0784fe66d936b8e3f688/68747470733a2f2f70656f706c652e656563732e6265726b656c65792e6564752f7e626d696c642f6e6572662f6c65676f5f3230306b5f323536772e676966" height="150px" alt=""/>
    </td>
    <td>
        <img src="https://github.com/autonomousvision/convolutional_occupancy_networks/raw/master/media/teaser_matterport.gif" height="150px" alt=""/>
    </td>
    <td>
        <img src="https://user-images.githubusercontent.com/1044197/123730898-1ca15900-d8d2-11eb-9125-426c8a6f4f82.gif" height="150px" alt=""/>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/pfnet/pynif3d/blob/master/examples/nerf/README.md">NeRF Tutorial</a>
    </td>
    <td>
        <a href="https://github.com/pfnet/pynif3d/blob/master/examples/con/README.md">CON Tutorial</a>
    </td>
    <td>
        <a href="https://github.com/pfnet/pynif3d/blob/master/examples/idr/README.md">IDR Tutorial</a>
    </td>
  </tr>
</table>

In addition to the tutorials, pretrained models are also provided and ready to be used.
Please consult [this page](https://github.com/pfnet/pynif3d/blob/master/examples/pretrained_models.md) for more information.

## License

PyNIF3D is released under the MIT license. Please refer to [this document](https://github.com/pfnet/pynif3d/blob/master/LICENSE) for more information.

## Contributing

We welcome any new contributions to PyNIF3D. Please make sure to read
the [contributing guidelines](https://github.com/pfnet/pynif3d/blob/master/CONTRIBUTING.md)
before submitting a pull request.

## Documentation

Learn more about PyNIF3D by reading
the [API documentation](http://pynif3d.readthedocs.io/en/latest/).

