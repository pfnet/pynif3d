# Pretrained Models

The following models are provided and ready to be used:

### Convolutional Occupancy Networks

| Dataset        | Mode     | Plane/Grid Size    | Accuracy (Original) | Link |
| -------------- | -------- | ------------------ | ------------------- | ---- |
| ShapeNet       | Grid     | 32 x 32 x 32       | 0.885 (0.884)       | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_grid_32^3_shapenet.pt)                 |
| ShapeNet       | 1 Plane  | 1 x 64 x 64 x 32   | 0.821 (0.833)       | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_1plane_64_shapenet.pt)                 |
| ShapeNet       | 3 Planes | 3 x 64 x 64 x 32   | 0.872 (0.870)       | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_3planes_64^2_shapenet.pt)              |
| Synthetic Room | Grid     | 64 x 64 x 64 x 32  | 0.859 (-)           | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_grid_64x64x64x32_synthetic_room.pt)    |
| Synthetic Room | 3 Planes | 3 x 64 x 64 x 32   | 0.797 (-)           | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_3_planes_64x64_synthetic_room.pt)      |
| Synthetic Room | 3 Planes | 3 x 128 x 128 x 32 | 0.835 (0.805)       | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_3_planes_128x128_synthetic_room.pt)    |
| Synthetic Room | 3 Planes | 3 x 256 x 256 x 64 | 0.861 (-)           | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_con_3_planes_256x256x64_synthetic_room.pt) |

### Neural Radiance Fields

| Dataset | Scene     | Link |
| ------- | --------- | ---- |
| Blender | Chair     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_chair.pt)     |
| Blender | Drums     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_drums.pt)     |
| Blender | Ficus     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_ficus.pt)     |
| Blender | Hotdog    | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_hotdog.pt)    |
| Blender | Lego      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_lego.pt)      |
| Blender | Materials | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_materials.pt) |
| Blender | Mic       | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_mic.pt)       |
| Blender | Ship      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_nerf_blender_ship.pt)      |

### Implicit Differentiable Renderer

| Dataset | Scan ID | Link |
| ------- | ------- | ---- |
| DTU MVR | 37      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan37_2000.pt)  |
| DTU MVR | 40      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan40_2000.pt)  |
| DTU MVR | 55      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan55_2000.pt)  |
| DTU MVR | 63      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan63_2000.pt)  |
| DTU MVR | 65      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan65_2000.pt)  |
| DTU MVR | 69      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan69_2000.pt)  |
| DTU MVR | 83      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan83_2000.pt)  |
| DTU MVR | 97      | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan97_2000.pt)  |
| DTU MVR | 105     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan105_2000.pt) |
| DTU MVR | 106     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan106_2000.pt) |
| DTU MVR | 110     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan110_2000.pt) |
| DTU MVR | 114     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan114_2000.pt) |
| DTU MVR | 118     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan118_2000.pt) |
| DTU MVR | 122     | [Download](https://static.preferred.jp/models/pynif3d/pynif3d_idr_scan122_2000.pt) |

# Usage

To use the models with the example scripts, you need to run the following commands:

| Model File | Evaluation Command | 
| ---------- | ------------------ |
| pynif3d_con_grid_32^3_shapenet.pt                 | `python3 examples/con/evaluate.py -dd [SHAPENET_PATH] -m [MODEL_PATH] --mode grid -ds shapenet -fd 64 -nl 3`                |
| pynif3d_con_1plane_64_shapenet.pt                 | `python3 examples/con/evaluate.py -dd [SHAPENET_PATH] -m [MODEL_PATH] --mode single -ds shapenet`                           |
| pynif3d_con_3planes_64_shapenet.pt                | `python3 examples/con/evaluate.py -dd [SHAPENET_PATH] -m [MODEL_PATH] --mode multi -ds shapenet`                            |
| pynif3d_con_grid_64x64x64x32_synthetic_room.pt    | `python3 examples/con/evaluate.py -dd [SYNTHETIC_ROOM_PATH] -m [MODEL_PATH] --mode multi -ds synthetic_room -fd 64 `        |
| pynif3d_con_3_planes_64x64_synthetic_room.pt      | `python3 examples/con/evaluate.py -dd [SYNTHETIC_ROOM_PATH] -m [MODEL_PATH] --mode multi -ds synthetic_room -fd 64`         |
| pynif3d_con_3_planes_128x128_synthetic_room.pt    | `python3 examples/con/evaluate.py -dd [SYNTHETIC_ROOM_PATH] -m [MODEL_PATH] --mode multi -ds synthetic_room -fs 128`        |
| pynif3d_con_3_planes_256x256x64_synthetic_room.pt | `python3 examples/con/evaluate.py -dd [SYNTHETIC_ROOM_PATH] -m [MODEL_PATH] --mode multi -ds synthetic_room -fs 256 -fd 64` |
| pynif3d_nerf_blender_X.pt                         | `python3 examples/nerf/evaluate.py -dd [BLENDER_PATH] -m [MODEL_PATH]`                                                      |
| pynif3d_idr_scanX_2000.pt                         | `python3 examples/idr/evaluate.py -dd [BLENDER_PATH] -m [MODEL_PATH] -s X`                                                  |

