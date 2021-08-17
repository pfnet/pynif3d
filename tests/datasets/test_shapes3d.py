from unittest import TestCase

from pynif3d.datasets import Shapes3dDataset


class TestShapes3d(TestCase):
    def test_forward(self):
        # Load dataset
        data_directory = "/tmp/datasets/shapenet"

        train_set = Shapes3dDataset(data_directory, "train")
        val_set = Shapes3dDataset(data_directory, "val", in_points_sample_count=10000)
        test_set = Shapes3dDataset(data_directory, "test", gt_point_sample_count=1234)

        # Check if it can be loaded
        train_sample = train_set[0]
        val_sample = val_set[0]
        test_sample = test_set[0]

        # Check size
        self.assertEqual(len(train_set), 2832)
        self.assertEqual(len(val_set), 404)
        self.assertEqual(len(test_set), 809)

        # Check subsampling
        self.assertEqual(len(train_sample["query_points"]), 2048)
        self.assertEqual(len(val_sample["in_points"]), 10000)
        self.assertEqual(len(val_sample["in_normals"]), 10000)
        self.assertEqual(len(test_sample["query_points"]), 1234)
        self.assertEqual(len(test_sample["gt_occupancies"]), 1234)
