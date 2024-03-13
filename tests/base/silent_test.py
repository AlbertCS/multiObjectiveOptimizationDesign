import os
import unittest
from unittest.mock import MagicMock, patch

from pyrosetta import Pose, init, rosetta

from mood.base import Silent


class TestSilent(unittest.TestCase):
    def setUp(self):
        init()
        self.pose = Pose()
        self.silent = Silent("test.silent")

    def test_init(self):
        self.assertEqual(self.silent.silent_file_path, "test.silent")
        self.assertEqual(self.silent.tags, [])

    @patch("silent_file.rosetta.core.io.silent.BinarySilentStruct")
    def test_write_pose(self, mock_struct):
        self.silent.write_pose(self.pose, "test_pose")
        self.assertIn("test_pose", self.silent.tags)

    def test_get_pose_from_tag(self):
        self.silent.write_pose(self.pose, "test_pose")
        pose = self.silent.get_pose_from_tag("test_pose")
        self.assertIsInstance(pose, Pose)

    def test_get_tags(self):
        self.silent.write_pose(self.pose, "test_pose")
        tags = self.silent.get_tags()
        self.assertIn("test_pose", tags)

    def test_read_poses_from_silent_file(self):
        self.silent.write_pose(self.pose, "test_pose")
        poses = self.silent.read_poses_from_silent_file()
        self.assertEqual(len(poses), 1)
        self.assertIsInstance(poses[0], Pose)

    @patch("os.mkdir")
    def test_extract_poses(self, mock_mkdir):
        self.silent.write_pose(self.pose, "test_pose")
        self.silent.extract_poses("output_folder")
        self.assertTrue(os.path.exists("output_folder/test_pose.pdb"))

    def tearDown(self):
        os.remove("test.silent")
        os.remove("output_folder/test_pose.pdb")


if __name__ == "__main__":
    unittest.main()
