import unittest

import numpy as np

from signal_design.axis import Axis
from signal_design.help_types import Number


class TestMethodsToCreateAxis(unittest.TestCase):
    def check_axis(
        self,
        axis: Axis,
        x_start: Number,
        x_end: Number,
        sample: Number,
        size: int,
    ):
        self.assertEqual(axis.start, x_start)
        self.assertEqual(axis.end, x_end)
        self.assertEqual(axis.sample, sample)
        self.assertEqual(axis.size, size)

    def test_size_axis(self):
        with self.assertRaises(ValueError):
            Axis(-1)

        with self.assertRaises(ValueError):
            Axis(0)

        axis = Axis(1)
        self.assertEqual(axis.size, 1)

    def test_init_axis(self):

        start = -1.0
        sample = 0.1
        size = 101

        end = 9

        axis = Axis(size, sample, start)

        self.assertEqual(axis.start, start)
        self.assertEqual(axis.sample, sample)
        self.assertEqual(axis.size, size)
        self.assertEqual(axis.end, end)

    def test_from_start_end_sample(self):

        start = -1.0
        end = 9.0
        sample = 0.1
        size = 101
        axis = Axis.get_using_end(start, end, sample)

        self.assertEqual(axis.size, size)
        self.assertEqual(axis.start, start)
        self.assertEqual(axis.end, end)
        self.assertEqual(axis.start, start)
        self.assertEqual(axis.sample, sample)

    def test_get_axis_from_list(self):

        int_array1 = [2, 3, 4, 5, 6, 7]
        int_axis_1 = Axis.get_from_array(int_array1, 1)
        self.check_axis(int_axis_1, 2, 7, 1, len(int_array1))

        int_array2 = [-5, 0, 5, 10, 15, 20, 25]
        int_axis_2 = Axis.get_from_array(int_array2, 5)
        self.check_axis(int_axis_2, -5, 25, 5, len(int_array2))

        float_array1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        float_axis1 = Axis.get_from_array(float_array1, 0.1)
        self.check_axis(float_axis1, 0, 0.5, 0.1, len(float_array1))

        float_array2 = [-5.5, -2.5, 0.5, 3.5]
        float_axis2 = Axis.get_from_array(float_array2, 3.0)
        self.check_axis(float_axis2, -5.5, 3.5, 3.0, len(float_array2))

    def test_bad_end(self):

        start = -1.0
        sample = 0.5
        end = 3.25
        actual_end = 3.25

        axis = Axis.get_using_end(start, end, sample)
        self.assertEqual(axis.array[0], start)
        self.assertEqual(axis.sample, sample)
        self.assertEqual(axis.array[-1], actual_end)
        self.assertEqual(axis.size, 9)

    def test_get_axis_from_np_array(self):

        dx = 0.2
        x_start = -3.0
        x_end = 5.6
        num_sample = int((x_end - x_start) / dx) + 2
        np_array = np.linspace(-3.0, 5.6, num_sample)
        np_axis = Axis.get_from_array(np_array, dx)
        self.check_axis(np_axis, x_start, x_end, dx, np_array.size)

        dx = 1.5
        x_start = -2.5
        x_end = 2.0
        num_sample = int((x_end - x_start) / dx) + 1
        np_array = np.linspace(x_start, x_end, num_sample)
        axis = Axis.get_from_array(np_array, dx)
        self.check_axis(axis, x_start, x_end, dx, np_array.size)

    def test_changes_array_axis(self):
        array_axis = Axis.get_using_end(start=0.0, end=10.0, sample=0.1)

        start_array_axis = array_axis.copy()
        start_array_axis.start = 1.0
        self.assertNotEqual(start_array_axis.end, array_axis.end)

        size_array_axis = array_axis.copy()
        size_array_axis.size = 10
        self.assertNotEqual(size_array_axis.end, array_axis.end)

        sample_array_axis = array_axis.copy()
        sample_array_axis.sample = 2.0
        self.assertNotEqual(sample_array_axis.end, array_axis.end)

    def test_samples(self):

        start = -1.0
        samples = [sample / 100 for sample in range(1, 1000)]

        for sample in samples:
            with self.subTest(sample=sample):
                axis = Axis.get_using_end(start, start + 10 * sample, sample)
                check_axis = Axis.get_from_array(axis.array, sample)
                self.assertEqual(check_axis.sample, sample)
