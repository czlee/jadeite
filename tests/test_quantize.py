"""Some simple unit tests for bits of code that do quantization."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import unittest

import numpy as np

from experiments.digital import SimpleStochasticQuantizationMixin


class TestSimpleStochasticQuantizationMixin(unittest.TestCase):

    def setUp(self):
        self.mixin = SimpleStochasticQuantizationMixin()
        self.mixin.params = self.mixin.default_params_to_add.copy()

    def set_quantization_range(self, m):
        self.mixin.params['quantization_range'] = m

    def set_zero_bits_strategy(self, strategy):
        self.mixin.params['zero_bits_strategy'] = strategy

    def test_quantize_basic(self):
        """Simple sanity checks."""
        self.set_quantization_range(5)
        values = np.array([0, 3, 8, 0, 4, -1], dtype=float)
        nbits = np.array([1, 1, 1, 2, 2, 3], dtype=int)
        indices = self.mixin.quantize(values, nbits)
        self.assertIn(indices[0], [0, 1])
        self.assertIn(indices[1], [0, 1])
        self.assertIn(indices[2], [1])
        self.assertIn(indices[3], [1, 2])
        self.assertIn(indices[4], [2, 3])
        self.assertIn(indices[5], [2, 3])

    def test_unquantize_basic(self):
        """Simple sanity checks."""
        self.set_quantization_range(5)
        indices = np.array([0, 1, 0, 1, 2, 3, 1, 4, 7], dtype=int)
        nbits = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=int)
        values = self.mixin.unquantize(indices, nbits)
        expected = np.array([-5, 5, -5, -5/3, 5/3, 5, -25/7, 5/7, 5])  # noqa: E226 (space around /)
        np.testing.assert_allclose(values, expected)

    def test_quantization_stochastically(self):
        """Quantizes and unquantizes a vector lots of times, and checks the
        averages are close to the original values."""
        M = 10  # noqa: N806
        n = 50
        nsamples = 10000
        self.set_quantization_range(M)
        values = (np.random.rand(n) * 2 - 1) * M
        nbits = np.random.randint(2, 8, size=(n,))
        quantized = np.zeros((nsamples, n))
        for i in range(nsamples):
            indices = self.mixin.quantize(values, nbits)
            quantized[i, :] = self.mixin.unquantize(indices, nbits)
        averages = quantized.mean(axis=0)

        # as tolerance, using 8 times the standard deviation of the quantized
        # values (as computed from the scaled binomial distribution)
        deltas = 8 * np.sqrt(2 * M / (2 ** nbits - 1) / 4 / nsamples)

        np.testing.assert_array_less(averages, M)
        np.testing.assert_array_less(-M, averages)

        diffs = np.abs(values - averages)
        np.testing.assert_array_less(diffs, deltas)

    def test_quantize_zero_bits(self):
        self.set_quantization_range(2)
        values = np.array([-2, 2, 2, -2, 2])
        nbits = np.array([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min_one')
        indices = self.mixin.quantize(values, nbits)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(indices, expected)

        self.set_zero_bits_strategy('read_zero')
        indices = self.mixin.quantize(values, nbits)
        expected = np.array([0, 0, 0, 0, 1])
        np.testing.assert_array_equal(indices, expected)

    def test_unquantize_zero_bits(self):
        self.set_quantization_range(2)
        indices = np.array([0, 0, 0, 0, 1])
        nbits = np.array([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min_one')
        values = self.mixin.unquantize(indices, nbits)
        expected = np.array([-2, -2, -2, -2, 2], dtype=float)
        np.testing.assert_array_equal(values, expected)

        self.set_zero_bits_strategy('read_zero')
        values = self.mixin.unquantize(indices, nbits)
        expected = np.array([0, 0, 0, -2, 2], dtype=float)
        np.testing.assert_array_equal(values, expected)
