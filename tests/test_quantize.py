"""Some simple unit tests for bits of code that do quantization."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import unittest

import torch

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
        values = torch.tensor([0, 3, 8, 0, 4, -1], dtype=torch.float64)
        nbits = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.int64)
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
        indices = torch.tensor([0, 1, 0, 1, 2, 3, 1, 4, 7], dtype=torch.int64)
        nbits = torch.tensor([1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=torch.int64)
        values = self.mixin.unquantize(indices, nbits)
        expected = torch.tensor([-5, 5, -5, -5/3, 5/3, 5, -25/7, 5/7, 5], dtype=torch.float64)  # noqa: E501 E226
        torch.testing.assert_close(values, expected)

    def test_quantization_stochastically(self):
        """Quantizes and unquantizes a vector lots of times, and checks the
        averages are close to the original values."""
        M = 10  # noqa: N806
        n = 50
        nsamples = 10000
        self.set_quantization_range(M)
        values = (torch.rand(n) * 2 - 1) * M
        nbits = torch.randint(2, 8, size=(n,))
        quantized = torch.zeros((nsamples, n))
        for i in range(nsamples):
            indices = self.mixin.quantize(values, nbits)
            quantized[i, :] = self.mixin.unquantize(indices, nbits)
        averages = quantized.mean(axis=0)

        # as tolerance, using 8 times the standard deviation of the quantized
        # values (as computed from the scaled binomial distribution)
        deltas = 8 * torch.sqrt(2 * M / (2 ** nbits - 1) / 4 / nsamples)

        self.assertTrue(averages.le(M).all())
        self.assertTrue(averages.ge(-M).all())

        diffs = torch.abs(values - averages)
        self.assertTrue(torch.lt(diffs, deltas).all())

    def test_quantize_zero_bits(self):
        self.set_quantization_range(2)
        values = torch.tensor([-2, 2, 2, -2, 2])
        nbits = torch.tensor([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min-one')
        indices = self.mixin.quantize(values, nbits)
        expected = torch.tensor([0, 1, 1, 0, 1])
        torch.testing.assert_equal(indices, expected)

        self.set_zero_bits_strategy('read-zero')
        indices = self.mixin.quantize(values, nbits)
        expected = torch.tensor([0, 0, 0, 0, 1])
        torch.testing.assert_equal(indices, expected)

    def test_unquantize_zero_bits(self):
        self.set_quantization_range(2)
        indices = torch.tensor([0, 0, 0, 0, 1])
        nbits = torch.tensor([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min-one')
        values = self.mixin.unquantize(indices, nbits)
        expected = torch.tensor([-2, -2, -2, -2, 2], dtype=torch.float64)
        torch.testing.assert_equal(values, expected)

        self.set_zero_bits_strategy('read-zero')
        values = self.mixin.unquantize(indices, nbits)
        expected = torch.tensor([0, 0, 0, -2, 2], dtype=torch.float64)
        torch.testing.assert_equal(values, expected)
