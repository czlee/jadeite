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

    def set_zero_bits_strategy(self, strategy):
        self.mixin.params['zero_bits_strategy'] = strategy

    def test_quantize_basic(self):
        """Simple sanity checks."""
        values = torch.tensor([0, 3, 8, 0, 4, -1], dtype=torch.float64)
        nbits = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.int64)
        indices = self.mixin.quantize(values, nbits, qrange=5)
        self.assertIn(indices[0], [0, 1])
        self.assertIn(indices[1], [0, 1])
        self.assertIn(indices[2], [1])
        self.assertIn(indices[3], [1, 2])
        self.assertIn(indices[4], [2, 3])
        self.assertIn(indices[5], [2, 3])

    def test_unquantize_basic(self):
        """Simple sanity checks."""
        indices = torch.tensor([0, 1, 0, 1, 2, 3, 1, 4, 7], dtype=torch.int64)
        nbits = torch.tensor([1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=torch.int64)
        values = self.mixin.unquantize(indices, nbits, qrange=5)
        expected = torch.tensor([-5, 5, -5, -5/3, 5/3, 5, -25/7, 5/7, 5], dtype=torch.float64)  # noqa: E501 E226
        torch.testing.assert_close(values, expected)

    def test_quantization_stochastically(self):
        """Quantizes and unquantizes a vector lots of times, and checks the
        averages are close to the original values."""
        qrange = 10
        n = 50
        nsamples = 10000
        values = (torch.rand(n) * 2 - 1) * qrange
        nbits = torch.randint(2, 8, size=(n,))
        quantized = torch.zeros((nsamples, n))
        for i in range(nsamples):
            indices = self.mixin.quantize(values, nbits, qrange)
            quantized[i, :] = self.mixin.unquantize(indices, nbits, qrange)
        averages = quantized.mean(axis=0)

        # as tolerance, using 8 times the standard deviation of the quantized
        # values (as computed from the scaled binomial distribution)
        deltas = 8 * torch.sqrt(2 * qrange / (2 ** nbits - 1) / 4 / nsamples)

        self.assertTrue(averages.le(qrange).all())
        self.assertTrue(averages.ge(-qrange).all())

        diffs = torch.abs(values - averages)
        self.assertTrue(torch.lt(diffs, deltas).all())

    def test_quantize_zero_bits(self):
        values = torch.tensor([-2, 2, 2, -2, 2])
        nbits = torch.tensor([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min-one')
        indices = self.mixin.quantize(values, nbits, qrange=2)
        expected = torch.tensor([0, 1, 1, 0, 1])
        torch.testing.assert_equal(indices, expected)

        self.set_zero_bits_strategy('read-zero')
        indices = self.mixin.quantize(values, nbits, qrange=2)
        expected = torch.tensor([0, 0, 0, 0, 1])
        torch.testing.assert_equal(indices, expected)

    def test_unquantize_zero_bits(self):
        indices = torch.tensor([0, 0, 0, 0, 1])
        nbits = torch.tensor([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min-one')
        values = self.mixin.unquantize(indices, nbits, qrange=2)
        expected = torch.tensor([-2, -2, -2, -2, 2], dtype=torch.float64)
        torch.testing.assert_equal(values, expected)

        self.set_zero_bits_strategy('read-zero')
        values = self.mixin.unquantize(indices, nbits, qrange=2)
        expected = torch.tensor([0, 0, 0, -2, 2], dtype=torch.float64)
        torch.testing.assert_equal(values, expected)
