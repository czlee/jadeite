"""Some simple unit tests for bits of code that do quantization."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import functools
import unittest

import torch

from experiments.digital import QuantizationWithEqualBinsMixin


assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


class TestQuantizationMixin(unittest.TestCase):

    def setUp(self):
        self.mixin = QuantizationWithEqualBinsMixin()
        self.mixin.params = self.mixin.default_params_to_add.copy()

    def set_zero_bits_strategy(self, strategy):
        self.mixin.params['zero_bits_strategy'] = strategy

    def set_rounding_method(self, method):
        self.mixin.params['rounding_method'] = method

    def test_quantize_stochastic_basic(self):
        self.set_rounding_method('stochastic')
        values = torch.tensor([0, 3, 8, 0, 3, -1], dtype=torch.float64)
        nbits = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.int64)
        indices = self.mixin.quantize(values, nbits, qrange=5)

        # This sanity check just checks this once. The
        # `test_quantization_stochastically` test below checks that the
        # quantized values differ between calls, since the average can only
        # approximately match the true value if that happens.
        self.assertIn(indices[0], [0, 1])
        self.assertIn(indices[1], [0, 1])
        self.assertIn(indices[2], [1])
        self.assertIn(indices[3], [1, 2])
        self.assertIn(indices[4], [2, 3])
        self.assertIn(indices[5], [2, 3])

    def test_quantize_deterministic_basic(self):
        self.set_rounding_method('deterministic')

        # do this lots of times to make sure it is actually deterministic
        for i in range(30):
            values = torch.tensor([0, 3, 8, 0, 3, -1], dtype=torch.float64)
            nbits = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.int64)
            indices = self.mixin.quantize(values, nbits, qrange=5)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[1], 1)
            self.assertEqual(indices[2], 1)
            self.assertEqual(indices[3], 2)
            self.assertEqual(indices[4], 2)
            self.assertEqual(indices[5], 3)

    def test_unquantize_basic(self):
        indices = torch.tensor([0, 1, 0, 1, 2, 3, 1, 4, 7], dtype=torch.int64)
        nbits = torch.tensor([1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=torch.int64)
        values = self.mixin.unquantize(indices, nbits, qrange=5)
        expected = torch.tensor([-5, 5, -5, -5/3, 5/3, 5, -25/7, 5/7, 5], dtype=torch.float64)  # noqa: E501 E226
        torch.testing.assert_close(values, expected)

    def _test_quantization_stochastically(self, nbits, qrange=10, nsamples=10000):
        """Quantizes and unquantizes a vector lots of times, and checks the
        averages are close to the original values."""
        self.set_rounding_method('stochastic')

        qrange = 10
        assert nbits.ndim == 1
        n = nbits.numel()
        nsamples = 10000
        values = (torch.rand(n) * 2 - 1) * qrange
        quantized = torch.zeros((nsamples, n))
        for i in range(nsamples):
            indices = self.mixin.quantize(values, nbits, qrange)
            self.assertIn(indices.dtype, [torch.int32, torch.int64])
            self.assertTrue(indices.le(2 ** nbits - 1).all())
            self.assertTrue(indices.ge(0).all())
            quantized[i, :] = self.mixin.unquantize(indices, nbits, qrange)
        averages = quantized.mean(axis=0)

        self.assertTrue(averages.le(qrange).all())
        self.assertTrue(averages.ge(-qrange).all())

        # as tolerance, using 8 times the standard deviation of the quantized
        # values (as computed from the scaled binomial distribution)
        deltas = 8 * torch.sqrt(2 * qrange / (2 ** nbits - 1) / 4 / nsamples)

        # except that there always seems to be a residual error less than 5e-6
        # I assume because of some interaction between floating point and fixed
        # point precision, but I'm not quite sure
        deltas = deltas.clip(min=5e-6)

        diffs = torch.abs(values - averages)
        worst_margin, index = ((diffs - deltas) / deltas).max(dim=0)
        self.assertTrue(torch.lt(diffs, deltas).all(),
            msg=f"worst margin was {worst_margin:.4g} ({diffs[index]:.4g} > {deltas[index]:.4g})")

    def test_quantization_stochastically(self):
        n = 50
        nbits = torch.randint(2, 63, size=(n,))
        self._test_quantization_stochastically(nbits)

    def test_quantization_stochastically_fixed_sizes(self):
        """Quantizes and unquantizes a vector lots of times, and checks the
        averages are close to the original values."""
        self.set_rounding_method('stochastic')
        n = 50

        for m in [2, 10, 17, 24, 32, 48, 63]:
            with self.subTest(m=m):
                nbits = torch.full((n,), m)
                self._test_quantization_stochastically(nbits)

    def test_quantize_zero_bits(self):
        values = torch.tensor([-2, 2, 2, -2, 2])
        nbits = torch.tensor([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min-one')
        indices = self.mixin.quantize(values, nbits, qrange=2)
        expected = torch.tensor([0, 1, 1, 0, 1])
        assert_equal(indices, expected)

        self.set_zero_bits_strategy('read-zero')
        indices = self.mixin.quantize(values, nbits, qrange=2)
        expected = torch.tensor([0, 0, 0, 0, 1])
        assert_equal(indices, expected)

        self.set_zero_bits_strategy('exclude')
        indices = self.mixin.quantize(values, nbits, qrange=2)
        expected = torch.tensor([0, 0, 0, 0, 1])
        assert_equal(indices, expected)

    def test_unquantize_zero_bits(self):
        indices = torch.tensor([0, 0, 0, 0, 1])
        nbits = torch.tensor([0, 0, 0, 1, 1])

        self.set_zero_bits_strategy('min-one')
        values = self.mixin.unquantize(indices, nbits, qrange=2)
        expected = torch.tensor([-2, -2, -2, -2, 2], dtype=torch.float64)
        assert_equal(values, expected)

        self.set_zero_bits_strategy('read-zero')
        values = self.mixin.unquantize(indices, nbits, qrange=2)
        expected = torch.tensor([0, 0, 0, -2, 2], dtype=torch.float64)
        assert_equal(values, expected)

        self.set_zero_bits_strategy('exclude')
        values = self.mixin.unquantize(indices, nbits, qrange=2)
        expected = torch.tensor([float('nan'), float('nan'), float('nan'), -2, 2], dtype=torch.float64)
        # nan != nan, so test this separately
        self.assertTrue(values[:3].isnan().all())
        assert_equal(values[3:], expected[3:])

    def test_client_average_zero_bits(self):
        nbits = [
            torch.tensor([0, 1, 0, 0, 1, 0]),
            torch.tensor([0, 1, 1, 1, 1, 0]),
            torch.tensor([1, 1, 1, 0, 0, 0]),
        ]
        values = [
            torch.tensor([-2, +2, +2, -2, +2, -2], dtype=torch.float64),
            torch.tensor([+2, -2, -2, +2, +2, -2], dtype=torch.float64),
            torch.tensor([+2, +2, -2, -2, -2, +2], dtype=torch.float64),
        ]

        # average using zero for no-bit values
        self.set_zero_bits_strategy('read-zero')
        quantized = [self.mixin.quantize(v, n, qrange=2) for v, n in zip(values, nbits)]
        unquantized = [self.mixin.unquantize(i, n, qrange=2) for i, n in zip(quantized, nbits)]
        average = self.mixin.compute_client_average(unquantized)
        expected = torch.tensor([+2, +2, -4, +2, +4, 0], dtype=torch.float64) / 3
        assert_equal(average, expected)

        # average excluding no-bit values
        self.set_zero_bits_strategy('exclude')
        quantized = [self.mixin.quantize(v, n, qrange=2) for v, n in zip(values, nbits)]
        unquantized = [self.mixin.unquantize(i, n, qrange=2) for i, n in zip(quantized, nbits)]
        average = self.mixin.compute_client_average(unquantized)
        expected = torch.tensor([+2, +2 / 3, -2, +2, +2, 0], dtype=torch.float64)
        assert_equal(average, expected)
