from math import sqrt

import torch
import pytest

from model_runner import CustomLoss


def is_close(x, y):
    tol = 1e-6
    if isinstance(x, torch.Tensor):
        x = x.item()
    return abs(x - y) < tol


def test_wrong_dimensions_raise_AssertionError():
    good = torch.tensor([3.0, 1, 5, 8]).unsqueeze(0)
    bad = torch.tensor([1.0, 2, 3, 4])
    loss = CustomLoss(N_dipoles=1)
    with pytest.raises(AssertionError):
        loss(good, bad)
    with pytest.raises(AssertionError):
        loss(bad, good)


def test_custom_returns_torch_float():
    loss = CustomLoss(N_dipoles=1)
    t_1 = torch.tensor([[1.0, 2, 3, 4]])
    t_2 = torch.tensor([[3.0, 1, 5, 8]])
    l = loss(t_1, t_2)
    assert isinstance(l,  torch.Tensor)
    assert l.dtype == torch.float32

    t_1 = torch.tensor([[1.0, 2, 3, 4], [1, 2, 3, 4]])
    t_2 = torch.tensor([[3.0, 1, 5, 8], [1, 2, 3, 4]])
    l = loss(t_1, t_2)
    assert isinstance(l,  torch.Tensor)
    assert l.dtype == torch.float32


def test_custom_1dipole_1sample():
    loss = CustomLoss(N_dipoles=1)

    t_1 = torch.tensor([[1.0, 2, 3, 4]])
    assert loss(t_1, t_1) == 0

    t_2 = torch.tensor([[3.0, 1, 5, 8]])
    assert loss(t_1, t_2) == 7
    assert loss(t_2, t_1) == 7

    t_2 = torch.tensor([[4.0, -2, -9, 2]])
    assert loss(t_1, t_2) == 15
    assert loss(t_2, t_1) == 15


def test_custom_1dipole_2samples():
    loss = CustomLoss(N_dipoles=1)
    t_1 = torch.tensor([[1.0, 2, 3, 4], [-3, -2, -1, 2]])
    t_2 = torch.tensor([[4.0, -1, 3, 0], [-2, 0, 5, 5]])
    exact = 0.5*(7 + sqrt(18) + sqrt(41))
    assert is_close(loss(t_1, t_2), exact)
    assert is_close(loss(t_2, t_1), exact)


def test_custom_2dipoles_1sample():
    loss = CustomLoss(N_dipoles=2)
    t_1 = torch.tensor([[0.0, 0, 0, 1, 5, 0, 0, 2]])
    t_1_swapped = torch.tensor([[5.0, 0, 0, 2, 0, 0, 0, 1]])
    t_2 = torch.tensor([[1.0, 0, 0, 0, 2, 0, 0, 0]])
    t_2_swapped = torch.tensor([[2.0, 0, 0, 0, 1, 0, 0, 0]])
    exact = 7
    assert loss(t_1, t_2) == exact
    assert loss(t_1_swapped, t_2) == exact
    assert loss(t_1, t_2_swapped) == exact
    assert loss(t_1_swapped, t_2_swapped) == exact

    loss = CustomLoss(N_dipoles=2)
    t_1 = torch.tensor([[1.0, 2, 3, 4, 5, 1, 2, 3, 5, 10]])
    t_1_swapped = torch.tensor([[1, 2, 3, 5, 10, 1.0, 2, 3, 4, 5]])
    t_2 = torch.tensor([[1.0, 2, 3, 6, 5, 1, 2, 3, 4, 9]])
    t_2_swapped = torch.tensor([[1, 2, 3, 4, 9, 1.0, 2, 3, 6, 5]])
    exact = 4
    assert loss(t_1, t_2) == exact
    assert loss(t_1_swapped, t_2) == exact
    assert loss(t_1, t_2_swapped) == exact
    assert loss(t_1_swapped, t_2_swapped) == exact


def test_custom_1dipole_1sample_radius():
    loss = CustomLoss(N_dipoles=1)

    t_1 = torch.tensor([[1.0, 2, 3, 4, 5]])
    assert loss(t_1, t_1) == 0

    t_2 = torch.tensor([[3.0, 1, 5, 8, 2]])
    assert loss(t_1, t_2) == 10
    assert loss(t_2, t_1) == 10

    t_2 = torch.tensor([[4.0, -2, -9, 2, 0]])
    assert loss(t_1, t_2) == 20
    assert loss(t_2, t_1) == 20


def test_custom_1dipole_2samples_radius():
    loss = CustomLoss(N_dipoles=1)
    t_1 = torch.tensor([[1.0, 2, 3, 4, 0.2], [-3, -2, -1, 2, 4.6]])
    t_2 = torch.tensor([[4.0, -1, 3, 0, 1.1], [-2, 0, 5, 5, 1.4]])
    exact = 0.5*(7 + sqrt(18) + sqrt(41) + 0.9 + 3.2)
    assert is_close(loss(t_1, t_2), exact)
    assert is_close(loss(t_2, t_1), exact)


# TODO:
    # - different positions of multiple dipoles
    # - 2 dipoles 2 samples
    # - the same with radius
    # - multiple dipoles with radius


if __name__ == '__main__':
    test_custom_1dipole_2samples()