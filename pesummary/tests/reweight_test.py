# Licensed under an MIT style license -- see LICENSE.md

import pytest
import numpy as np
from ..core.reweight import rejection_sampling
from ..gw.reweight import uniform_in_comoving_volume_from_uniform_in_volume
from ..utils.samples_dict import SamplesDict
from .base import gw_parameters

n_samples = 100
n_params = 10


def test_rejection_sampling():
    """Test that the pesummary.core.reweight.rejection_sampling works as
    expected
    """
    # Check that it works with a numpy array
    original_samples = np.random.uniform(0, 10, (n_samples, n_params))
    weights = np.random.uniform(0, 5, n_samples)
    new_samples = rejection_sampling(original_samples, weights)
    # new_samples should have less samples than what we started with originally
    assert len(new_samples) <= n_samples
    # Each sample should be in the original posterior table
    assert all(new_sample in original_samples for new_sample in new_samples)
    # Each sample should be unique
    unique = np.unique(new_samples, axis=0)
    assert len(unique) == len(new_samples)

    # Now check that it works as expected for the
    # pesummary.utils.samples_dict.SamplesDict object
    original_samples = SamplesDict(
        {param: np.random.uniform(0, 10, n_samples) for param in gw_parameters()}
    )
    weights = np.random.uniform(0, 5, n_samples)
    new_samples = rejection_sampling(original_samples, weights)
    assert new_samples.number_of_samples <= original_samples.number_of_samples
    assert new_samples.parameters == original_samples.parameters
    assert all(
        new_sample in original_samples.samples.T for new_sample in
        new_samples.samples.T
    )


def test_uniform_in_comoving_volume_from_uniform_in_volume():
    """Test that the
    pesummary.gw.reweight.uniform_in_comoving_volume_from_uniform_in_volume
    function works as expected
    """
    original_samples = SamplesDict(
        {param: np.random.uniform(0, 10, n_samples) for param in gw_parameters()}
    )
    new_samples = uniform_in_comoving_volume_from_uniform_in_volume(
        original_samples
    )
    assert new_samples.number_of_samples <= original_samples.number_of_samples
    assert all(
        new_sample in original_samples.samples.T for new_sample in
        new_samples.samples.T
    )
    # check that if there are no redshift samples it still reweights
    original_samples.pop("redshift")
    new_samples = uniform_in_comoving_volume_from_uniform_in_volume(
        original_samples
    )
    assert new_samples.number_of_samples <= original_samples.number_of_samples
    assert all(
        new_sample in original_samples.samples.T for new_sample in
        new_samples.samples.T
    )
    # check that if there are no distance samples it still reweights
    original_samples = SamplesDict(
        {param: np.random.uniform(0, 10, n_samples) for param in gw_parameters()}
    )
    original_samples.pop("luminosity_distance")
    new_samples = uniform_in_comoving_volume_from_uniform_in_volume(
        original_samples
    )
    assert new_samples.number_of_samples <= original_samples.number_of_samples
    assert all(
        new_sample in original_samples.samples.T for new_sample in
        new_samples.samples.T
    )
    # check that if there are no redshift or distance samples it fails
    original_samples.pop("redshift")
    with pytest.raises(Exception):
        new_samples = uniform_in_comoving_volume_from_uniform_in_volume(
            original_samples
        )
