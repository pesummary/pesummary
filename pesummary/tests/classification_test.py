# License under an MIT style license -- see LICENSE.md

import os
import shutil
import numpy as np
from ligo.em_bright.em_bright import source_classification_pe
from .base import make_result_file, testing_dir
from pesummary.io import read
from pesummary.utils.decorators import no_latex_plot
from pesummary.gw.classification import Classify, EMBright, PAstro

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class _Base(object):
    """Base testing class
    """
    def setup_method(self):
        """Setup the class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        make_result_file(gw=True, extension="dat", outdir=".outdir/")
        f = read(".outdir/test.dat")
        # regenerate the mass_1_source, mass_2_source posteriors because these
        # are randomly chosen and do not correspond to
        # mass_1 / (1. + z) and mass_2 / (1. + z)
        f.generate_all_posterior_samples(
            regenerate=["mass_1_source", "mass_2_source"]
        )
        f.write(
            filename="test_converted.dat", file_format="dat", outdir=".outdir",
            overwrite=True
        )
        f.write(
            filename="test_lalinference.hdf5", file_format="hdf5",
            outdir=".outdir", overwrite=True
        )

    def teardown_method(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    @no_latex_plot
    def test_plotting(self, cls=EMBright, **kwargs):
        """Test the .plot method
        """
        import matplotlib.figure
        samples = read(".outdir/test_converted.dat").samples_dict
        _cls = cls(samples, **kwargs)
        ptable = _cls.classification()
        for plot in _cls.available_plots:
            fig = _cls.plot(ptable, type=plot)
            assert isinstance(fig, matplotlib.figure.Figure)


class TestPAstro(_Base):
    """Test the pesummary.gw.classification.PAstro class and
    pesummary.gw.classification.Classify class
    """
    def test_classification(self):
        """Test the base classification method agrees with the
        pepredicates.predicate_table function
        """
        pass

    def test_plotting(self):
        """Test the .plot method
        """
        import pytest
        # catch ValueError when no terrestrial probability is given
        with pytest.raises(ValueError):
            super(TestPAstro, self).test_plotting(
                cls=PAstro, category_data=f"{testing_dir}/rates.yml",
            )
        super(TestPAstro, self).test_plotting(
            cls=PAstro, category_data=f"{testing_dir}/rates.yml",
            terrestrial_probability=0.
        )


class TestEMBright(_Base):
    """Test the pesummary.gw.classification.EMBright class and
    pesummary.gw.classification.Classify class
    """
    def test_classification(self):
        """Test the base classification method agrees with the
        ligo.em_bright.source_classification_pe function
        """
        p_astro = source_classification_pe(
            ".outdir/test_lalinference.hdf5"
        )
        samples = read(".outdir/test_converted.dat").samples_dict
        pesummary = EMBright(samples).classification()
        np.testing.assert_almost_equal(p_astro[0], pesummary["HasNS"], 5)
        np.testing.assert_almost_equal(p_astro[1], pesummary["HasRemnant"], 5)
        pesummary2 = EMBright.classification_from_file(
            ".outdir/test_converted.dat"
        )
        pesummary3 = Classify.classification_from_file(
            ".outdir/test_converted.dat",
            category_data=f"{testing_dir}/rates.yml",
            terrestrial_probability=0.
        )
        for key, val in pesummary2.items():
            np.testing.assert_almost_equal(pesummary[key], val, 5)
            np.testing.assert_almost_equal(pesummary[key], pesummary3[key], 5)

    def test_plotting(self):
        """Test the .plot method
        """
        super(TestEMBright, self).test_plotting(cls=EMBright)
