# License under an MIT style license -- see LICENSE.md

import os
import shutil
import numpy as np
from ligo.em_bright.em_bright import source_classification_pe
from .base import make_result_file
from pesummary.io import read
from pesummary.utils.decorators import no_latex_plot
from pesummary.gw.classification import Classify, PEPredicates, PAstro

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
    def test_plotting(self, cls=PEPredicates):
        """Test the .plot method
        """
        import matplotlib.figure
        samples = read(".outdir/test_converted.dat").samples_dict
        _cls = cls(samples)
        for plot in _cls.available_plots:
            fig = _cls.plot(type=plot)
            assert isinstance(fig, matplotlib.figure.Figure)


class TestPEPredicates(_Base):
    """Test the pesummary.gw.classification.PEPredicates class and
    pesummary.gw.classification.Classify class
    """
    def test_classification(self):
        """Test the base classification method agrees with the
        pepredicates.predicate_table function
        """
        from pepredicates import (
            predicate_table, BNS_p, NSBH_p, BBH_p, MG_p
        )
        from pandas import DataFrame
        samples = read(".outdir/test_converted.dat").samples_dict
        pesummary = PEPredicates(samples).classification()
        pesummary2 = PEPredicates.classification_from_file(
            ".outdir/test_converted.dat"
        )
        pesummary3 = Classify.classification_from_file(
            ".outdir/test_converted.dat"
        )
        pesummary4 = Classify.dual_classification_from_file(
            ".outdir/test_converted.dat"
        )["default"]
        df = DataFrame.from_dict(
            {
                "m1_source": samples["mass_1_source"],
                "m2_source": samples["mass_2_source"],
            }
        )
        probs = predicate_table(
            {"BNS": BNS_p, "BBH": BBH_p, "MassGap": MG_p, "NSBH": NSBH_p},
            df
        )
        for key in probs.keys():
            np.testing.assert_almost_equal(probs[key], pesummary[key])
            np.testing.assert_almost_equal(probs[key], pesummary2[key])
            np.testing.assert_almost_equal(probs[key], pesummary3[key])
            np.testing.assert_almost_equal(probs[key], pesummary4[key])


class TestPAstro(_Base):
    """Test the pesummary.gw.classification.PAstro class and
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
        print(samples)
        pesummary = PAstro(samples).classification()
        np.testing.assert_almost_equal(p_astro[0], pesummary["HasNS"])
        np.testing.assert_almost_equal(p_astro[1], pesummary["HasRemnant"])
        pesummary2 = PAstro.classification_from_file(
            ".outdir/test_converted.dat"
        )
        pesummary3 = Classify.classification_from_file(
            ".outdir/test_converted.dat"
        )
        pesummary4 = Classify.dual_classification_from_file(
            ".outdir/test_converted.dat"
        )["default"]
        for key, val in pesummary2.items():
            np.testing.assert_almost_equal(pesummary[key], val)
            np.testing.assert_almost_equal(pesummary[key], pesummary3[key])
            np.testing.assert_almost_equal(pesummary[key], pesummary4[key])

    def test_reweight_classification(self):
        """Test that the population reweighted classification method agrees
        with the ligo.em_bright.source_classification_pe function.
        """
        from pepredicates import rewt_approx_massdist_redshift
        from pesummary.gw.conversions import mchirp_from_m1_m2, q_from_m1_m2
        from pandas import DataFrame

        rerun = True
        while rerun:
            samples = read(".outdir/test_converted.dat").samples_dict
            df = DataFrame.from_dict(
                dict(
                    m1_source=samples["mass_1_source"],
                    m2_source=samples["mass_2_source"],
                    a1=samples["a_1"],
                    a2=samples["a_2"],
                    dist=samples["luminosity_distance"],
                    redshift=samples["redshift"]
                )
            ) 
            df["mc_source"] = mchirp_from_m1_m2(df["m1_source"], df["m2_source"])
            df["q"] = q_from_m1_m2(df["m1_source"], df["m2_source"])
            _reweighted_samples = rewt_approx_massdist_redshift(df)
            # ligo.em_bright.source_classification_pe fails if there is only
            # one sample
            if len(_reweighted_samples["m1_source"]) != 1:
                rerun = False
            else:
                self.setup_method()
        _reweighted_samples.to_csv(
            ".outdir/test_reweighted.dat", sep=" ", index=False
        )
        reweighted_file = read(".outdir/test_reweighted.dat")
        reweighted_file.write(
            filename="test_reweighted.hdf5", file_format="hdf5", outdir=".outdir",
            overwrite=True
        )
        p_astro = source_classification_pe(
            ".outdir/test_reweighted.hdf5"
        )
        _samples, pesummary = PAstro(samples).classification(
            population=True, return_samples=True
        )
        np.testing.assert_almost_equal(p_astro[0], pesummary["HasNS"])
        np.testing.assert_almost_equal(p_astro[1], pesummary["HasRemnant"])
        pesummary2 = PAstro.classification_from_file(
            ".outdir/test_converted.dat", population=True
        )
        pesummary3 = Classify.classification_from_file(
            ".outdir/test_converted.dat", population=True
        )
        pesummary4 = Classify.dual_classification_from_file(
            ".outdir/test_converted.dat"
        )["population"]
        for key, val in pesummary2.items():
            np.testing.assert_almost_equal(pesummary[key], val)
            np.testing.assert_almost_equal(pesummary[key], pesummary3[key])
            np.testing.assert_almost_equal(pesummary[key], pesummary4[key])

    def test_plotting(self):
        """Test the .plot method
        """
        super(TestPAstro, self).test_plotting(cls=PAstro)
