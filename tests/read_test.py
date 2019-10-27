import os
import shutil
import numpy as np

from base import make_result_file
import pesummary
from pesummary.gw.file.read import read as GWRead
from pesummary.core.file.read import read as Read


class BaseRead(object):
    """Base class to test the core functions in the Read and GWRead functions
    """
    def test_parameters(self, true, pesummary=False):
        """Test the parameter property
        """
        if pesummary:
            assert all(i in self.result.parameters[0] for i in true)
            assert all(i in true for i in self.result.parameters[0])
        else:
            assert all(i in self.result.parameters for i in true)
            assert all(i in true for i in self.result.parameters)

    def test_samples(self, true, pesummary=False):
        """Test the samples property
        """
        if pesummary:
            assert len(self.result.samples[0]) == 1000
            assert len(self.result.samples[0][0]) == 15
            true_flat = [item for sublist in true for item in sublist]
            flat = [item for sublist in self.result.samples[0] for item in sublist]
            assert all(i in true_flat for i in flat)
            assert all(i in flat for i in true_flat)
        else:
            assert len(self.result.samples) == 1000
            assert len(self.result.samples[0]) == 15
            true_flat = [item for sublist in true for item in sublist]
            flat = [item for sublist in self.result.samples for item in sublist]
            assert all(i in true_flat for i in flat)
            assert all(i in flat for i in true_flat)

    def test_samples_dict(self, true):
        """Test the samples_dict property
        """
        parameters = true[0]
        samples = true[1]
        for num, param in enumerate(parameters):
            specific_samples = [i[num] for i in samples]
            drawn_samples = self.result.samples_dict[param]
            assert all(i == j for i, j in zip(drawn_samples, specific_samples))

    def test_version(self, true=None):
        """Test the version property
        """
        if true is None:
            assert self.result.input_version == "No version information found"
        else:
            assert self.result.input_version == true

    def test_extra_kwargs(self, true=None):
        """Test the extra_kwargs property
        """
        if true is None:
            assert self.result.extra_kwargs == {"sampler": {"nsamples": 1000}, "meta_data": {}}
        else:
            assert sorted(self.result.extra_kwargs) == sorted(true)

    def test_injection_parameters(self, true, pesummary=False):
        """Test the injection_parameters property
        """
        if true is None:
            assert self.result.injection_parameters is None
        else:
            import math

            assert all(i in list(true.keys()) for i in self.parameters)
            assert all(i in self.parameters for i in list(true.keys()))

            if not pesummary:
                for i in true.keys():
                    if math.isnan(true[i]):
                        assert math.isnan(self.result.injection_parameters[i])
                    else:
                        assert true[i] == self.result.injection_parameters[i]

    def test_to_dat(self):
        """Test the to_dat method
        """
        self.result.to_dat(outdir=".outdir", label="label")
        assert os.path.isfile(os.path.join(".outdir", "pesummary_label.dat"))
        data = np.genfromtxt(
            os.path.join(".outdir", "pesummary_label.dat"), names=True)
        assert all(i in self.parameters for i in list(data.dtype.names))
        assert all(i in list(data.dtype.names) for i in self.parameters)


class GWBaseRead(BaseRead):
    """Base class to test the GWRead specific functions
    """
    def test_parameters(self, true, pesummary=False):
        """Test the parameter property
        """
        super(GWBaseRead, self).test_parameters(true, pesummary=pesummary)
        full_parameters = [
            'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl',
            'phi_12', 'psi', 'theta_jn', 'ra', 'dec', 'luminosity_distance',
            'geocent_time', 'log_likelihood', 'mass_ratio', 'total_mass',
            'chirp_mass', 'symmetric_mass_ratio', 'iota', 'spin_1x', 'spin_1y',
            'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z', 'chi_p', 'chi_eff',
            'cos_tilt_1', 'cos_tilt_2', 'redshift', 'comoving_distance',
            'mass_1_source', 'mass_2_source', 'total_mass_source',
            'chirp_mass_source', 'phi_1', 'phi_2', 'cos_theta_jn', 'cos_iota']

        self.result.generate_all_posterior_samples()
        assert all(i in self.result.parameters for i in full_parameters)
        assert all(i in full_parameters for i in self.result.parameters)

    def test_injection_parameters(self, true):
        """Test the injection_parameters property
        """
        import math

        super(GWBaseRead, self).test_injection_parameters(true)
        self.result.add_injection_parameters_from_file("./tests/main_injection.xml")
        true = {
            'mass_1': 53.333333, 'mass_2': 26.666667, 'a_1': 0,
            'a_2': 0, 'tilt_1': float('nan'), 'tilt_2': float('nan'),
            'phi_jl': float('nan'), 'phi_12': float('nan'), 'psi': 1.75,
            'theta_jn': float('nan'), 'ra': float('nan'), 'dec': 1.949725,
            'luminosity_distance': 139.76429, 'geocent_time': float('nan'),
            'log_likelihood': float('nan'), 'mass_ratio': 0.5,
            'total_mass': 80., 'chirp_mass': 32.446098,
            'symmetric_mass_ratio': 0.222222,
            'redshift': 0.030857, 'comoving_distance': 135.580633,
            'mass_1_source': 51.736872, 'mass_2_source': 25.868437,
            'total_mass_source': 77.605309, 'chirp_mass_source': 31.474869}
        assert all(i in list(true.keys()) for i in self.parameters)
        for i in true.keys():
            if math.isnan(true[i]):
                assert math.isnan(self.result.injection_parameters[i])
            else:
                assert np.round(true[i], 2) == np.round(self.result.injection_parameters[i], 2)

    def test_calibration_data_in_results_file(self):
        """Test the calibration_data_in_results_file property
        """
        pass

    def test_add_injection_parameters_from_file(self):
        """Test the add_injection_parameters_from_file method
        """
        pass

    def test_add_fixed_parameters_from_config_file(self):
        """Test the add_fixed_parameters_from_config_file method
        """
        pass


class TestCoreJsonFile(BaseRead):
    """Class to test loading in a JSON file with the core Read function
    """
    def setup(self):
        """Setup the TestCoreJsonFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(extension="json", gw=False)
        self.result = Read(os.path.join(".outdir", "test.json"))

    def teardown(self):
        """Remove all files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.core.file.formats.default.Default)

    def test_parameters(self):
        """Test the parameter property of the default class
        """
        super(TestCoreJsonFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the default class
        """
        super(TestCoreJsonFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        true = [self.parameters, self.samples]
        super(TestCoreJsonFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreJsonFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreJsonFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestCoreJsonFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreJsonFile, self).test_to_dat()


class TestCoreHDF5File(BaseRead):
    """Class to test loading in an HDF5 file with the core Read function
    """
    def setup(self):
        """Setup the TestCoreHDF5File class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(extension="hdf5", gw=False)
        self.result = Read(os.path.join(".outdir", "test.h5"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.core.file.formats.default.Default)

    def test_parameters(self):
        """Test the parameter property of the default class
        """
        super(TestCoreHDF5File, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the default class
        """
        super(TestCoreHDF5File, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the default class
        """
        true = [self.parameters, self.samples]
        super(TestCoreHDF5File, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreHDF5File, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreHDF5File, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestCoreHDF5File, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreHDF5File, self).test_to_dat()


class TestCoreDatFile(BaseRead):
    """Class to test loading in an dat file with the core Read function
    """
    def setup(self):
        """Setup the TestCoreDatFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(extension="dat", gw=False)
        self.result = Read(os.path.join(".outdir", "test.dat"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.core.file.formats.default.Default)

    def test_parameters(self):
        """Test the parameter property of the default class
        """
        super(TestCoreDatFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the default class
        """
        super(TestCoreDatFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the default class
        """
        true = [self.parameters, self.samples]
        super(TestCoreDatFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreDatFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreDatFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestCoreDatFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreDatFile, self).test_to_dat()


class BilbyFile(BaseRead):
    """Base class to test loading in a bilby file with the core Read function
    """
    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.core.file.formats.bilby.Bilby)

    def test_parameters(self):
        """Test the parameter property of the bilby class
        """
        super(BilbyFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the bilby class
        """
        super(BilbyFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the bilby class
        """
        true = [self.parameters, self.samples]
        super(BilbyFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the bilby class
        """
        true = "bilby=0.5.3:"
        super(BilbyFile, self).test_version(true)

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        true = {"sampler": {
            "log_bayes_factor": 0.5,
            "log_noise_evidence": 0.1,
            "log_evidence": 0.2,
            "log_evidence_err": 0.1},
            "meta_data": {'time_marginalization': True}}
        super(BilbyFile, self).test_extra_kwargs(true)

    def test_injection_parameters(self, true):
        """Test the injection_parameters property
        """
        super(BilbyFile, self).test_injection_parameters(true)


class TestCoreJsonBilbyFile(BilbyFile):
    """Class to test loading in a bilby json file with the core Read function
    """
    def setup(self):
        """Setup the TestCoreBilbyFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(
            extension="json", gw=False, bilby=True)
        self.result = Read(os.path.join(".outdir", "test.json"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        super(TestCoreJsonBilbyFile, self).test_class_name()

    def test_parameters(self):
        """Test the parameter property of the bilby class
        """
        super(TestCoreJsonBilbyFile, self).test_parameters()

    def test_samples(self):
        """Test the samples property of the bilby class
        """
        super(TestCoreJsonBilbyFile, self).test_samples()

    def test_samples_dict(self):
        """Test the samples_dict property of the bilby class
        """
        super(TestCoreJsonBilbyFile, self).test_samples_dict()

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreJsonBilbyFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreJsonBilbyFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: 1. for par in self.parameters}
        super(TestCoreJsonBilbyFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreJsonBilbyFile, self).test_to_dat()


class TestCoreHDF5BilbyFile(BilbyFile):
    """Class to test loading in a bilby hdf5 file with the core Read function
    """
    def setup(self):
        """Setup the TestCoreBilbyFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(
            extension="hdf5", gw=False, bilby=True)
        self.result = Read(os.path.join(".outdir", "test.h5"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        super(TestCoreHDF5BilbyFile, self).test_class_name()

    def test_parameters(self):
        """Test the parameter property of the bilby class
        """
        super(TestCoreHDF5BilbyFile, self).test_parameters()

    def test_samples(self):
        """Test the samples property of the bilby class
        """
        super(TestCoreHDF5BilbyFile, self).test_samples()

    def test_samples_dict(self):
        """Test the samples_dict property of the bilby class
        """
        super(TestCoreHDF5BilbyFile, self).test_samples_dict()

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreHDF5BilbyFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreHDF5BilbyFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: 1. for par in self.parameters}
        super(TestCoreHDF5BilbyFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreHDF5BilbyFile, self).test_to_dat()


class PESummaryFile(BaseRead):
    """Base class to test loading in a PESummary file with the core Read function
    """

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.core.file.formats.pesummary.PESummary)

    def test_parameters(self):
        """Test the parameter property of the PESummary class
        """
        super(PESummaryFile, self).test_parameters(
            self.parameters, pesummary=True)

    def test_samples(self):
        """Test the samples property of the PESummary class
        """
        super(PESummaryFile, self).test_samples(
            self.samples, pesummary=True)

    def test_version(self):
        """Test the version property of the default class
        """
        true = ["No version information found"]
        super(PESummaryFile, self).test_version(true)

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        true = [{"sampler": {"log_evidence": 0.5}, "meta_data": {}}]
        super(PESummaryFile, self).test_extra_kwargs(true)

    def test_samples_dict(self):
        """Test the samples_dict property
        """
        assert list(self.result.samples_dict.keys()) == ["label"]

        parameters = self.parameters
        samples = self.samples
        for num, param in enumerate(parameters):
            specific_samples = [i[num] for i in samples]
            drawn_samples = self.result.samples_dict["label"][param]
            assert all(i == j for i, j in zip(drawn_samples, specific_samples))

    def test_to_bilby(self):
        """Test the to_bilby method
        """
        from pesummary.core.file.read import is_bilby_json_file

        bilby_object = self.result.to_bilby()["label"]
        bilby_object.save_to_file(
            filename=os.path.join(".outdir", "bilby.json"))
        assert is_bilby_json_file(os.path.join(".outdir", "bilby.json"))

    def test_to_dat(self):
        """Test the to_dat method
        """
        self.result.to_dat(outdir=".outdir")
        assert os.path.isfile(os.path.join(".outdir", "pesummary_label.dat"))
        data = np.genfromtxt(
            os.path.join(".outdir", "pesummary_label.dat"), names=True)
        assert all(i in self.parameters for i in list(data.dtype.names))
        assert all(i in list(data.dtype.names) for i in self.parameters)


class TestCoreJsonPESummaryFile(PESummaryFile):
    """Class to test loading in a PESummary json file with the core Read
    function
    """
    def setup(self):
        """Setup the TestCorePESummaryFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(
            extension="json", gw=False, pesummary=True)
        self.result = Read(os.path.join(".outdir", "test.json"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        super(TestCoreJsonPESummaryFile, self).test_class_name()

    def test_parameters(self):
        """Test the parameter property of the PESummary class
        """
        super(TestCoreJsonPESummaryFile, self).test_parameters()

    def test_samples(self):
        """Test the samples property of the PESummary class
        """
        super(TestCoreJsonPESummaryFile, self).test_samples()

    def test_samples_dict(self):
        """Test the samples_dict property
        """
        super(TestCoreJsonPESummaryFile, self).test_samples_dict()

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreJsonPESummaryFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreJsonPESummaryFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestCoreJsonPESummaryFile, self).test_injection_parameters(
            true, pesummary=True)

    def test_to_bilby(self):
        """Test the to_bilby method
        """
        super(TestCoreJsonPESummaryFile, self).test_to_bilby()

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreJsonPESummaryFile, self).test_to_dat()


class TestCoreHDF5PESummaryFile(PESummaryFile):
    """Class to test loading in a PESummary hdf5 file with the core Read
    function
    """
    def setup(self):
        """Setup the TestCorePESummaryFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(
            extension="hdf5", gw=False, pesummary=True)
        self.result = Read(os.path.join(".outdir", "test.h5"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        super(TestCoreHDF5PESummaryFile, self).test_class_name()

    def test_parameters(self):
        """Test the parameter property of the PESummary class
        """
        super(TestCoreHDF5PESummaryFile, self).test_parameters()

    def test_samples(self):
        """Test the samples property of the PESummary class
        """
        super(TestCoreHDF5PESummaryFile, self).test_samples()

    def test_samples_dict(self):
        """Test the samples_dict property
        """
        super(TestCoreHDF5PESummaryFile, self).test_samples_dict()

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestCoreHDF5PESummaryFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestCoreHDF5PESummaryFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestCoreHDF5PESummaryFile, self).test_injection_parameters(
            true, pesummary=True)

    def test_to_bilby(self):
        """Test the to_bilby method
        """
        super(TestCoreHDF5PESummaryFile, self).test_to_bilby()

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestCoreHDF5PESummaryFile, self).test_to_dat()


class TestGWDatFile(GWBaseRead):
    """Class to test loading in an dat file with the core Read function
    """
    def setup(self):
        """Setup the TestGWDatFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(extension="dat", gw=True)
        self.result = GWRead(os.path.join(".outdir", "test.dat"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.gw.file.formats.default.Default)

    def test_parameters(self):
        """Test the parameter property of the default class
        """
        super(TestGWDatFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the default class
        """
        super(TestGWDatFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the default class
        """
        true = [self.parameters, self.samples]
        super(TestGWDatFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestGWDatFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestGWDatFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestGWDatFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestGWDatFile, self).test_to_dat()


class TestGWHDF5File(GWBaseRead):
    """Class to test loading in an HDF5 file with the gw Read function
    """
    def setup(self):
        """Setup the TestCoreHDF5File class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(extension="hdf5", gw=True)
        self.result = GWRead(os.path.join(".outdir", "test.h5"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.gw.file.formats.default.Default)

    def test_parameters(self):
        """Test the parameter property of the default class
        """
        super(TestGWHDF5File, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the default class
        """
        super(TestGWHDF5File, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the default class
        """
        true = [self.parameters, self.samples]
        super(TestGWHDF5File, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestGWHDF5File, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestGWHDF5File, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestGWHDF5File, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestGWHDF5File, self).test_to_dat()


class TestGWJsonFile(GWBaseRead):
    """Class to test loading in an json file with the gw Read function
    """
    def setup(self):
        """Setup the TestGWDatFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(extension="json", gw=True)
        self.result = GWRead(os.path.join(".outdir", "test.json"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.gw.file.formats.default.Default)

    def test_parameters(self):
        """Test the parameter property of the default class
        """
        super(TestGWJsonFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the default class
        """
        super(TestGWJsonFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the default class
        """
        true = [self.parameters, self.samples]
        super(TestGWJsonFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestGWJsonFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestGWJsonFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: float("nan") for par in self.parameters}
        super(TestGWJsonFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestGWJsonFile, self).test_to_dat()


class TestGWJsonBilbyFile(GWBaseRead):
    """Class to test loading in a bilby json file with the gw Read function
    """
    def setup(self):
        """Setup the TestCoreBilbyFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(
            extension="json", gw=True, bilby=True)
        self.result = GWRead(os.path.join(".outdir", "test.json"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(self.result, pesummary.gw.file.formats.bilby.Bilby)

    def test_parameters(self):
        """Test the parameter property of the bilby class
        """
        super(TestGWJsonBilbyFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the bilby class
        """
        super(TestGWJsonBilbyFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the bilby class
        """
        true = [self.parameters, self.samples]
        super(TestGWJsonBilbyFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        true = "bilby=0.5.3:"
        super(TestGWJsonBilbyFile, self).test_version(true)

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        true = {"sampler": {
            "log_bayes_factor": 0.5,
            "log_noise_evidence": 0.1,
            "log_evidence": 0.2,
            "log_evidence_err": 0.1},
            "meta_data": {"time_marginalization": True}}
        super(TestGWJsonBilbyFile, self).test_extra_kwargs(true)

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        true = {par: 1. for par in self.parameters}
        super(TestGWJsonBilbyFile, self).test_injection_parameters(true)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestGWJsonBilbyFile, self).test_to_dat()


class TestGWLALInferenceFile(GWBaseRead):
    """Class to test loading in a LALInference file with the gw Read function
    """
    def setup(self):
        """Setup the TestCoreBilbyFile class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        self.parameters, self.samples = make_result_file(
            extension="hdf5", gw=True, lalinference=True)
        self.result = GWRead(os.path.join(".outdir", "test.hdf5"))

    def teardown(self):
        """Remove the files and directories created from this class
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_class_name(self):
        """Test the class used to load in this file
        """
        assert isinstance(
            self.result, pesummary.gw.file.formats.lalinference.LALInference)

    def test_parameters(self):
        """Test the parameter property of the bilby class
        """
        super(TestGWLALInferenceFile, self).test_parameters(self.parameters)

    def test_samples(self):
        """Test the samples property of the bilby class
        """
        super(TestGWLALInferenceFile, self).test_samples(self.samples)

    def test_samples_dict(self):
        """Test the samples_dict property of the bilby class
        """
        true = [self.parameters, self.samples]
        super(TestGWLALInferenceFile, self).test_samples_dict(true)

    def test_version(self):
        """Test the version property of the default class
        """
        super(TestGWLALInferenceFile, self).test_version()

    def test_extra_kwargs(self):
        """Test the extra_kwargs property of the default class
        """
        super(TestGWLALInferenceFile, self).test_extra_kwargs()

    def test_injection_parameters(self):
        """Test the injection_parameters property
        """
        super(TestGWLALInferenceFile, self).test_injection_parameters(None)

    def test_to_dat(self):
        """Test the to_dat method
        """
        super(TestGWLALInferenceFile, self).test_to_dat()
