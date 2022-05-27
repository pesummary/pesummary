## 0.13.5 [27/05/2021]

Re-upload of v0.13.4 due to faulty wheel

## 0.13.4 [27/05/2021] - Yanked

### Changed

- astropy==5.1 - Update the pesummary.gw.cosmology module to be compatible
  with astropy==5.1

### Added

## 0.13.3 [16/12/2021]

### Changed

- Testing runtime - Reduce the time taken to run the testing suite
- Python3.6 - Drop support for `python==3.6`
- Bug fix - Prevent foward spin evolution from being applied to non-precessing
  NSBH systems
- Bug fix - Allow `pesummary.core.file.mcmc.burnin_by_step_number` to remove
  N samples as burnin
- astropy==5.0 - Update code to be compatible with `astropy==5.0`
- ligo.em-bright - Migrate from the `p_astro` package to the `ligo.em_bright`
  package
- `summarycombine_posteriors` - Only find common parameters among mixed labels
  when using `summarycombine_posteriors`

### Added

- Added conversion to calculate the SNR in subdominant multipole moments
  following Mills et al. https://arxiv.org/abs/2007.04313.
- Allow the user to specify a catalog when fetching
  public data with the `pesummary.gw.fetch.fetch_open_samples` function
- Allow the user to modify the config data with `summarymodify`
- Allow the user to remove an entire analysis with `summarymodify`


## 0.13.2 [10/09/2021]

### Changed

- pycbc dependent tests - perform `pycbc` dependent tests in all python
  environments
- logger.warning - replace `logger.warn` with `logger.warning`
- numpy aliases - replace deprecated numpy aliases of builtins with the builtins
  they aliased 
- Adding waveform to strain plot - Prevent ValueError from being raised when
  adding a waveform to a strain plot

### Added

- Added a new command line argument `--preferred` to allow the user to set a
  preferred analysis within a pesummary metafile. The preferred analysis can be
  printed with the `.preferred` property


## 0.13.1 [31/08/2021]

### Changed

- `astropy pin` - removed the temporary `astropy<=4.2.1` pin
- `SpinTaylorT4` - use the `SpinTaylorT4` approximant for the backwards spin
  evolution by default after a bug was found in the `SpinTaylorT5` approximant
  which is the default approximant used in the LALSimulation function

### Added

## 0.13.0 [23/08/2021]

### Changed

- `pycbc_inference` - update pesummary.core.file.formats.hdf5 and testing suite
  to ensure compatibility with `pycbc_inference`.
- `f_low` - try and extract `f_low` from the config file stored within a `bilby`
  result file (if present) if it cannot be extracted from the `bilby` metadata
- unpin `tqdm` - unpin the `tqdm` requirement
- `lalsuite >= 7.0.0` - update lalsuite requirement

### Added

- Update testing suite to test PESummary with python3.9
- add the `pesummary.gw.file.formats.princeton` module to read in samples
  released by Venumadhav et al.
- Update docs to explain how to compare samples from non-LIGO GW groups
- logger message for redshift computation - Display logger message when running
  the `pesummary.gw.conversions.cosmology.z_from_dL_exact` method notifying the
  user of the progress
- Add a new `summarysplit` executable which splits a set of posterior samples
  stored in a given file into N separate files
- Add a conversion to calculate the spins at infinite separation. See
  Johnson-McDaniel et al. for details. This can be performed with `summarypages`
  by passing the `--evolve_spins_backwards` flag
- Allow the user to pass e.g. `--samples *.dat` to pesummay executables
- Allow the user to calculate and plot the time domain waveform uncertainty
  associated with a table of posterior samples.
- Allow the user to reweight a set of posterior samples a different prior.
  Currently the user can only reweigh to a uniform_in_comoving_volume prior.
  This can be performed with `summarypages` with the
  `--reweight uniform_in_comoving_volume` flag
- Add a new field to the pesummary metafile which allows the user to store a
  short string which describes a given analysis. This description can be
  added via `summarypages` with the `--descriptions` flag.


## 0.12.1 [18/05/2021]

### Changed

- `chi_p_2spin` conversion - bug fix in the
  `pesummary.gw.conversions.spins.chi_p_2spin` function in order to ensure that
  the conversion matches Eqs. 8,9 in arXiv:2012.02209
- AttributeError when loading Deprecated metafile - fix the
  `pesummary.core.file.formats.pesummary.PESummaryDeprecated` class to prevent
  an AttributeError from being raised when loading a Deprecated JSON file
- `numpy` requirement - unpin the `numpy` requirement
- Calibration posterior - tidy up how pesummary interacts with the calibration
  posterior samples

### Added

- Added a test in the CI to ensure that all builds can read in a random
  selection of result files released as part of GWTC-2
- Add the functionality to store gravitational wave strain data in the
  metafile if requested

## 0.12.0 [27/04/2021]

### Changed

- NSBH conversions - bug fix in the NSBH conversions where the primary
  spin magnitude was incorrectly used. The conversions now correctly use
  spin_1z
- matplotlib rcParams - prevent seaborn/gwpy/pesummary affecting the matplotlib
  rcParams when pesummary is imported within a python console
- pandas - unpin the pandas requirement
- IndexError - fix a bug where an IndexError was raised when adding to an
  existing webpage
- m1--m2 triangle plot - use the Bounded_2d_kde class when generating an
  m1--m2 triangle plot to handle the q=1 boundary
- NSBH conversions 2 - use the NSBH fits when lambda_1 samples exist in the
  posterior table but are all 0
- License - change license at the top of each file
- Conversions - move all conversions from pesummary.gw.file.conversions to
  pesummary.gw.conversions
- Key data - display key data for a given parameter on the 1d_histogram pages

### Added

- Added the conversion to calculate the precession SNR from Fairhurst et al.
  see https://arxiv.org/abs/1908.05707
- Added the conversion to calculate the modified chi_p from L. M. Thomas et al.
  see https://arxiv.org/abs/2012.02209
- Added the conversion to calculate the viewing_angle
- Added a new `summaryextract` executable which extracts the posterior samples
  for a given analysis stored within a pesummary metafile and writes it to a
  separate file
- Added a new `summarycombine_posteriors` executable which performs the same
  task as `cbcBayesCombinePosteriors`
- Added a new `summarytgr` executable which performs the post-processing for
  the IMRCT of General Relativity
- Added checkpointing to pesummary. You may now restart from checkpoint by
  passing the `--restart_from_checkpoint` flag
- Add an all tab for each 1d_histogram category when running with the gw
  module. This means that all mass 1d posteriors can easily be compared
- Allow the user to read a file from a remote server
- Display the bilby prior file on the config tab when running with the gw module
- Add new `pe_algorithm` property of the read object which aims to identify
  which algorithm/software was used to generate the posterior samples
- Generate a set of 'expert plots' when running with the gw module. These are
  displayed on the webpages when the toggle is activated. These may be
  disabled with the new `--disable_expert` flag
- Added a new argparse action to check that the files exist before continuing
  with the rest of the workflow
- Add a property to the `Parameter` class which stores the parameter description
- Extract the config file stored within a bilby result file

## 0.11.0 [15/12/2020]

### Changed

- FutureWarning - fix FutureWarning: elementwise comparison failed

### Added

- Allow the user to add previously generated plots to the webpages through the
  `--add_existing_plot` command line argument.
- Add a ProbabilityDict class to handle discrete PDFs
- Add NSBH specific conversions to the `pesummary.gw.file.conversion` module

## 0.10.0 [27/11/2020]

### Changed

- `summaryreview` - upgrade the `summaryreview` script to include additional
  tests and display them on a single html page
- `summarypipe` - upgrade `summarypipe` to allow for `pbilby` rundirs
- `seaborn` - update the code to allow for `seaborn==0.11.0`
- `Array` class - separate `Array` class into its own `pesummary.utils.array`
  module

### Added

- Add a section to the Documentation indicating the file formats that
  PESummary is able to read it through its `io.read` function
- Add a 'Preliminary' watermark to GW html pages and plots if they are not
  reproducible (i.e. if the pages do not include PSD and configuration
  settings)
- Allow for `.npy` files to be read in with the `io.read` function
- Allow for `.csv` files to be read in with the `io.read` function
- Allow for `sql` databases to be read in with the `io.read` function
- Allow the user to generate `triangle`, `reverse_triangle` and `2d_kde` plots
  from the SamplesDict class
- Add the injected line(s) to 1d comparison histograms if an injection file
  is provided
- Add a `--disable_remnant` flag to prevent remnant conversions from being
  calculated
- Add a module to fetch public data
- Be able to fetch public samples released as part of the `GWTC-2` publication
- Print a summary of the opened file giving the number of samples, list of
  parameters etc
- Add a summary table to the comparison pages giving 90% credible intervals,
  injected values etc for each analysis
- Test the pesummary docker image as part of the CI
- Add a `converted_parameters` attribute to the opened result file so you can
  see which parameters have been added to the posterior table through the
  conversion module

## 0.9.1 [04/09/2020]

### Changed

- list index - correct typo in list index

### Added

## 0.9.0 [04/09/2020]

### Changed

- `network_matched_filter_snr` - correct the `network_matched_filter_snr`
  conversion
- Documentation - reorganize the documentation to make it easier to navigate
- Violin plots - allow for the user to pass weights to the violin plot function

### Added

- Test that all examples as part of the testing-suite
- Export the environment to a conda yml file
- Add a `.AUTHORS` file which documents all those that have contributed to the
  development of PESummary
- Add a notebook module which allows for the generation of jupyter notebooks


## 0.8.0 [13/08/2020]

### Changed

- Install path - write install path at runtime
- Injection data - store injection data as numpy structured array in the
  pesummary metafile
- `summarytest` - make `summarytest` an entry point

### Added

- Allow for the user to convert the 4PolyEOS and 4SpectralDecomp parameters
  to `lambda_1` and `lambda_2`
- Adding functionality to generate a split violin plot
- Add multiple methods for how the `Bounded_1d_kde` treats the domain walls
- Allow for the user to specify the bilby `PriorDict` class when loading a bilby
  prior file
- Add class methods to make it easy to initialise SamplesDict classes from a
  file
- Add further documentation to explain how to generate plots from the loaded
  PESummary result file
- Add injected value to default skymap if an injection has been performed
- Add injection quantities to summary table if they are present
- Allow the user to specify the corner parameters they wish to see. This is done
  with the `--add_to_corner` command line argument
- Allow the user to pass a `psd.xml.gz` file to the `summarypages` executable.
  This can be done with `--psd H1:psd.xml.gz L1:psd.xml.gz` for example.
- Add maxP quantity to the summary table

## 0.7.0 [09/07/2020]

### Changed

- `matplotlib.pyplot` calls - remove all `matplotlib.pyplot` calls from the
  `pesummary.core.plots.plot` and `pesummary.gw.plots.plot` modules. All
  `matplotlib.pyplot` calls are now in `pesummary.core.plots.figure`.
- Modify `ligo.skymap` fits header - add `DATE_OBS` to the ligo.skymap fits file

### Added

- Allow the user to easy generate comparison histogram plots, corner plots
  and skymap comparison plots from the 
  `pesummary.utils.samples_dict.MultiAnalysisSamplesDict`,
  `pesummary.utils.samples_dict.SamplesDict` and
  `pesummary.gw.file.skymap.SkyMapDict` objects
- Add tests for python3.8
- Use colored logs for console logging if `coloredlogs` is installed
- Move all tests inside the `pesummary` package so they are included in the
  tarball
- Allow the user to specify which parameters they wish to include in the
  corner plot via the `--add_to_corner` command line argument. If the `core`
  package is used, only these parameters are used, defaulting to all parameters
  if None provided. If the `gw` package is used, the parameters are added to the
  default `gw` corner parameters
- Allow for violin plots to use `Bounded_1d_kde`s rather than the default
  `gaussian_kde`.
- Allow the user to specify a random seed with the `--seed` command line
  argument. This ensures reproducability.
- Allow for a `bilby` prior file to be read in and stored in the pesummary
  metafile.
- Allow the user to specify the `gracedb` service url they wish to use. This is
  via the `--gracedb_server` command line argument
- Add the Reiss H0 cosmology to the `pesummary.gw.cosmology` module. All
  available cosmologies can use the Reiss H0 cosmology by adding the
  `_with_Riess2019_H0`
- Allow the user to specify the path to samples` when more than one posterior
  table is located in the file. This can be done with the `path_to_samples`
  kwarg in the `pesummary.io.read` function or the `--path_to_samples` command
  line argument


## 0.6.0 [11/06/2020]

### Changed

- HTML tables - tidy up how the tables are displayed in the html pages
- store_skymap delimiter - change the delimiter for the store_skymap job in
  `pesummary.gw.finish` from `:` to `|`
- Mock GraceDB events - prevent `InputError` from being raised when a mock
  gracedb ID is provided
- Requirements - Update `tqdm` requirements

### Added

- Add a history field to the metafile which contains information about
  when the file was created, who created it, and the command line used to
  create the file
- Add a new `summarycompare` executable which compares the contents of two
  result files
- Allow for hdf5 datasets to be compressed with the `--hdf5_compression`
  command line argument
- Allow for each analysis to be saved as seperate PESummary metafiles but
  connected to each other through a single PESummary metafile with external hdf5
  links when the `--external_hdf5_links` command line argument is provided
- Add 5th and 95th percentiles to the summarytables shown on the result files
  homepage
- Add gracedb module which allows for information to be downloaded from gracedb.
- Add the `--gracedb_data` command line argument which allows the user to
  specify what information is downloaded from gracedb and stored in the metafile
- Allow for extra kwargs to be passed to the `pesummary.io.read.read` function
  to prevent prior samples from being collected (applicable to `bilby` result
  files) for example.


## 0.5.6 [27/05/2020]

### Changed

- Parallelize spin evolution - allow the spin evolution calculation to run on
  multiple cpu's to speed up the evaluation
- Versioneer - utilize the `versioneer` package to keep track of versions
- Downsample - downsample the result file prior to passing to the conversion
  suite
- rename non-precessing remnant fits - rename the `{}_non_evolved` quantities to
  `{}` for non-precessing remnant properties.
- Bounded 1d KDEs for JS divergence - use the Bounded_1d_kde function for
  JS divergences calculated for the `gw` package
- BBH remnant fits - only apply BBH remnant fits to BBH systems. This can be
  forced with the `force_remnant_computation` kwarg to the `Conversion` class
- Parallelize redshift computation - Allow the exact redshift method to run on
  multiple cpus to speed up the computation

### Added

- Add more options to the PSD and Calibration `save_to_file` method
- Add new parser function to handle unknown command line arguments provided to
  `summarypages`
- Add a `write` method to easily transform posterior samples between multiple
  file formats
- Store `ligo.skymap` data in the metafile and add a new class to handle this
  data
- Allow the user to specify the name of the metafile produced with the
  `--posterior_samples_filename` command line argument

## 0.5.5 [11/05/2020]

### Changed

- KeyError bug - Fix bug where a KeyError was being raised when initializing
  the prior dict

### Added


## 0.5.4 [08/05/2020]

### Changed

- Modify existing kwargs - Allow the user to modify existing kwargs with
  the `summarymodify` executable
- Remnant conversions - Only calculate `final_mass_non_evolved` if `final_mass`
  is not already in the result file
- Change `log_evidence` to be `ln_evidence` so the base is clear
- Comparison statistics - Allow for the comparison statistics to always be
  generated even when a single matrix exception error is raised from the
  gaussian_kde


### Added

- Added a smart_round function which rounds quantities according to the
  significant digit of the lowest uncertainty
- Allow for entire posterior distributions to be replaced or removed via the
  `summarymodify` executable
- Allow for the conversion module to not be executed. This is done by passing
  the `--no_conversion` command line is passed
- Disable the corner plot from being generated by adding the `--disable_corner`
  command line argument
- Allow for the user to generate plots directly from the SamplesDict object
- Allow for the user to easily generate comparison statistics from the
  pesummary result file
- Add a `combine` property to the `MCMCSamplesDict` object  which concatenates
  the samples from multiple chains
- Added a new `--file_format` command line argument which allows the user to
  specify the file format of the input file.


## 0.5.3 [23/04/2020]

### Changed

- bilby posterior load - small change in how the pandas DataFrame
  is converted to a numpy array

### Added

- Allowing the user to specify a chosen cosmology for redshift calculations.
  This is done via the `--cosmology` command line argument
- Allow configuration files that do not include section headers to be read
  in with PESummary.

## 0.5.2 [21/04/2020]

### Changed

- Storing priors - Fix a bug where the priors were not being stored correctly
  in the PESummary meta file
- `--compare results` - Fix a bug where the injection dataset was not being
  calculated properly when the `--compare_results` option is passed.


### Added

- Allow the user to pass MCMC chains to the `summarypages` executable. This is
  activated with the `--mcmc_samples` command line argument
- Add extra information to the `summaryreview` script. This includes computing
  the maximum difference between the samples
- Store the `ligo.skymap` statistics in the PESummary meta data
- Store the gracedb ID in the PESummary meta data
- Allow the user to choose which method they wish to adopt when computing the
  redshift. This is done with the `--redshift_method` argument
- Allow the user to pass a custom matplotlib style file for custom plotting.
  This is done with the `--style_file` argument


## 0.5.1 [03/04/2020]

### Changed

- Building PESummary - build PESummary from tarball in CI to prevent failed
  tarballs from being released

### Added

- Added the matplotlib style file MANIFEST.in to fix v0.5.0 tarball

## 0.5.0 [03/04/2020]

### Changed

- lalsuite requirement - increase the lalsuite requirement to be `>=6.70.0`
- `chi_p` conversion - correct the definition of `chi_p` in the conversion
  module
- conversions tests - improve the conversion testing suite to prevent
  conversion bugs from being introduced

### Added

- Allow a `SamplesDict` object to be converted to pandas dataframe with the
  `to_pandas` function
- Add a new `summaryrecreate` executable which allows the user to recreate the
  analysis that is stored in a PESummary metafile
- Allow the user to regenerate derived posterior distributions with the
  `--regenerate` command line argument
- Ability to extract meta data (`f_low`, `f_ref` etc.) from a GW `.dat` result
  file
- Allow for the user to calculate the remnant properties using the
  `IMRPhenomPv3HM` and `SEOBNRv4PHM` waveform models. This can be done by adding
   the `--waveform_fits` command line argument.

## 0.4.0 [25/03/2019]

### Changed

- `summaryconvert` removal - removed the `summaryconvert` executable as this
  is no longer supported
- `pesummary.core.file.formats.default.Default` - improve the Default reading
  class such that it can read more file formats
- `pesummary.utils.utils.trange` - remove hidden function import
- `pandas < 1.0.0` - remove the `pandas < 1.0.0` requirement.
- Redirect old doc links - All old documentation links are redirected to the
  stable_docs.

### Added

- Add new `summarymodify` executable which allows for result file modification
  from the command line
- Allow the for the user to change file format to a `lalinference` dat file
  with the `summaryclean` executable
- Store the package information in the PESummary metafile
- Improve the documentation to include a new landing page for stable vs unstable
  file versions
- Allow for the user to calculate the remnant properties using the `NRSurrogate`
  models. This can be done by adding the `--NRSur_fits` command line argument.

## 0.3.4 [12/03/2019]

### Changed
- `_Input.is_pesummary_metafile` function - check for both standard and deprecated PESummary metafiles
- `_make_directories` method - check to see if the directory webdir/dir exists not dir.

### Added

## 0.3.3 [11/03/2019]

### Changed

- Command line buttons - All command lines are contained within the Bootstrap
  `popover` object.
- IFO colors - Use the gwpy IFO colors
- Multiple PESummary metafiles - Allow the user to pass multiple PESummary
  metafiles from the command line with the `--samples` command line argument
- Row major - Store the PESummary metafile in row major format rather than
  column major for hdf5 file formats
- `.to_lalinference` method - Tidy up the `.to_lalinference` method to allow
  for more versatility
- Remove the `summarycombine_metafiles` executable - The `summarycombine` and
  `summarycombine_metafiles` are now combined into a single `summarycombine`
  executable.
- Permute result file - Permute the file format from key -> label to
  label -> key
- Store all attributes - Store all available attributes stored in the
  lalinference result file in the PESummary metafile
- ligo.skymap fits file - Store all required meta data in the ligo.skymap
  fits file
- Duplicated code - Started to remove duplicated code between the `core` and
  `gw` packages.

### Added

- Add captions below some of the images to explain the key features
- Allow the user to download psds and calibration envelopes from the
  `Downloads` page
- Add a `pesummary.gw.file.psd.PSD` and a
  `pesummary.gw.file.calibration.Calibration` class to handle the psds and
  calibration envelopes.
- Add an extra test to check the `summarycombine_metafiles`,
  `summaryclassification` and `summaryclean` executables are working as expected
- Allow html tables to be exported to CSV files from the webpages
- Allow html tables to be exported to JSON files from the webpages
- Allow the bash command that was used to generate the webpages to be downloaded
  from the webpages
- Allow the user to pass custom PSD command line arguments for a specific
  label `--{label}_psd`.
- Allow the user to pass custom calibration command line arguments for a
  specific label `--{label}_calibration`.
- Add a new `summarypipe` executable which will generate a `summarypages`
  command line for you provided that you have passed a valid run directory.
- Added a function to allow the user to convert the data stored in a result
  file to a latex table via the `to_latex_table` function
- Added a function to allow the user to convert the data stored in a result
  file to a list of latex macros via the `generate_latex_macros` function
- Add a function to calculated a prior conditioned on a given set of posterior
  samples
- Add a new `pesummary.core.plots.population` module to create scatter plots
  for a given 'population' of result files
- Add softlinks in the hdf5 PESummary metafile to help reduce the size
- Add an `inverted_mass_ratio` conversion
- Add function to unzip files that are passed from the command line
- Allow the user to easily convert the any result file to a `lalinference`
  posterior_samples.dat by using the `.to_lalinference` method.
- Add conversions to calculate the remnant properties
- Add conversions to calculate the spin-evolved remnant properties

## 0.3.2 [17/01/2020]

### Changed

- Save format - Change the default to always save the metafile to HDF5 format
  unless the `--save_to_json` flag is passed
- ligo.skymap - Update the code to work with ligo.skymap >= 0.1.12

### Added

-

## 0.3.1 [14/01/2020]

### Changed

- Thumbnails on homepage - Fix the javascript so now when a thumbnail is
  clicked you are presented with that image and not always the very first
  image
- Package information - Use `conda list` to get a complete list of packages
  falling back to pypi if the current environment is not a conda environment

### Added

- Allow the user to generate a latex table of the results file when read in
  with PESummary. Both for a single result file or a PESummary metafile
- Added an extra command line argument `--multi_process` to the `summarypages`
  executable which allows the user to specify the number of cores they wish to
  run on when generating plots.
- 

## 0.3.0 [05/01/2020]

### Changed

- Documentation: improve the documentation to explain how to extract information
  from the PESummary metafile
- `bilby` matched filter SNR: fix a bug where the `bilby` matched filter
  SNRs were not being extracted correctly
- Pass all parameters: allow all parameters stored in the result file to be
  plotted when the `gw` module is used (not just those with a 'standard name').
- Corner plot: Fixed a bug where there were multiple `all` buttons when more
  than one result file was passed.
- Webpage footer: Changed the layout of the webpage footer to make it look more
  professional
- Width of container and images: Changed the width of the container on the
  html pages to make it easier to inspect the plots
- `summaryclassification` command line arguments: Changed the command line
  arguments of the `summaryclassification` executable to try and make it more
  intuitive.
- Bug from Will Farr: Fix a bug found by Will Farr which prevented certain
  PESummary metafile from being able to be read in with PESummary
- `--disable_interactive` bug: Fix a bug where the `--disable_interactive`
  command line argument did not actually disable the interactive plots.
- `_Conversion` class: Change how the user interacts with the
  `pesummary.gw.file.conversions._Conversion` class to make it more intuitive.


### Added

- Added a new `summarypageslw` executable to produce what the user asked for
  and no more. It is designed to be a lightweight version of the
  `summarypages` executable
- Added a new `ignore_parameters` command line argument to the `summarypages`
  executable to allow for user to ignore certain parameters in their result
  file if they wish to.
- Added a new `downloads` and `about` page to contain links to all the
  documents that you can download and information about how to pages were
  produced respectively
- Added a new `--public` command line argument for the `summarypages` executable
  which produces the public facing summary pages
- Added a new `--publication_kwargs` command line argument which allows the user
  to modify the publication quality plots.
- Added a new function to control how many columns to use when generating
  comparison plots. This will prevent the labels from overlapping.
- Added tidal parameters to the `gw` corner plots if they are in the result
  file.

## 0.2.4 [05/12/2019]

### Changed

- Input classes: reduce duplicate code by making pesummary.gw.inputs.GWInput
  inherit from pesummary.core.inputs.Input
- JSON encoder: modify the pesummary.core.file.meta_file.PESummaryJsonEncoder
  to work with `bilby==0.6.1`

### Added

- Modified CI to add a `consequences` job to run a `bilby` and a `lalinference`
  job and ensure that the output files are compatible with PESummary
- Added a `--nsamples` arguments to `summarypages` which allows the user
  to downsample their result file
- Add a scheduled CI job to run the `bilby` and `lalinference` jobs every night

## 0.2.3 [27/11/2019]

### Changed
- Colors and linestyles: currently, a seaborn palette will duplicate colors if
  n > 10. Therefore if a user passed more than 10 result files, you would not
  be able to distinguish the posteriors. Now, if more than 10 result files are
  passed, the linestyle is changed to allow for distinguishability
- GW Conversion: remove duplicate code to handle the conversion of parameters.
  Now only the pesummary.gw.file.conversions._Conversion class is used.
- GWTC1: Added a pesummary.gw.file.formats.GWTC1 module which acts functionality
  to read in the GWTC1 posterior samples
- Bounded 1d kdes: If `--kde_plot` is passed and the `--gw` module used,
  bounded 1d kdes will be made by default
- Docs: Updated the docs to the current version of the code

### Added
- Added an 'interactive' module which allows for an interactive corner plot to
  be produced with `plotly`.
- Added an option to not produce interactive plots `--disable_interactive`
- Added an option to not produce comparison plots `--disable_comparison`
- Added a `summaryversion` executable which will display the version of
  PESummary from the command line
- Added an "all" page to display all posteriors for a given result file on
  a single page
- Allow the user to pass custom colors and linestyles with the `--colors` and
  `--linestyles` command line arguments
- Added a `pesummary.core.plots.bounded_1d_kde` module which creates
  bounded 1d KDEs
- Added interactive comparison plots which are made by default
- Added a new `summaryversion` executable

## 0.2.2 [31/10/2019]

### Changed

### Added
- Added plots which show the state of the detector at the time of the GW event if
  the GW data is passed via the `--gwdata` command line option
- Added a `summarydetchar` executable which allows for generation of detector
  related plots easily
- Allow for a `bilby` and `bilby_pipe` datadump pickle file to be passed to
  PESummary via the `--gwdata` command line option

## 0.2.1 [27/10/2019]

### Changed

### Added
- Bar chart to show the classification probabilities and added them to the
  webpages
- Generate both source and extrinsic corner plots for gw result files
- Compute `HasNS` and `HasRemnant` probabilities with the `p_astro` package

## 0.2.0 [24/10/2019]

### Changed
- Calibration Plot: Include both the prior and posterior
- `summaryplots` executable: Made the `summaryplots` executable more versitle
  such that individual 1d histograms/skymaps can be generated from command_line
- Moved `cli` into `pesummary` package

### Added
- Priors: Priors are extracted from bilby result files and stored in the
  metafile
- Argument `--include_prior` which if passed, add priors to the 1d_histogram
  plots
- Conversion is now done via the `pesummary.gw.file.conversion._Conversion`
  class
- Argument `--palette` flag allows the user to choose a seaborn color palette to
  distinguish result files
- Samples are now internally handled via the `pesummary.utils.utils.SamplesDict`
  class
- A prior file can now be passed with the `--prior_file` command line argument

## 0.1.8 [16/09/2019]

### Changed
- Comparison pages: Display both the CDF and box plots on the comparison pages
- Reading in files: Changed how files are read in with PESummary. We now have
  seperate classes for `bilby` and `lalinference` as well as a default class.

### Added
- Generate box plots to compare the different result files
- Conversion to calculate network SNR
- Functions to convert the PESummary metafile to either a bilby results object
  or save as a lalinference results file
- Arguments `--nsamples_for_skymap` and `--multi_threading_for_skymap` to
  reduce the runtime of ligo.skymap
- Extra information to the .version file (builder, last committer, date)
- Pre-commit hook which will run `black` in order to autoformat commited files
  to PEP8
- Argument `--custom_plotting` which allows the user to pass a python file
  containing custom plots
- Version tab on homepage to display PESummary version used to generate the
  pages
- Logging tab on homepage to display the output from the PESummary code
- `summarypublication` executable which produces publication quality plots
  based on the passed result files.
- Argument `--publication` calls the `summarypublication` executable and adds
  a new Publication tab to the homepage to show publication plots
- Added a `command_line` button which is displayed under the plot showing the
  command line used to generate the plot.
- Added a `summaryclassification` executable which produces the source
  classification probabilities
- Added a `summaryclean` executable which cleans the input data file
- Added a `--kde_plot` option which plots kdes of the 1d histograms rather than
  a conventional histogram
- Added the `extra_kwargs` property to the read function. This will try and
  scrap all extra information from the result file. This information is also
  printed on the homepage.
- Allow the CI to release new versions of the code
- Added a `summarycombine_metafile` executable which will combine multiple
  PESummary metafiles into a single file
- Added a `summarycombine` executable which will combine multiple result
  files into a single PESummary metafile

## 0.1.7 [15/06/2019]

### Changed
- Latex_labels: Moved the latex labels into pesummary.core.plots.latex_labels
  and pesummary.gw.plots.latex_labels
- z_from_dL: The redshift is no longer calculated exactly from the luminosity
  distance because it was very slow. Now we compute the redshift for 100
  distances and interpolate to find the redshift for all distances. This
  sped of the function `pesummary.gw.file.conversions.z_from_dL` by 4000x.
- Aspect ratio: Fix the aspect ratio for the plots showed on the multiple tab
- LALInference.fits - LALInference.fits file produced from the
  `pesummary.gw.plots.plot._ligo_skymap_plot` now saved in the web directory

### Added
- Added a conversion to calculate the posterior for the time in each detector
- Add a conversion to work out the time in each detector from samples for
  `geocent_time`
- Allow the user to pass a trigger file (coinc.xml) to PESummary using the
  `--trigfile` flag
- Allow the user to pass the command line arguments in a configuration file
- Plot the skymap using the ligo.skymap module if it is installed
- Added an `existing_samples_dict` property to the ExistingFile class which
  puts the existing_labels, existing_parameters and exising_samples in a
  dictionary
- Added a `compare_results` flag which allows the user to specify which results
  files they wish to compare. In order to use this flag you must pass a
  PESummary meta file.
- Added a `--no_ligo_skymap` option which will prevent ligo.skymap from
  generating a skymap
- Added a `--gwdata` flag which allows the user to pass a gw strain cache file.
  If a valid cache file is passed, the timeseries is plotted and the maxL
  waveform superimposed.

## 0.1.6 [13/05/2019]

### Changed
- Python2.7: Change the yield statements in pesummary.core.file.one_format to
  make the code python2.7 compatible.
- setuptools: Change the setup.py file to import setup from `setuptools` rather
  than using `distutils.core`
- Labels: Labels are now used under the approximant tab to distinguish
  different runs.
- Multiple psds/calibration: The user can now pass different psds and different
  calibration envelopes for different results files. Before if you passed a PSD
  or calibration envelope it was assigned to all results files.
- guess_url: Change the guess_url function to include LLO
- Skymap: Flipped the x axis on the skymap plot to run from 24 -> 0 instead of
  0 -> 24.
- BUG: Fixed a bug in the pesummary.gw.file.conversions.spin_angles function 

### Added
- PESummary now supports non GW specific results files
- Docker image for PESummary. Simply run `docker pull 08hoyc/pesummary:v0.1.6
- Example python script for running PESummary from within python shell
- Extract all information including psd and calibration information from the
  gw specific pesummary metafile
- Add class to handle dictionary command line inputs

## 0.1.5 [13/04/2019]

### Changed
- Executables: excutables now no longer have file extensions
- Entry points: all excutables are now entry points
- Version file: The code version is now written to a .version file and opened
  when you type `pesummary.__version__`.

## 0.1.4 [05/04/2019]

### Added
- Marginalized parameters: handle marginalized parameters for `LALInference`
  results files. This is not necessary for `bilby`
  [#64](https://git.ligo.org/lscsoft/pesummary/issues/64)
- PSD plot: plot the PSD when passed with the `--psds` flag
  [#39](https://git.ligo.org/lscsoft/pesummary/issues/39)
- Calibration plot: plot the calibration uncertainties when passed either
  the calibration envelope via the `--calibration` flag or constant values
  via the configuration file
- Error page: adding an error page which you are redirected too if there is a
  problem [#90](https://git.ligo.org/lscsoft/pesummary/issues/90)
- LALInference fixed parameters: load the configuration file and look to see
  if there are any fixed parameters. If there are, add them to the list of
  parameters and samples [#62](https://git.ligo.org/lscsoft/pesummary/issues/62)
- JSON vs HDF5: allow the user to choose whether they would like to save the
  meta file as a JSON or HDF5 format. By default, the data will be saved as
  JSON. To save as HDF5, pass the `--save_to_hdf5` flag
  [#91](https://git.ligo.org/lscsoft/pesummary/issues/91)
- rapidPE and GWModel: ability to run on rapidPE and GWModel results files
- Custom labels: pass custom labels with the `--labels` flag
- Additional data in meta file: store the config file, psd data and
  calibration data in the meta file [#57](https://git.ligo.org/lscsoft/pesummary/issues/57)
- Download links: links on the navigation bar to download either the meta file
  or the individual histogram dat files [#72](https://git.ligo.org/lscsoft/pesummary/issues/72)
- Bilby JSON files: handle the input of bilby JSON files
  [#91](https://git.ligo.org/lscsoft/pesummary/issues/91)
- Distance and Redshift pages: add distance and redshift pages to the
  navigation bar [#87](https://git.ligo.org/lscsoft/pesummary/issues/87)
- Time domain plots: add plots showing the time domain waveform on the
  homepage. If more than one results file is given, show the time domain
  comparison plot [#74](https://git.ligo.org/lscsoft/pesummary/issues/74)
- CDF plots: Add CDF plots to the parameter pages. These are shown when you
  click on the 1d histogram plot [#73](https://git.ligo.org/lscsoft/pesummary/issues/73)
- Generate examples as part of CI: trigger the CI of
  [pesummary_examples](https://git.ligo.org/charlie.hoy/pesummary_examples?nav_source=navbar)
  such that example pages are always up to dat
- LALInferenceResultsFile class: class to handle the LALInference results file


### Changed
- Removed the banner on the webpages and moved the navigation bar to the top of
  the page [#94](https://git.ligo.org/lscsoft/pesummary/issues/94)
- Changed the logic of the `OneFormat` class to handle the different data
  types [#80](https://git.ligo.org/lscsoft/pesummary/issues/80)
  [#91](https://git.ligo.org/lscsoft/pesummary/issues/91)
  [#79](https://git.ligo.org/lscsoft/pesummary/issues/79)
- Use `theta_jn` instead of `iota` [#82](https://git.ligo.org/lscsoft/pesummary/issues/82)
- Tests to enhance coverage
- All paths are now relative meaning that you can move the html pages to
  another location and everything will still work. This also means that it will
  also work on your laptop and not just on clusters
  [#81](https://git.ligo.org/lscsoft/pesummary/issues/81)
- `multiple_posterior.js` to fix bug where multiple posteriors were not being
  displayed for the comparison pages [#90](https://git.ligo.org/lscsoft/pesummary/issues/90)
- Normalised posterior distributions for the comparison plots allowing for easy
  comparisons [#86](https://git.ligo.org/lscsoft/pesummary/issues/86)
- Move to LSCSoft namespace [#63](https://git.ligo.org/lscsoft/pesummary/issues/63)
- Colours for the waveform plots are now standard colours for each detector
  [#84](https://git.ligo.org/lscsoft/pesummary/issues/84)
- Default colors are the same as the seaborn colorblind palette
- Moved the standard_names to a seperate file
