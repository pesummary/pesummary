## 0.3.3 [unreleased]

### Changed

- Command line buttons - All command lines are contained within the Bootstrap
  `popover` object.

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
