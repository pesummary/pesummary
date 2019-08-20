## 0.1.8 [unreleased]

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
