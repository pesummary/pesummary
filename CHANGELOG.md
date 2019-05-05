## 0.1.6 [unreleased]

### Changed
- Python2.7: Change the yield statements in pesummary.core.file.one_format to
  make the code python2.7 compatible.
- setuptools: Change the setup.py file to import setup from `setuptools` rather
  than using `distutils.core` 

### Added
- PESummary now supports non GW specific results files
- Docker image for PESummary. Simply run `docker pull 08hoyc/pesummary:v0.1.6
- Example python script for running PESummary from within python shell
- Extract all information including psd and calibration information from the
  gw specific results file

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
