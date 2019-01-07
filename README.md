# PESummary

[![Coverage report](https://docs.ligo.org/charlie.hoy/pesummary/coverage_badge.svg)](https://docs.ligo.org/charlie.hoy/pesummary/htmlcov/index.html)

Installation
------------

To install PESummary into a virtual environment please run the following commands,

```bash
$ mkdir -p ~/virtualenvs
$ cd ~/virtualenvs
$ virtualenv pesummary
$ source pesummary/bin/activate
$ pip install pesummary
```

Description
-----------

This code is designed to generate an html page to display the results from **all** Parameter Estimation codes. If only a single approximant wants to be studied, then a summary page similar to the LALInference posterior pages will be generated. If you wish to study multiple approximants, then a single summary page for both approximants are generated. This includes both individual 1d histograms as well as the combined 1d histograms.

PESummary offers the following modules,

```python
pesummary.webpage
pesummary.plot
pesummary.utils
pesummary.one_format
```

The `pesummary.webpage` module offers functions and classes to generate and manipulate html pages. The function within the `page` class  allow the user to fully customise their webpage to their exact specifications. 

The `pesummary.plot` module offers functions to generate all posterior distributions and other helpful plots given the mcmc samples.

The `pesummary.utils` module offers helpful functions.

The `pesummary.one_format` module offers functions which will read in the outputs from either LALInference or BILBY and put them into a standard format. This is such that both codes can be used and compared with this package.

PESummary also offers the following executable,

```bash
$ summarypages.py
```

This executable uses functions from the above modules to produce a standard output page given the output from either the LALInference or BILBY codes. You are able to pass PESummary `n` output files and a single html page is produced which compares the posterior distributions from all output files.

Flags
-------------

The `summarypage.py` executable takes the following command line arguments:

```bash
$ summarypages.py --help

usage: summarypages.py [-h] [-w DIR] [-b DIR] [-s SAMPLES [SAMPLES ...]]
                       [-a APPROXIMANT [APPROXIMANT ...]]
                       [--email user@ligo.org] [--dump]
                       [-c CONFIG [CONFIG ...]] [--sensitivity]
                       [--add_to_existing] [-e EXISTING]
                       [-i INJ_FILE [INJ_FILE ...]]

Built-in functions, exceptions, and other objects. Noteworthy: None is the
`nil' object; Ellipsis represents `...' in slices.

optional arguments:
  -h, --help                show this help message and exit
  -w DIR, --webdir DIR      make page and plots in DIR
  -b DIR, --baseurl DIR     make the page at this url
  -s SAMPLES [SAMPLES ...], --samples SAMPLES [SAMPLES ...]
                            Posterior samples hdf5 file
  -a APPROXIMANT [APPROXIMANT ...], --approximant APPROXIMANT [APPROXIMANT ...]
                            waveform approximant used to generate samples
  --email user@ligo.org     send an e-mail to the given address with a link to the
                            finished page.
  --dump                    dump all information onto a single html page
  -c CONFIG [CONFIG ...], --config CONFIG [CONFIG ...]
                            configuration file associcated with each samples file.
  --sensitivity             generate sky sensitivities for HL, HLV
  --add_to_existing         add new results to an existing html page
  -e EXISTING, --existing_webdir EXISTING
                            web directory of existing output
  -i INJ_FILE [INJ_FILE ...], --inj_file INJ_FILE [INJ_FILE ...]
                            path to injetcion file
```

Examples
-------------

Single Approximant
-------------

If only one approximant has been run, then you can generate a summary page with the following

```bash
$ summarypages.py --email hoyc1@cardiff.ac.uk \
                  --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/one_approximant \
                  --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                  --approximant IMRPhenomPv2 \
                  --config /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/config.ini
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/one_approximant/home.html

Double Approximant
-------------

If multiple approximants have been run, then you can generate a single summary page with the following

```bash
$ summarypages.py --email hoyc1@cardiff.ac.uk \
                  --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/two_approximants \
                  --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                  --approximant IMRPhenomPv2 IMRPhenomP \
                  --config /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/config.ini /home/c1737564/projects/bilby/GW150914/IMRPhenomP/config.ini
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants/home.html

Existing html
-------------

If you have already generated a summary page using this code, then you are able to add another n approximants to the existing html pages. This is done using the following code,

```bash
$ summarypages.py --email hoyc1@cardiff.ac.uk \
                  --add_to_existing \
                  --existing_webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/add_to_existing \
                  --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                  --approximant IMRPhenomP \
                  --config /home/c1737564/projects/bilby/GW150914/IMRPhenomP/config.ini
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/add_to_existing/home.html. To generate the existing summary pages, I ran the following,

```bash
$ summarypages.py --email hoyc1@cardiff.ac.uk \
                  --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/add_to_existing \
                  --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                  --approximant IMRPhenomPv2 \
                  --config /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/config.ini
```

dump
-------------

If the `--dump` flag is specified, all plots are dumped to a single tab if only one approximant is used. To generate this, run,

```bash
$ summarypages.py --email hoyc1@cardiff.ac.uk \                                  
                  --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/one_approximant \
                  --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                  --approximant IMRPhenomPv2 \
                  --dump
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/one_approximant_dump/home.html

If n approximants are used, then there will be n+1 tabs, with the `home` tab showing all comparison plots, and individual approximant tabs showing the individual plots for each approximant. To generate this, run,

```bash
$ summarypages.py --email hoyc1@cardiff.ac.uk \                                  
                  --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/two_approximants \
                  --baseurl https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants \
                  --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                  --approximant IMRPhenomPv2 IMRPhenomP \
                  --dump
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants_dump/home.html
