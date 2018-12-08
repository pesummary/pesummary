# PESummary

Description
-------------

This code is designed to generate an html page to display the results from **all** Parameter Estimation codes. If only a single approximant wants to be studied, then a summary page similar to the LALInference posterior pages will be generated. If you wish to study multiple approximants, then a single summary page for both approximants are generated. This includes individual 1d histograms as well as the combined 1d histograms. 

By default, the html page uses a multi tab approach as I feel it is more user friendly. However, this can be changed by adding the `--dump` flag. This will then dump all results onto a single tab.

Single Approximant
-------------

If only one approximant has been run, then you can generate a summary page with the following

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
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
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/two_approximants \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomPv2 IMRPhenomP \
                 --config /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/config.ini --config /home/c1737564/projects/bilby/GW150914/IMRPhenomP/config.ini
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants/home.html

Existing html
-------------

If you have already generated a summary page using this code, then you are able to add another n approximants to the existing html pages. This is done using the following code,

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --add_to_existing \
                 --existing_webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/add_to_existing \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomP \
                 --config /home/c1737564/projects/bilby/GW150914/IMRPhenomP/config.ini
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/add_to_existing/home.html. To generate the existing summary pages, I ran the following,

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/add_to_existing \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomPv2 \
                 --config /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/config.ini
```

dump
-------------

If the `--dump` flag is specified, all plots are dumped to a single tab if only one approximant is used. To generate this, run,

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \                                  
                 --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/one_approximant \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomPv2 \
                 --dump
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/one_approximant_dump/home.html

If n approximants are used, then there will be n+1 tabs, with the `home` tab showing all comparison plots, and individual approximant tabs showing the individual plots for each approximant. To generate this, run,

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \                                  
                 --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/two_approximants \
                 --baseurl https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomPv2 IMRPhenomP \
                 --dump
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants_dump/home.html
