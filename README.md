# PESummary

Description
-------------

This code is designed to generate an html page to display the results from **all** Parameter Estimation codes. If only a single approximant wants to be studied, then a summary page similar to the LALInference posterior pages will be generated. If you wish to study multiple approximants, then a single summary page for both approximants are generated. This includes individual 1d histograms as well as the combined 1d histograms. 

Single Approximant
-------------

If only one approximant has been run, then you can generate a summary page with the following

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/one_approximant \
                 --baseurl https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilbyGW150914/one_approximant \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomPv2
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/one_approximant/home.html

Double Approximant
-------------

If multiple approximants have been run, then you can generate a single summary page with the following

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --webdir /home/c1737564/public_html/LVC/projects/bilby/GW150914/two_approximants \
                 --baseurl https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants \
                 --samples /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                 --approximant IMRPhenomPv2 IMRPhenomP
```

An example of this is shown here: https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/two_approximants/home.html
