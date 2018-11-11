# PESummary

Description
-------------

This code is designed to generate an html page to display the results from **all** Parameter Estimation codes.

Usage
------------- 

If only one approximant has been run, then you can generate a summary page with the following

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --webdir /home/c1737564/public_html/LVC/projects/bilby \
                 --baseurl https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby \
                 --samples1 /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                 --approximant1 IMRPhenomPv2
```

If you have run multiple approximants, then you can generate a summary page which compares both using the code

```bash
$ python main.py --email hoyc1@cardiff.ac.uk \
                 --webdir /home/c1737564/public_html/LVC/projects/bilby \
                 --baseurl https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby \
                 --number_of_waveforms two \
                 --samples1 /home/c1737564/projects/bilby/GW150914/IMRPhenomPv2/outdir/GW150914_result.h5 \
                 --samples2 /home/c1737564/projects/bilby/GW150914/IMRPhenomP/outdir/GW150914_result.h5 \
                 --approximant1 IMRPhenomPv2 \
                 --approximant2 IMRPhenomP
```
