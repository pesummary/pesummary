=============================
Making public PESumamry pages
=============================

Producing the public facing summarypages showcasing the parameter estimation
results produced by the LIGO Virgo collaboration is as easy as simply adding
the `--public` command line argument to the 
`summarypages <../executables/summarypages.html>`_ executable. Below is an
example command line:

.. code-block:: console

   $ summarypages.py --webdir /home/albert.einstein/public_html/public_release \
                     --samples ./GW150914_result.h5 \
                     --config ./config.ini \
                     --psd H1:IFO0_psd.dat L1:IFO1_psd.dat V1:IFO2_psd.dat \
                     --calibration H1:IFO0_cal.dat L1:IFO1_cal.dat V1:IFO2_cal.dat \
                     --public \
                     --gw 
