========
Examples
========

Single approximant
------------------

If only one approximant has been run, then you can generate a summary page with the following,

.. code-block:: console

   $ summarypages.py --email albert.einstein@ligo.org \
                     --webdir /home/albert.einstein/public_html/one_approximant \
                     --samples ./GW150914_result.h5

An example of this is shown `here <https://docs.ligo.org/charlie.hoy/pesummary_examples/single/home.html>`_.

Double approximant
------------------

If multiple approximants have been run, then you can generate a single summary page with the following,

.. code-block:: console

   $ summarypages.py --email albert.einstein@ligo.org \
                     --webdir /home/albert.einstein/public_html/two_approximants \
                     --samples ./IMRPhenomPv2/GW150914_result.h5 ./IMRPhenomP/GW150914_result.h5

An example of this is shown `here <https://docs.ligo.org/charlie.hoy/pesummary_examples/double/home.html>`_.

Existing html
-------------

If you have already generated a summary page using :code:`pesummary`, then you are able to add another n approximants to this existing html page. This is done using the following code,

.. code-block:: console

   $ summarypages.py --email albert.einstein@ligo.org \
                     --add_to_existing \
                     --existing_webdir /home/albert.einstein/public_html/add_to_existing \
                     --samples ./GW150914_result.h5

An example of this is shown `here <https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/add_to_existing/home.html>`_.

.. note::
   To generate the the original webpage I ran the following code,

   .. code-block:: console
      
      $ summarypages.py --email albert.einstein@ligo.org \
                        --webdir /home/albert.einstein/public_html/add_to_existing \
                        --samples ./GW150914_result.h5
