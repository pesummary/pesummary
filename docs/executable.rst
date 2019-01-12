===============
Using PESummary
===============

PESummary executable
----------------------

:code:`pesummary` offers an executable which is available after following the `installation instructions <installation.rst>`_. This executable is named :code:`summarypages.py`. To see help for this executable, run:

.. code-block:: console

   $ summarypages.py --help

Running summarypages.py
-----------------------

To run :code:`pesummary`, you first need to generate a results file from either `lalinference`_ or `bilby`_.

.. _lalinference:

.. _bilby:

Once you have a results file (for clarity lets call it :code:`results_file.h5`), you then need to decide where you would like to store the webpages. If you are working on an LDG cluster then please use your :code:`public_html` directory. You can run :code:`summarypages.py` with the following,

.. code-block:: console

   $ summarypages.py --webdir /home/albert.einstein/public_html/LVC/projects --samples ./results_file.h5

This will then produce a web directory structure as follows:

.. code-block:: console

   home.html
   config/
     -> config.ini
   css/
     -> image_styles.css
     -> side_bar.css
   js/
     -> combine_corner.js
     -> grab.js
     -> modal.js
     -> multi_dropbar.js
     -> multiple_posteriors.js
     -> search.js
     -> side_bar.js
   plots/
     ->
   samples/
     -> results_file_posterior_samples.h5
