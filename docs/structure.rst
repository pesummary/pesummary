=============
Code overview
=============

Python Modules
--------------

At the top-level, the :code:`pesummary` python package provides several modules as visualised here:

.. graphviz::

   digraph {
         "pesummary" -> ".webpage";
         "pesummary" -> ".utils";
         "pesummary" -> ".plot";
         "pesummary" -> ".one_format";
            }

each module (e.g., :code:`pesummary.webpage.webpage`) serves a different purpose. On this page, we will give a short description of each module.

pesummary.webpage
-----------------

The :code:`pesummary.webpage` module offers the submodule :code:`pesummary.webpage.webpage`. This contains functions and classes to generate an manipulate html webpages. The functions within the :code:`pesummary.webpage.webpage.page` class allow the user to fully customise their webpage to their exact specifications.

pesummary.utils
---------------

The :code:`pesummary.utils` module offers the submodule :code:`pesummary.utils.utils`. This submodule contains helpful functions for making directories and determining the url based on the web directory.

pesummary.plot
--------------

The :code:`pesummary.plot` module offers the submodule :code:`pesummary.plot.plot`. This submodule contains functions for generating all plots which will be generated and inserted in the webpages.

pesummary.one_format
--------------------

The :code:`pesummary.one_format` module offers the submodule :code:`pesummary.one_format.data_format`. This submodule contains functions to analyse the given posterior samples. Here, we identify if either :code:`lalinference` or :code:`bilby` generated the input and extract the samples from the file. We then check which parameters have posterior samples and calculate the posterior samples for parameters that are not included. We then store all information in a standard format that can be read by :code:`pesummary`.
