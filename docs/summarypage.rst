=========================
Understanding the webpage
=========================

Basic layout
------------

All webpages produced by :code:`PESummary` have the following structure,

.. literalinclude:: images/navbar.pdf

.. code-block:: console

   home.html
     -> Results_file1
       -> Corner
       -> Config
       -> 1d histogram pages
         -> multiple
         -> A-E
           -> amplitude

.. note::
    
    If you run with a GW specific results file, then the 1d histogram pages are sorted according to their common parameter. For instance, mass1, mass2, mass_ratio are all sorted under the mass heading.

This webpage opts for a multiple tab approach for each of navigation. Below we will discuss each section and give examples of what the page looks like. 

.. note::
   As a helpful reminder, all pages have a footer displaying when the given page was generated and who by, the location of the run directory and the exact command that was run to generate this page. You will also notice 3 links. These links guide you towards the code that was used to generate the webpage, the issue tracker where we encourge you to make an issue if there is something missing or something wrong with the webpages, and finally a link to this documentation for help and guidance.

   .. image:: images/footer.png

Homepage
--------

Approximant homepage
--------------------

.. image:: images/approximant_homepage.png

The approximant homepage offers a first glimpse of your analysis. Useful plots are shown in the container and a summary table containing the mean, median, maxL value and standard deviation of the posteriors is given below. All approximant homepages show the :code:`mass_1` posterior, :code:`mass_2` posterior, :code:`distance` posterior, skymap, maximum likelihood waveform, :code:`inclination` posterior and finally the two spin posteriors for both black holes. The skymap is produced from the samples for :code:`ra` and :code:`dec` and displayed are the 90%, 70% and 50% credible regions.

Clicking on any of the plots opens a popup with a zoomed in version of that plot. You are then able to either click on the left/right hand sides of this popup to show the next image, or simply use the arrow keys on your keyboard.

As you will have noticed, all posterior plots have a title informing you of the median as well as the :code:`90%` credible region. For your convenience, the :code:`90%` credible region is also shown on the plot.

Corner
######

.. image:: images/corner.png

The corner plot is a fully interactive method for finding correlations between parameters that you are interested in. To operate it, either type the parameters that you are interested in into the search bar or open the side bar and tick the parameters that you are interested in. Once you have done this, simply click the submit button and a corner plot for those parameters will be generated. 

.. note::
   The search bar must be empty if you choose to tick the options in the side bar

:code:`PESummary` also offers preselected `popular choices` which can also be clicked. Once you are happy with your custom corner plot, you are able to download it to you local machine by simply right clicking on the corner plot and navigating to `Save Image As`. For further details on how the corner plot is generated, please refer to `generating a custom corner plot using a cut and paste technique <cutom_corner_plot.html>`_.

Config
######

.. image:: images/config.png

If a configuration file was passed to :code:`PESummary` (see `Using PESummary <executable.html>`_ for details of the named arguments), :code:`PESummary` will display the contents directly on the webpage with correct highlighting. This is such that all information about how the run was generated is in one place.

1d histogram pages
##################

.. image:: images/1d_histogram.png

The 1d histogram pages offers two different ways of sharing the information. You can either view `multiple` posterior distributions at once by clicking on the `multiple` tab, or you can look at the posteriors for a single event by navigating to their location in the navigation bar. As you will notice, all parameters are categorised by groups. In the `masses` group resides all parameters that are directly related to the mass of the system. For instance, the individual mass components, the total mass, the chirp mass, the mass ratio and the symmetric mass ratio are all shown here. I have tried to make the headings as intuitive as possible so you can easily navigate to the plot that you are interested in.

The mutliple tab opens to a page that is similar to the corner window as shown above. You are again able to either, type the parameters that you are interested in seeing, tick the parameters in the side bar or click one of the preselected options. Images for these posteriors will then be shown when they have all loaded. You are also able to click the `all` button. Here all posteriors available will be displayed in the same categorised order as the navigation bar.

The individual parameter pages display all the information that an expert needs to identify the reliability of the run. The autocorrelation function is plotted (showing how correlated the samples are) and a scatter plot showing the sample distribution is also shown. In most cases, you are interested in seeing the given posterior distribution, which is why this image is larger than the other two. If you have injected a given waveform, then a red dashed line will appear on the plot showing the injected value.  

Comparison pages
----------------
