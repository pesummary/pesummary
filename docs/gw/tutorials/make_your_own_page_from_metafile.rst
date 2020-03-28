========================================================
Making your own Summary pages from a PESummary meta file
========================================================

All information to generate the PESummary summary pages is stored in the
PESummary meta file. As a result, you are able to reproduce exactly the same
summary pages by feeding PESummary the meta file as input. This tutorial
shows you how to how to do this step by step.

Firstly, you need to download the PESummary meta file from the summary pages.
This can be done by navigating to the `downloads` page. Secondly, you need
to run the following:

.. code-block:: console

    $ summarypages.py --webdir /home/albert.einstein/public_html/copy \
                      --samples ./posterior_samples.json \
                      --gw
