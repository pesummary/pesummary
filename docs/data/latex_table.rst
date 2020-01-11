===================================
Making a latex table of the results
===================================

PESummary provides functions to make a latex table in order to show the results
stored in either a results file or a PESummary metafile. You are able to
specify which parameters you want included in the latex table as well as a
detailed description of what each parameter is. Below is an example
for how to make a latex table for the results stored in a result file:

.. code-block:: python

    >>> from pesummary.gw.file.read import read
    >>> f = read("result.hdf5")
    >>> description_mapping = {
    ...    "chirp_mass": "Detector frame chirp mass",
    ...    "mass_1": "Detector frame primary mass"
    ... }
    >>> f.to_latex_table(parameter_dict=description_mapping)
    \begin{table}[hptb]
    \begin{ruledtabular}
    \begin{tabular}{l c }
    Detector frame chirp mass & $35.04^{+8.00}_{-5.00}$\\
    Detector frame primary mass & $76.01^{+7.56}_{-0.45}$\\
    \end{tabular}
    \end{ruledtabular}
    \caption{}
    \end{table}
    >>> f.to_latex_table(save_to_file="table.tex", parameter_dict=description_mapping)

If you run on a PESummary metafile, then you are able to specify which set of
results you wish to include in the latex table by passing a list of labels.
For example:

.. code-block:: python

    >>> from pesummary.gw.file.read import read
    >>> f = read("posterior_samples.json")
    >>> description_mapping = {
    ...    "chirp_mass": "Detector frame chirp mass"
    ... }
    >>> f.to_latex_table(labels=["example1", "example2"], parameter_dict=description_mapping)
    \begin{table}[hptb]
    \begin{ruledtabular}
    \begin{tabular}{l c c }
     & example1 & example2\\
    \hline \\
    Detector frame chirp mass & $1.49^{1.49}_{1.49}$ & $1.49^{1.49}_{1.49}$\\
    \end{tabular}
    \end{ruledtabular}
    \caption{}
    \end{table}
