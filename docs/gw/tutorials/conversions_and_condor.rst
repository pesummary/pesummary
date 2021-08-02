========================
Conversions and HTCondor
========================

Some of the `conversions <../Conversion.html>`_ available in :code:`pesummary`
can be time-consuming. This is why many of the conversions can be parallelised
through Python's
`multiprocessing module <https://docs.python.org/3/library/multiprocessing.html>`_.
However, it is not always ideal to highly parallelise the conversions through
the multiprocessing module. This is because if, for example, you wanted to run
on 100 CPUs and submit this job to a scheduler e.g.
`HTCondor <https://research.cs.wisc.edu/htcondor/>`_, 100 CPUs will
need to be available simultaneously before the job starts. This means that it is
possible for a highly parallised job to take a similar wall time as one which
runs in series since now the job will spend longer in the queue. To avoid
this :code:`pesummary` provides all of the tools needed to efficiently interact
with schedulers. This tutorial details how :code:`pesummary` can efficiently
interact with the `HTCondor` schedular and reduce wall time on time-consuming
conversions by parallelising over many CPUs. In this example we consider the
spin evolution to negative infinity. We will interact with `HTCondor` by writing a
`Directed acyclic graph (DAG) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_
which uses the
`summarysplit <../../core/cli/summarysplit.html>`_,
`summarycombine <../../core/cli/summarycombine.html>`_ and
`summarycombine_posteriors <../../core/cli/summarycombine_posteriors.html>`_
executables. 

Designing the DAG
-----------------

Our job needs to:

#. Split the input file into N files
#. Run the conversion module on each file and evolve the spins to negative infinity
#. Combine the N converted files into a single file

Step 1 can be achieved with the `summarysplit <../../core/cli/summarysplit.html>`_
executable, e.g.,

.. code-block:: bash

    $ summarysplit --multi_process 4 --samples PATH_TO_SAMPLES \
                   --outdir ./output --file_format dat

Step 2 can be achieved with the
`summarycombine <../../core/cli/summarycombine.html>`_ executable, e.g.,

.. code-block:: bash

    $ summarycombine --webdir ./output/0 \
                     --samples ./output/split_posterior_samples_0.dat
                     --evolve_spins_backwards hybrid_orbit_averaged \
                     --labels evolve --gw


Finally we can combine all of the individually converted files with the
`summarycombine_posteriors <../../core/cli/summarycombine_posteriors.html>`_
executable, e.g.,

.. code-block:: bash

    $ summarycombine_posteriors --outdir ./output --filename combined_samples.dat \
                                --use_all --file_format dat --labels evolve \
                                --samples ./output/*/samples/evolve_pesummary.dat \

Writing the DAG
---------------

The following code snippet creates a dag which a) splits the input
file into N seperate analyses where N is the number of posterior samples
contained with the input file, b) evolve each sample to negative infinity and
c) combine all files into a single dat.

.. code-block:: python

    # Licensed under an MIT style license -- see
    # https://git.ligo.org/lscsoft/pesummary/-/blob/master/LICENSE.md

    import os
    import glob
    import argparse
    import numpy as np
    from pesummary.io import read
    from pesummary.core.webpage.main import _WebpageGeneration

    __author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


    def command_line():
        """Generate an Argument Parser object to control command line options
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d", "--dir", required=True, help="directory of split posterior samples",
        )
        parser.add_argument(
            "-f", "--f_ref", help="Reference frequency to use for spin evolution in Hz",
            default=20
        )
        parser.add_argument(
            "--filename", help="dat file containing posterior samples to evolve",
            default=None
        )
        return parser


    def create_submit_file(
        filename, executable, arguments, stage, cpus=1, memory=2048,
        accounting_group="ligo.prod.o3.cbc.pe.lalinference",
        accounting_user="charlie.hoy"
    ):
        """Write a basic submit file for HTCondor

        Parameters
        ----------
        filename: str
            name of the submit file
        executable: str
            the executable you wish to use
        arguments: str
            a space seperated list of arguments you wish to pass to the executable.
            If you do not wish to pass any arguments simply pass an empty string
        stage: str
            a unique string which is used when naming the log, error and output files
        cpus: int, optional
            number of cpus to use when executing the job. Default 1
        memory: int, optional
            amount of memory to allocate for the job in MB. Default 2048 MB
        accounting_group: str, optional
            accounting group to use for the job. Default 'ligo.prod.o3.cbc.pe.lalinference'
        accounting_user: str, optional
            accounting user to use for the job. Default 'charlie.hoy'
        """
        with open(filename, "w") as f:
            f.writelines(
                [
                    'universe = vanilla\n',
                    f'executable = {executable}\n',
                    f'arguments = "{arguments}"\n',
                    f'request_cpus = {cpus}\n',
                    'getenv = True\n',
                    'requirements = \n',
                    f'request_memory = {memory}\n',
                    f'log = $(webdir)/{stage}.log\n',
                    f'error = $(webdir)/{stage}.err\n',
                    f'output = $(webdir)/{stage}.out\n',
                    f'accounting_group = {accounting_group}\n',
                    f'accounting_group_user = {accounting_user}\n',
                    'notification = never\n',
                    'queue 1\n'
                ]
            )
        return


    def main():
        """Top level interface for `make_dag.py`
        """
        parser = command_line()
        opts = parser.parse_args()
        _base_dir = os.path.abspath(opts.dir)

        # write submit files
        create_submit_file(
            os.path.join(_base_dir, "split.sub"),
            _WebpageGeneration.get_executable('summarysplit'),
            '--multi_process 4 --samples $(samples) --outdir $(webdir) --file_format dat',
            'split', cpus=4, memory=8192
        )
        create_submit_file(
            os.path.join(_base_dir, "generate.sub"),
            _WebpageGeneration.get_executable('summarycombine'), (
                '--webdir $(webdir) --samples $(samples) --evolve_spins_backwards '
                'hybrid_orbit_averaged --labels $(label) --gw --no_ligo_skymap '
                '--f_ref $(f_ref)'
            ), 'evolve', cpus=1, memory=16384
        )
        create_submit_file(
            os.path.join(_base_dir, "combine.sub"),
            _WebpageGeneration.get_executable('summarycombine_posteriors'), (
                '--samples $(webdir)/*/samples/evolve_pesummary.dat '
                '--use_all --file_format dat --filename combined_samples.dat '
                '--labels evolve --outdir $(webdir)'
            ), 'combine', cpus=1, memory=4096
        )

        if opts.filename is None:
            _base_file = f'{os.path.join(_base_dir, "extracted_samples.dat")}'
        else:
            _base_file = opts.filename
        # write the dag
        with open(os.path.join(_base_dir, "generate_hybrid_orbit_averaged.dag"), "w") as f:
            f.writelines(
                [
                    f'JOB SPLIT {os.path.join(_base_dir, "split.sub")}\n',
                    f'VARS SPLIT samples="{_base_file}" webdir="{_base_dir}"\n'
                ]
            )
            _open = read(_base_file).samples_dict
            N_samples = _open.number_of_samples
            for num in range(N_samples):
                ff = os.path.join(_base_dir, f"split_posterior_samples_{num}.dat")
                f.writelines(
                    [
                        f'JOB GENERATE{num} {os.path.join(_base_dir, "generate.sub")}\n', (
                            f'VARS GENERATE{num} label="evolve" samples="{ff}" '
                            f'webdir="{os.path.join(_base_dir, str(num))}" '
                            f'f_ref="{opts.f_ref}"\n'
                        )
                    ]
                )
                try:
                    os.makedirs(os.path.join(_base_dir, str(num)))
                except FileExistsError:
                    pass
            f.writelines(
                [
                    "\n", f'JOB COMBINE {os.path.join(_base_dir, "combine.sub")}\n',
                    f'VARS COMBINE webdir="{_base_dir}"\n'
                ]
            )
            for num in range(N_samples):
                f.writelines([f"PARENT SPLIT CHILD GENERATE{num}\n"])
                f.writelines([f"PARENT GENERATE{num} CHILD COMBINE\n"])
        return


    if __name__ == "__main__":
        main()


We can then create and submit the dag with the following,

.. code-block:: bash

    $ python make_dag.py --dir ./output --filename ./posterior_samples.dat
    $ condor_submit_dag ./output/generate_hybrid_orbit_averaged.dag

The evolved and post-processed posterior samples can then be found at
:code:`./output/combined_samples.dat`.
