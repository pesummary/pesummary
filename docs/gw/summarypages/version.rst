================
The version page
================

The `version page <https://pesummary.github.io/GW190412/html/Version.html>`_
displays the version information for each of the files provided to pesummary
as well as the packages installed in the environment in which the `summarypages`
executable was run.

For this case, the environment was a virtualenv and all packages installed via
`pypi`. For this case, the package table is 2 columns -- the first column
displaying the package name, and the second column giving the version. At the
bottom of the package table, we see that we can export this table to a
`requirements.txt` file, allowing for the user to create the exact same
environment.

If the environment was actually a `conda` environment, the package table would
be 4 columns -- the first column showing the package name, the second giving
the package version, third showing the channel from which it was installed from,
and finally the fourth showing the build string. For this case, the table
may be exported as a `conda yml` file, allowing for the user to easily recreate
the exact same `conda` environment.

This allows for complete reproducability.
