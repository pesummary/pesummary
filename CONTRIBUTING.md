# Contributing to PESummary

## Getting started
PESummary lives in a git repository which is hosted here:
https://git.ligo.org/lscsoft/pesummary. If you haven't already, you should
[fork](https://docs.gitlab.com/ee/gitlab-basics/fork-project.html) this
repository and clone your fork to your local machine, 

```bash
$ git clone git@git.ligo.org:albert.einstein/pesummary.git
```

replacing the SSH url to that of your fork. You can then install `PESummary` by
running,

```bash
$ cd pesummary
$ python setup.py install
```

which will install `PESummary`. For further instructions on how to install
PESummary please visit the [docs](https://docs.ligo.org/lscsoft/pesummary/installation.html).

## Update your fork
If you already have a fork of `PESummary`, and are starting work on a new
project you can link your clone to the main (`lscsoft`) repository and pull in
changes that have been merged since the time you created your fork, or last
updated by following the instructions below:

1. Link your fork to the main repository:

```bash
$ cd pesummary
$ git remote add lscsoft https://git.ligo.org/lscsoft/pesummary
```

2. Fetch new changes from the `lscsoft` repo,

```bash
$ git fetch lscsoft
```

3. Merge in the changes,

```bash
$ git merge lscsoft/master
```

## Reporting issues
All issues should be reported through the
[issue workflow](https://docs.gitlab.com/ee/user/project/issues/). When
reporting an issue, please include as much detail as possible to reproduce the
error, including information about your operating system and the version of
code. If possible, please include a brief, self-contained code example that
demonstrates the problem.

## Merge Requests
If you would like to make a change to the code then this should be done through
the [merge-request workflow](https://docs.gitlab.com/ee/user/project/merge_requests/).
We recommend that before starting a merge request, you should open an issue
laying out what you would like to do. This lets a conversation happen early in
case other contributors disagree with what you'd like to do or have ideas
that will help you do it.

Once you have made a merge request, expect a few comments and questions from
other contributors who also have suggestions. Once all discussions are resolved,
core developers will approve the merge request and then merge it into master.

All merge requests should aim to either add one feature, solve a bug, address 
some stylistic issues, or add to the documentation. If multiple changes are
lumped together, this makes it harder to review.

All merge requests should also be recorded in the CHANGELOG.md.
This just requires a short sentence describing describing the change that you
have made.

## Creating a new feature branch
All changes should be developed on a feature branch, in order to keep them
separate from other work, simplifying review and merge once the work is done.

To create a new feature branch:

```bash
$ git fetch lscsoft
$ git checkout -b my-new-feature lscsoft/master
```

## Pre-commit hook
PEP8 checks can be run automatically as part of a pre-commit hook. In order to
install this pre-commit hook, run the following commands,

```bash
$ cd pesummary
$ pip install pre-commit
$ pre-commit install
$ pre-commit install --hook-type pre-commit
```

This will then run [black](https://black.readthedocs.io/en/stable/) prior to
every commit and only commit the reformatted files. A successful pre-commit hook
will show a message like:

```bash
black....................................................................Passed
[pre-commit-hook 201237f] commit message
 1 file changed, 6 insertions(+)
```

A failed pre-commit hook will show a message like:

```bash
black....................................................................Failed
hookid: black

error: cannot format ...
```

For a failed pre-commit hook, the file will still be commited, but will not be
PEP8 compatible.

## Unit tests
Unit tests and code coverage measurement are run automatically as part of the
continuous integration (CI) for every branch and for every merge request. New
code contributions must have 100% test coverage. Modifications to existing code
must not decrease test coverage. In order to run unit tests, run the following
commands,

```bash
$ cd pesummary
$ pip install optional_requirements.txt
$ covarage run -m pytest tests/
$ coverage html
```

This will save a coverage report that you can view in a web browser by opening
the file,

```bash
$ open html/index.html
```

## Code style
Code should be written in the [PEP8](https://www.python.org/dev/peps/pep-0008/)
style. To check code style, run the following commands,

```bash
$ cd pesummary
$ pip install optional_requirements.txt
$ flake8 .
```

## Documentation
Documentation strings should be written in the
[NumpyDoc style](https://numpydoc.readthedocs.io/en/latest/).
