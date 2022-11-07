==================================
pesummary.gw.classification module
==================================

The :code:`pesummary.gw.classification` module allows for classification
probabilities to be generated directly from a set of posterior samples. We
calculate probability that the source is a binary black hole (BBH),
binary neutron star (BNS), neutron star black hole (NSBH) and/or lies
within the lower mass gap (MassGap) by interacting with the
`PEPredicates <https://git.ligo.org/will-farr/pepredicates>`_ package. We also
calculate the probability that the binary contains at least one neutron star
(HasNS) and the probability that the binary has a visible remnant
(HasRemnant) by interacting with the
`p-astro <https://git.ligo.org/lscsoft/p-astro>`_ package. :code:`pesummary`
provides 3 classes for calculating classification probabilities:
:code:`pesummary.gw.classification.PEPredicates`,
:code:`pesummary.gw.classification.PAstro` and
:code:`pesummary.gw.classification.Classify` as well as a helper
:code:`pesummary.gw.classification.classify` function. We discuss each of them
below.

`pesummary.gw.classification.PEPredicates`
------------------------------------------

We may calculate the probability that the binary is a BBH, BNS, NSBH and/or
lies within the lower mass gap by passing a set of posterior samples to the
:code:`pesummary.gw.classification.PEPredicates` class,

.. code-block:: python

    >>> from pesummary.gw.classification import PEPredicates
    >>> x = PEPredicates(
    ...    {
    ...        "mass_1_source": [20, 30], "mass_2_source": [10, 20],
    ...        "a_1": [0.5, 0.2], "a_2": [0.3, 0.1], "redshift": [0.4, 0.2]
    ...    }
    ... )
    >>> print(x.classification())
    {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0}

We may also see how these probabilities change if we reweigh the posterior
samples to a population inferred prior by passing the
:code:`population=True` kwarg,

.. code-block:: python

    >>> print(x.classification(population=True))
    {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0}

If we wish to calculate the probabilities for both the raw samples and the
reweighted posterior samples in a single command, we can use the
:code:`dual_classification()` method,

.. code-block:: python

    >>> print(x.dual_classification())
    {'default': {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0}, 'population': {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0}}

.. autoclass:: pesummary.gw.classification.PEPredicates
    :members:

`pesummary.gw.classification.PAstro`
------------------------------------

Similar to the :code:`pesummary.gw.classification.PEPredicates` class
the :code:`pesummary.gw.classification.PAstro` class can be used to calculate
the probability that the source has a neutron star and visible remnant by
passing a set of posterior samples,

.. code-block:: python

    >>> from pesummary.gw.classification import PAstro
    >>> x = PAstro(
    ...    {
    ...        "mass_1_source": [20, 30], "mass_2_source": [10, 20],
    ...        "a_1": [0.5, 0.2], "a_2": [0.3, 0.1], "redshift": [0.4, 0.2]
    ...    }
    ... )
    >>> print(x.classification())
    {'HasNS': 0.0, 'HasRemnant': 0.0}

We may again calculate the probabilities with samples reweighted to a population
prior with,

.. code-block:: python

    >>> print(x.classification(population=True))
    {'HasNS': 0.0, 'HasRemnant': 0.0}

and the combination can be printed with,

.. code-block:: python

    >>> print(x.dual_classification())
    {'default': {'HasNS': 0.0, 'HasRemnant': 0.0}, 'population': {'HasNS': 0.0, 'HasRemnant': 0.0}}

.. autoclass:: pesummary.gw.classification.PAstro
    :members:

`pesummary.gw.classification.Classify`
--------------------------------------

The :code:`pesummary.gw.classification.Classify` class combines the
:code:`pesummary.gw.classification.PEPredicates` and
:code:`pesummary.gw.classification.PAstro` classes into one and returns the
probability that the binary is a BBH, BNS, NSBH and/or lies within the lower
mass gap as well as probability that the source has a neutron star and visible
remnant. For example,

.. code-block:: python

    >>> from pesummary.gw.classification import Classify
    >>> x = Classify(
    ...    {
    ...        "mass_1_source": [20, 30], "mass_2_source": [10, 20],
    ...        "a_1": [0.5, 0.2], "a_2": [0.3, 0.1], "redshift": [0.4, 0.2]
    ...    }
    ... )
    >>> print(x.classification())
    {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0, 'HasNS': 0.0, 'HasRemnant': 0.0}
    >>> print(x.classification(population=True))
    {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0, 'HasNS': 0.0, 'HasRemnant': 0.0}
    >>> print(x.dual_classification())
    {'default': {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0, 'HasNS': 0.0, 'HasRemnant': 0.0}, 'population': {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0, 'HasNS': 0.0, 'HasRemnant': 0.0}}

.. autoclass:: pesummary.gw.classification.Classify
    :members:

`pesummary.gw.classification.classify`
--------------------------------------

The :code:`pesummary.gw.classification.classify` function provides an easy-to-use
interface to the :code:`classification` method provides by the
:code:`pesummary.gw.classification.Classify` class. For example,

. code-block:: python

    >>> from pesummary.gw.classification import classify
    >>> posterior = {
    ...     "mass_1_source": [20, 30], "mass_2_source": [10, 20],
    ...     "a_1": [0.5, 0.2], "a_2": [0.3, 0.1], "redshift": [0.4, 0.2]
    ... }
    >>> print(classify(posterior))
    {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0, 'HasNS': 0.0, 'HasRemnant': 0.0}
    >>> print(classify(posterior, population=True))
    {'BNS': 0.0, 'NSBH': 0.0, 'BBH': 1.0, 'MassGap': 0.0, 'HasNS': 0.0, 'HasRemnant': 0.0}

.. autofunction:: pesummary.gw.classification.classify
    :members:
