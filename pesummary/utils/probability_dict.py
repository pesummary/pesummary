# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.samples_dict import SamplesDict
from pesummary.utils.dict import Dict
from pesummary.utils.pdf import DiscretePDF, DiscretePDF2D
from pesummary.core.plots.latex_labels import latex_labels
from pesummary.gw.plots.latex_labels import GWlatex_labels

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

latex_labels.update(GWlatex_labels)


class ProbabilityDict(Dict):
    """Class to store integers with discrete probabilities for multiple
    parameters.

    Parameters
    ----------
    args: dict
        dictionary containing the discrete probabilities for each parameter.
        Key should be the parameter name and value should be a 2d array of
        length 2. First element integers and second element probabilities
        corresponding to integers. See pesummary.utils.pdf.DiscretePDF for more
        details

    Methods
    -------
    rvs:
        randomly draw samples for each distribution in ProbabilityDict

    Examples
    --------
    >>> from pesummary.utils.probability_dict import ProbabilityDict
    >>> numbers = [1,2,3,4]
    >>> probabilities = [0.1, 0.2, 0.3, 0.4]
    >>> pdf = ProbabilityDict({"parameter": [numbers, probabilities]})
    >>> print(type(pdf["parameter"]))
    <class 'pesummary.utils.pdf.DiscretePDF'>
    >>> print(pdf["parameter"].probs)
    [0.1, 0.2, 0.3, 0.4]
    """
    def __init__(self, *args, logger_warn="warn", autoscale=False):
        super(ProbabilityDict, self).__init__(
            *args, value_class=DiscretePDF, logger_warn=logger_warn,
            make_dict_kwargs={"autoscale": autoscale}, latex_labels=latex_labels
        )

    @property
    def plotting_map(self):
        existing = super(ProbabilityDict, self).plotting_map
        modified = existing.copy()
        modified.update(
            {
                "marginalized_posterior": self._marginalized_posterior,
                "hist": self._marginalized_posterior,
                "pdf": self._analytic_pdf,
            }
        )
        return modified

    def rvs(
        self, size=None, parameters=None, interpolate=True, interp_kwargs={}
    ):
        """Randomly draw samples from each distribution in ProbabilityDict

        Parameters
        ----------
        size: int
            number of samples to draw
        parameters: list, optional
            list of parameters you wish to draw samples for. If not provided
            draw samples for all parameters in ProbabilityDict. Default None
        interpolate: Bool, optional
            if True, interpolate the discrete probabilities and then draw
            samples from the continous distribution. Default True
        interp_kwargs: dict, optional
            kwargs to pass to the DiscretePDF.interpolate() method
        """
        if size is None:
            raise ValueError(
                "Please specify the number of samples you wish to draw from "
                "the interpolated distributions"
            )
        if parameters is not None:
            if any(param not in self.parameters for param in parameters):
                raise ValueError(
                    "Unable to draw samples because not all parameters have "
                    "probabilities stored in ProbabilityDict. The list of "
                    "available parameters are: {}".format(
                        ", ".join(self.parameters)
                    )
                )
        else:
            parameters = self.parameters

        mydict = {}
        for param in parameters:
            if interpolate:
                mydict[param] = self[param].interpolate(**interp_kwargs).rvs(
                    size=size
                )
            else:
                mydict[param] = self[param].rvs(size=size)
        return SamplesDict(mydict)

    def _marginalized_posterior(
        self, parameter, size=10**5, module="core", **kwargs
    ):
        """Draw samples from the probability distribution and histogram them
        via the `pesummary.core.plots.plot._1d_histogram_plot` or
        `pesummary.gw.plots.plot._1d_histogram_plot`

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to plot
        size: int, optional
            number of samples to from the probability distribution to histogram
        module: str, optional
            module you wish to use for the plotting
        **kwargs: dict
            all additional kwargs are passed to the `_1d_histogram_plot`
            function
        """
        samples = SamplesDict({parameter: self[parameter].rvs(size=size)})
        return samples.plot(parameter, type="hist", module=module, **kwargs)

    def _analytic_pdf(self, parameter, **kwargs):
        """Wrapper for the `pesummary.core.plots.plot._1d_analytic_plot
        function

        Parameters
        ----------
        parameter: str
            name of the parameter you wish to plot
        **kwargs: dict, optional
            all additional kwargs are passed ot the `_1d_analytic_plot` function
        """
        from pesummary.core.plots.plot import _1d_analytic_plot

        return _1d_analytic_plot(
            parameter, self[parameter].x, self[parameter].probs,
            self.latex_labels[parameter], **kwargs
        )


class ProbabilityDict2D(Dict):
    """Class to store the 2D discrete probability distributions for multiple
    parameters

    Parameters
    ----------
    args: dict
        dictionary containing the discrete probabilities for 2 or more
        parameters. Key should be 2 parameter names joined with `_` and
        value should be an array of length 3. First element integers for
        x, second element integers for y and third element the 2D discrete
        probability distribution for xy. See
        pesummary.utils.pdf.DiscretePDF2D for more details

    Methods
    -------
    plot:
        generate a plot to display the stored probability density functions

    Examples
    --------
    >>> from  pesummary.utils.probability_dict.ProbabilityDict2D
    >>> x = [-1., 0., 1.]
    >>> y = [-1., 0., 1.]
    >>> prob_xy = [
    ...     [1./9, 1./9, 1./9],
    ...     [1./9, 1./9, 1./9],
    ...     [1./9, 1./9, 1./9]
    ... ]
    >>> pdf = ProbabilityDict2D({"x_y": [x, y, prob_xy]})
    >>> print(type(pdf["x_y"]))
    <class 'pesummary.utils.pdf.DiscretePDF2D'>
    """
    def __init__(self, *args, logger_warn="warn", autoscale=False):
        super(ProbabilityDict2D, self).__init__(
            *args, value_class=DiscretePDF2D, logger_warn=logger_warn,
            make_dict_kwargs={"autoscale": autoscale}, latex_labels=latex_labels
        )

    @property
    def plotting_map(self):
        existing = super(ProbabilityDict2D, self).plotting_map
        modified = existing.copy()
        modified.update(
            {
                "2d_kde": self._2d_kde,
                "triangle": self._triangle,
                "reverse_triangle": self._reverse_triangle
            }
        )
        return modified

    def _2d_kde(self, parameter, **kwargs):
        """Generate a 2d contour plot showing the probability distribution for
        xy.

        Parameters
        ----------
        parameter: str
            the key you wish to plot
        """
        from pesummary.core.plots.publication import analytic_twod_contour_plot

        if "levels" in kwargs.keys():
            levels = self[parameter].probs_xy.level_from_confidence(
                kwargs["levels"]
            )
            kwargs["levels"] = levels
        return analytic_twod_contour_plot(
            self[parameter].x, self[parameter].y, self[parameter].probs,
            **kwargs
        )

    def _base_triangle(self, parameter, function, **kwargs):
        pdf = self[parameter].marginalize()
        if "levels" in kwargs.keys():
            levels = pdf.probs_xy.level_from_confidence(
                kwargs["levels"]
            )
            kwargs["levels"] = levels
        return function(
            pdf.x, pdf.y, pdf.probs_x.probs, pdf.probs_y.probs,
            pdf.probs_xy.probs, **kwargs
        )

    def _triangle(self, parameter, **kwargs):
        """Generate a triangle plot showing the probability distribution for
        x, y and xy. The probability distributions for x and y are found
        through marginalization

        Parameters
        ----------
        parameter: str
            the key you wish to plot
        """
        from pesummary.core.plots.publication import analytic_triangle_plot

        return self._base_triangle(
            parameter, analytic_triangle_plot, **kwargs
        )

    def _reverse_triangle(self, parameter, **kwargs):
        """Generate a reverse triangle plot showing the probability distribution
        for x, y and xy. The probability distributions for x and y are found
        through marginalization

        Parameters
        ----------
        parameter: str
            the key you wish to plot
        """
        from pesummary.core.plots.publication import (
            analytic_reverse_triangle_plot
        )

        return self._base_triangle(
            parameter, analytic_reverse_triangle_plot, **kwargs
        )
