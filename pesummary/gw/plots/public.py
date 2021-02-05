#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.plots.latex_labels import latex_labels
from .latex_labels import GWlatex_labels, public_GWlatex_labels
from .main import _PlotGeneration as _GWPlotGeneration

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
latex_labels.update(GWlatex_labels)
latex_labels.update(public_GWlatex_labels)


class _PlotGeneration(_GWPlotGeneration):
    def __init__(
        self, savedir=None, webdir=None, labels=None, samples=None,
        kde_plot=False, existing_labels=None, existing_injection_data=None,
        existing_file_kwargs=None, existing_samples=None,
        existing_metafile=None, same_parameters=None, injection_data=None,
        result_files=None, file_kwargs=None, colors=None, custom_plotting=None,
        add_to_existing=False, priors={}, no_ligo_skymap=False,
        nsamples_for_skymap=None, detectors=None, maxL_samples=None,
        gwdata=None, calibration=None, psd=None,
        multi_threading_for_skymap=None, approximant=None,
        pepredicates_probs=None, include_prior=False, publication=False,
        existing_approximant=None, existing_psd=None, existing_calibration=None,
        existing_weights=None, weights=None, disable_comparison=False,
        linestyles=None, disable_interactive=False, disable_corner=False,
        publication_kwargs={}, multi_process=1, corner_params=None,
        preliminary_pages=False, expert_plots=False, checkpoint=False
    ):
        super(_PlotGeneration, self).__init__(
            savedir=savedir, webdir=webdir, labels=labels,
            samples=samples, kde_plot=kde_plot, existing_labels=existing_labels,
            existing_injection_data=existing_injection_data,
            existing_file_kwargs=existing_file_kwargs,
            existing_samples=existing_samples,
            existing_metafile=existing_metafile,
            same_parameters=same_parameters,
            injection_data=injection_data,
            result_files=result_files, file_kwargs=file_kwargs,
            colors=colors, custom_plotting=custom_plotting,
            add_to_existing=add_to_existing, priors=priors,
            no_ligo_skymap=no_ligo_skymap,
            nsamples_for_skymap=nsamples_for_skymap, detectors=detectors,
            maxL_samples=maxL_samples, gwdata=gwdata, calibration=calibration,
            psd=psd, multi_threading_for_skymap=multi_threading_for_skymap,
            approximant=approximant, pepredicates_probs=pepredicates_probs,
            include_prior=include_prior, publication=publication,
            existing_approximant=existing_approximant,
            existing_psd=existing_psd,
            existing_calibration=existing_calibration,
            existing_weights=existing_weights, weights=weights,
            disable_comparison=disable_comparison, linestyles=linestyles,
            disable_interactive=disable_interactive,
            disable_corner=disable_corner,
            publication_kwargs=publication_kwargs,
            multi_process=multi_process, corner_params=corner_params,
            preliminary_pages=preliminary_pages, expert_plots=expert_plots,
            checkpoint=checkpoint
        )
