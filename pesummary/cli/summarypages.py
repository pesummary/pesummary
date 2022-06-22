#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger, gw_results_file

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class WebpageGeneration(object):
    """Wrapper class for _GWWebpageGeneration and _CoreWebpageGeneration

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default", gw=False):
        self.inputs = inputs
        self.colors = colors
        self.gw = gw
        self.generate_webpages()

    def generate_webpages(self):
        """Generate all plots for all result files passed
        """
        logger.info("Starting to generate webpages")
        if self.gw and self.inputs.public:
            object = _PublicGWWebpageGeneration(self.inputs, colors=self.colors)
        elif self.gw:
            object = _GWWebpageGeneration(self.inputs, colors=self.colors)
        else:
            object = _CoreWebpageGeneration(self.inputs, colors=self.colors)
        object.generate_webpages()
        logger.info("Finished generating webpages")


class _CoreWebpageGeneration(object):
    """Class to generate all webpages for all result files with the Core module

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default"):
        from pesummary.core.webpage.main import _WebpageGeneration
        key_data = inputs.grab_key_data_from_result_files()
        self.webpage_object = _WebpageGeneration(
            webdir=inputs.webdir, samples=inputs.samples, labels=inputs.labels,
            publication=inputs.publication, user=inputs.user, config=inputs.config,
            same_parameters=inputs.same_parameters, base_url=inputs.baseurl,
            file_versions=inputs.file_version, hdf5=inputs.hdf5, colors=inputs.colors,
            custom_plotting=inputs.custom_plotting,
            existing_labels=inputs.existing_labels,
            existing_config=inputs.existing_config,
            existing_file_version=inputs.existing_file_version,
            existing_injection_data=inputs.existing_injection_data,
            existing_samples=inputs.existing_samples,
            existing_metafile=inputs.existing,
            existing_file_kwargs=inputs.existing_file_kwargs,
            existing_weights=inputs.existing_weights,
            add_to_existing=inputs.add_to_existing, notes=inputs.notes,
            disable_comparison=inputs.disable_comparison,
            disable_interactive=inputs.disable_interactive,
            package_information=inputs.package_information,
            mcmc_samples=inputs.mcmc_samples,
            external_hdf5_links=inputs.external_hdf5_links, key_data=key_data,
            existing_plot=inputs.existing_plot, disable_expert=inputs.disable_expert,
            analytic_priors=inputs.analytic_prior_dict
        )

    def generate_webpages(self):
        """Generate all webpages within the Core module
        """
        self.webpage_object.generate_webpages()


class _GWWebpageGeneration(object):
    """Class to generate all webpages for all result files with the GW module

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default"):
        from pesummary.gw.webpage.main import _WebpageGeneration
        key_data = inputs.grab_key_data_from_result_files()
        self.webpage_object = _WebpageGeneration(
            webdir=inputs.webdir, samples=inputs.samples, labels=inputs.labels,
            publication=inputs.publication, user=inputs.user, config=inputs.config,
            same_parameters=inputs.same_parameters, base_url=inputs.baseurl,
            file_versions=inputs.file_version, hdf5=inputs.hdf5, colors=inputs.colors,
            custom_plotting=inputs.custom_plotting, gracedb=inputs.gracedb,
            pepredicates_probs=inputs.pepredicates_probs,
            approximant=inputs.approximant, key_data=key_data,
            file_kwargs=inputs.file_kwargs, existing_labels=inputs.existing_labels,
            existing_config=inputs.existing_config,
            existing_file_version=inputs.existing_file_version,
            existing_injection_data=inputs.existing_injection_data,
            existing_samples=inputs.existing_samples,
            existing_metafile=inputs.existing,
            add_to_existing=inputs.add_to_existing,
            existing_file_kwargs=inputs.existing_file_kwargs,
            existing_weights=inputs.existing_weights,
            result_files=inputs.result_files, notes=inputs.notes,
            disable_comparison=inputs.disable_comparison,
            disable_interactive=inputs.disable_interactive,
            pastro_probs=inputs.pastro_probs, gwdata=inputs.gwdata,
            publication_kwargs=inputs.publication_kwargs,
            no_ligo_skymap=inputs.no_ligo_skymap,
            psd=inputs.psd, priors=inputs.priors,
            package_information=inputs.package_information,
            mcmc_samples=inputs.mcmc_samples, existing_plot=inputs.existing_plot,
            external_hdf5_links=inputs.external_hdf5_links,
            preliminary_pages=inputs.preliminary_pages,
            disable_expert=inputs.disable_expert,
            analytic_priors=inputs.analytic_prior_dict
        )

    def generate_webpages(self):
        """Generate all webpages within the Core module
        """
        self.webpage_object.generate_webpages()


class _PublicGWWebpageGeneration(object):
    """Class to generate all webpages for all result files with the GW module

    Parameters
    ----------
    inputs: argparse.Namespace
        Namespace object containing the command line options
    colors: list, optional
        colors that you wish to use to distinguish different result files
    """
    def __init__(self, inputs, colors="default"):
        from pesummary.gw.webpage.public import _PublicWebpageGeneration
        key_data = inputs.grab_key_data_from_result_files()
        self.webpage_object = _PublicWebpageGeneration(
            webdir=inputs.webdir, samples=inputs.samples, labels=inputs.labels,
            publication=inputs.publication, user=inputs.user, config=inputs.config,
            same_parameters=inputs.same_parameters, base_url=inputs.baseurl,
            file_versions=inputs.file_version, hdf5=inputs.hdf5, colors=inputs.colors,
            custom_plotting=inputs.custom_plotting, gracedb=inputs.gracedb,
            pepredicates_probs=inputs.pepredicates_probs,
            approximant=inputs.approximant, key_data=key_data,
            file_kwargs=inputs.file_kwargs, existing_labels=inputs.existing_labels,
            existing_config=inputs.existing_config,
            existing_file_version=inputs.existing_file_version,
            existing_injection_data=inputs.existing_injection_data,
            existing_samples=inputs.existing_samples,
            existing_metafile=inputs.existing,
            add_to_existing=inputs.add_to_existing,
            existing_file_kwargs=inputs.existing_file_kwargs,
            existing_weights=inputs.existing_weights,
            result_files=inputs.result_files, notes=inputs.notes,
            disable_comparison=inputs.disable_comparison,
            disable_interactive=inputs.disable_interactive,
            pastro_probs=inputs.pastro_probs, gwdata=inputs.gwdata,
            publication_kwargs=inputs.publication_kwargs,
            no_ligo_skymap=inputs.no_ligo_skymap,
            psd=inputs.psd, priors=inputs.priors,
            package_information=inputs.package_information,
            mcmc_samples=inputs.mcmc_samples, existing_plot=inputs.existing_plot,
            external_hdf5_links=inputs.external_hdf5_links,
            preliminary_pages=inputs.preliminary_pages,
            disable_expert=inputs.disable_expert,
            analytic_priors=inputs.analytic_prior_dict
        )

    def generate_webpages(self):
        """Generate all webpages within the Core module
        """
        self.webpage_object.generate_webpages()


def main(args=None, _parser=None, _core_input_cls=None, _gw_input_cls=None):
    """Top level interface for `summarypages`
    """
    from pesummary.gw.cli.parser import parser
    from pesummary.utils import functions, history_dictionary

    if _parser is None:
        _parser = parser()
    opts, unknown = _parser.parse_known_args(args=args)
    if opts.restart_from_checkpoint:
        from pesummary.core.cli.inputs import load_current_state
        from pesummary import conf
        import os
        if opts.webdir is None:
            raise ValueError(
                "In order to restart from checkpoint please provide a webdir"
            )
        resume_file_dir = conf.checkpoint_dir(opts.webdir)
        resume_file = conf.resume_file
        state = load_current_state(os.path.join(resume_file_dir, resume_file))
        if state is not None:
            _gw = state.gw
        else:
            _gw = False
        func = functions(opts, gw=_gw)
        if _gw and _gw_input_cls is not None:
            args = _gw_input_cls(opts, checkpoint=state)
        elif _core_input_cls is not None:
            args = _core_input_cls(opts, checkpoint=state)
        else:
            args = func["input"](opts, checkpoint=state)
    else:
        func = functions(opts)
        if opts.gw and _gw_input_cls is not None:
            args = _gw_input_cls(opts)
        elif _core_input_cls is not None:
            args = _core_input_cls(opts)
        else:
            args = func["input"](opts)
    from .summaryplots import PlotGeneration

    plotting_object = PlotGeneration(args, gw=args.gw)
    WebpageGeneration(args, gw=args.gw)
    _history = history_dictionary(
        program=_parser.prog, creator=args.user,
        command_line=_parser.command_line
    )
    func["MetaFile"](args, history=_history)
    if gw_results_file(opts):
        kwargs = dict(ligo_skymap_PID=plotting_object.ligo_skymap_PID)
    else:
        kwargs = {}
    func["FinishingTouches"](args, **kwargs)


if __name__ == "__main__":
    main()
