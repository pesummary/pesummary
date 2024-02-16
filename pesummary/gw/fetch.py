# Licensed under an MIT style license -- see LICENSE.md

from pathlib import Path

from pesummary.core.fetch import (
    download_and_read_file, _download_authenticated_file
)
from pesummary.utils.decorators import deprecation
from gwosc.api import fetch_event_json

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def fetch(url, download_kwargs={}, **kwargs):
    """Download and read files from LIGO authenticated URLs

    Parameters
    ----------
    url: str
        url you wish to download
    download_kwargs: dict, optional
        optional kwargs passed to _download_autheticated_file
    **kwargs: dict, optional
        additional kwargs passed to pesummary.io.read function
    """
    if "idp" not in download_kwargs.keys():
        download_kwargs["idp"] = "LIGO"
    return download_and_read_file(
        url, download_kwargs=download_kwargs,
        _function=_download_authenticated_file, **kwargs
    )


def _DCC_url(
    event, type="posterior", catalog=None, sampling_rate=16384, format="gwf",
    duration=32, IFO="L1", version=None,
):
    """Return the url for posterior samples stored on the DCC for a given event

    Parameters
    ----------
    event: str
        name of the event you wish to return posterior samples for
    type: str, optional
        type of data you wish to query. Default "posterior"
    catalog: str, optional
        Name of catalog that hosts the event. Default None
    sampling_rate: int, optional
        sampling rate of strain data you wish to download. Only used when
        type="strain". Default 16384
    format: str, optional
        format of strain data you wish to download. Only used when
        type="strain". Default "gwf"
    duration: int, optional
        duration of strain data you wish to download. Only used when
        type="strain". Default 32
    IFO: str, optional
        detector strain data you wish to download. Only used when type="strain".
        Default 'L1'
    version: str, optional
        version of the file to download. Default None
    """
    if type not in ["posterior", "strain"]:
        raise ValueError(
            "Unknown data type: '{}'. Must be either 'posterior' or "
            "'strain'.".format(type)
        )
    data, = fetch_event_json(
        event, catalog=catalog, version=version
    )["events"].values()
    url = None
    if type == "posterior":
        for key, item in data["parameters"].items():
            if "_pe_" in key:
                url = item["data_url"]
                break
    elif type == "strain":
        strain = data["strain"]
        for _strain in strain:
            cond = (
                _strain["sampling_rate"] == sampling_rate
                and _strain["format"] == format
                and _strain["duration"] == duration
                and _strain["detector"] == IFO
            )
            if cond:
                url = _strain["url"]
    if url is None:
        raise RuntimeError("Failed to find data URL for {}".format(event))
    return url


@deprecation(
    "The 'fetch_open_data' function has changed its name to "
    "'fetch_open_samples' and 'fetch_open_data' may not be supported in future "
    "releases. Please update"
)
def fetch_open_data(event, **kwargs):
    """Download and read publically available gravitational wave posterior
    samples

    Parameters
    ----------
    event: str
        name of the gravitational wave event you wish to download data for
    """
    return fetch_open_samples(event, **kwargs)


def _fetch_open_data(
    event, type="posterior", catalog=None, version=None, sampling_rate=16384,
    format="gwf", duration=32, IFO="L1", **kwargs
):
    """Download and read publcally available gravitational wave data

    Parameters
    ----------
    event: str
        name of the gravitational wave event you wish to download data for
    type: str, optional
        type of data you wish to download. Default "posterior"
    catalog: str, optional
        Name of catalog that hosts the event. Default None
    version: str, optional
        Version of the file to download. Default None
    sampling_rate: int, optional
        sampling rate of strain data you wish to download. Only used when
        type="strain". Default 16384
    format: str, optional
        format of strain data you wish to download. Only used when
        type="strain". Default "gwf"
    duration: int, optional
        duration of strain data you wish to download. Only used when
        type="strain". Default 32
    IFO: str, optional
        detector strain data you wish to download. Only used when type="strain".
        Default 'L1'
    """
    try:
        url = _DCC_url(
            event, type=type, catalog=catalog, sampling_rate=sampling_rate,
            format=format, duration=duration, IFO=IFO, version=version
        )
    except RuntimeError:
        raise ValueError(
            "Unknown URL for {}. If the URL is known, please run "
            "download_and_read_file(URL)".format(event)
        )
    if type == "strain":
        kwargs.update({"IFO": IFO})
    return download_and_read_file(url, **kwargs)


def fetch_open_samples(event, **kwargs):
    """Download and read publically available gravitational wave posterior
    samples

    Parameters
    ----------
    event: str
        name of the gravitational wave event you wish to download data for
    **kwargs: dict, optional
        all additional kwargs passed to _fetch_open_data
    """
    # fetch posterior data
    out = _fetch_open_data(event, type="posterior", **kwargs)

    # if asked to read the data, or unpack a tarball, just return it now
    if (
        kwargs.get("read_file", True)
        or kwargs.get("unpack", False)
    ):
        return out

    # otherwise, if Zenodo returned a file without a suffix, we need to add one
    # see https://git.ligo.org/gwosc/client/-/issues/95
    out = Path(out)
    if not out.suffix:
        out = out.rename(out.with_suffix(".h5"))
    return str(out)


def fetch_open_strain(event, format="gwf", **kwargs):
    """Download and read publically available gravitational wave strain data

    Parameters
    ----------
    event: str
        name of the gravitational wave event you wish to download data for
    format: str, optional
        format of strain data you wish to download. Default "gwf"
    **kwargs: dict, optional
        all additional kwargs passed to _fetch_open_data
    """
    _kwargs = kwargs.copy()
    _kwargs["format"] = "gwf"
    return _fetch_open_data(event, type="strain", **_kwargs)
