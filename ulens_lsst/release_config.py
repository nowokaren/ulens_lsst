"""Resolve Rubin data-release configuration through the RSP abstraction.

This module deliberately contains no release-name mapping.  The installed
``data_release_config`` package is the single source of truth for release-
specific Butler and TAP configuration.
"""

from __future__ import annotations

from typing import Any


def resolve_data_release(data_release: Any = "DP2") -> Any:
    """Return a resolved RSP ``DataReleaseConfig``.

    Parameters
    ----------
    data_release : str or DataReleaseConfig, optional
        Release name accepted by the installed RSP configuration package, or
        an already-resolved configuration object.

    Returns
    -------
    DataReleaseConfig
        The resolved release configuration supplied by RSP.

    Raises
    ------
    RuntimeError
        If the RSP configuration package is unavailable.
    TypeError
        If ``data_release`` is neither a release name nor a resolved
        ``DataReleaseConfig`` instance.
    """
    try:
        from data_release_config import DataReleaseConfig, get_data_release
    except ImportError as exc:  # pragma: no cover - depends on the RSP env
        raise RuntimeError(
            "Resolving a Rubin data release requires the RSP "
            "'data_release_config' package."
        ) from exc

    if not isinstance(data_release, (str, DataReleaseConfig)):
        raise TypeError(
            "data_release must be a release name or a DataReleaseConfig "
            f"instance, not {type(data_release).__name__}."
        )

    return get_data_release(data_release)
