import numpy as np
from scipy.stats import gaussian_kde
import multiprocessing


class KDEList(object):
    """Class to generate and evaluate a set of KDEs at the same points

    Parameters
    ----------
    samples: np.ndarray
        2d array of samples to generate kdes
    kde: func, optional
        kde function to use. Default scipy.stats.gaussian_kde
    kde_kwargs: dict, optional
        kwargs to pass to kde
    """
    def __init__(self, samples, kde=gaussian_kde, kde_kwargs={}, pts=None):
        self.samples = samples
        if not all(isinstance(_, (list, np.ndarray)) for _ in self.samples):
            raise ValueError("2d array of samples must be provided")
        self.kdes = np.array(
            [kde(_, **kde_kwargs) for _ in self.samples], dtype=object
        )
        self.pts = pts

    def __call__(self, pts=None, multi_process=1, idx=None):
        if pts is None and self.pts is None:
            raise ValueError("Please provide a set of points to evaluate the KDE")
        elif pts is None:
            pts = self.pts
        singular_idx = not isinstance(idx, (list, np.ndarray)) and idx is not None
        if idx is None:
            idx = np.ones(len(self.kdes), dtype=bool)
        elif not isinstance(idx, (list, np.ndarray)):
            idx = np.array([idx])
        if multi_process == 1:
            out = np.array(
                [self._evaluate_single_kde(kde, pts) for kde in self.kdes[idx]]
            )
        else:
            with multiprocessing.Pool(multi_process) as pool:
                args = np.array([[kde, pts] for kde in self.kdes[idx]], dtype=object)
                out = np.array(
                    pool.map(KDEList._wrapper_for_evaluate_single_kde, args)
                )
        if singular_idx:
            return out[0]
        return out

    def evaluate(self, **kwargs):
        if self.pts is None:
            raise ValueError(
                "No points stored. Please use the __call__ method and provide "
                "a list of points or re-initalise class"
            )
        return self.__call__(self.pts, **kwargs)

    @staticmethod
    def _wrapper_for_evaluate_single_kde(args):
        return KDEList._evaluate_single_kde(*args)

    @staticmethod
    def _evaluate_single_kde(kde, pts):
        return kde(pts)
