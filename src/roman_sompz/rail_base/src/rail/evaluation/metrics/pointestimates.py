import numpy as np

from .base import MetricEvaluator


class PointStatsEz(MetricEvaluator):
    """Copied from PZDC1paper repo. Adapted to remove the cut based on
    magnitude."""

    def __init__(self, pzvec: np.ndarray, szvec: np.ndarray) -> None:
        """An object that takes in the vectors of the point photo-z
        the spec-z, and the i-band magnitudes for calculating the
        point statistics

        Calculates:
        ez = (pz - sz) / (1 + sz), the quantity will be useful for calculating
        statistics


        Parameters
        ----------
        pzvec : ndarray
            Array of the point photo-z values
        szvec : ndarray
            array of the spec-z values
        """
        super().__init__(None)
        self.pzs = pzvec
        self.szs = szvec
        ez = (pzvec - szvec) / (1.0 + szvec)
        self.ez = ez

    def evaluate(self) -> float | np.ndarray:
        """Return the ez values"""
        return self.ez


class PointSigmaIQR(PointStatsEz):
    """Calculate sigmaIQR"""

    def evaluate(self) -> float:
        """Calculate the width of the e_z distribution
        using the Interquartile range

        Returns
        -------
        ``sigma_IQR`` float. Width of ez distribution for full sample
        """
        x75, x25 = np.percentile(self.ez, [75.0, 25.0])
        iqr = x75 - x25
        sigma_iqr = iqr / 1.349
        return sigma_iqr


class PointBias(PointStatsEz):
    """calculates the bias of the ez and ez_magcut samples.

    In keeping with the Science Book, this is just the median of the ez values
    """

    def evaluate(self) -> float:
        """
        Returns
        -------
        ``bias`` ndarray. Median of the full ez sample
        """
        return np.median(self.ez)


class PointOutlierRate(PointStatsEz):
    """Calculates the catastrophic outlier rate, defined in the
    Science Book as the number of galaxies with ez larger than
    max(0.06,3sigma).  This keeps the fraction reasonable when
    sigma is very small.
    """

    def evaluate(self) -> float:
        """
        Returns
        -------
        ``frac`` float. Fraction of catastrophic outliers for full sample
        """
        num = len(self.ez)
        sig_iqr = PointSigmaIQR(self.pzs, self.szs).evaluate()
        threesig = 3.0 * sig_iqr
        cutcriterion = np.maximum(0.06, threesig)
        mask = np.fabs(self.ez) > cutcriterion
        outlier = np.sum(mask)
        frac = float(outlier) / float(num)
        return frac


class PointSigmaMAD(PointStatsEz):
    """Function to calculate median absolute deviation and sigma
    based on MAD (just scaled up by 1.4826) for the full and
    magnitude trimmed samples of ez values
    """

    def evaluate(self) -> float:
        """
        Returns
        -------
        ``sigma_mad`` float. Sigma median absolute deviation for full sample.
        """
        tmpmed = np.median(self.ez)
        tmpx = np.fabs(self.ez - tmpmed)
        mad = np.median(tmpx)
        sigma_mad = mad * 1.4826
        return sigma_mad
