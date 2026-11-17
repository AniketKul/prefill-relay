"""Measured uncached-length distribution for controller updates."""
from statistics import fmean
from .analytics import count, positive


class EmpiricalWorkload:
    """Capacity planning is conditional on misses; full hits bypass prefill.

    Keep the previous allocation when ``has_misses`` is false. The fraction of
    complete hits is reported separately rather than inventing a profile at zero.
    """
    def __init__(self, lengths, low=1, high=None):
        positive(low, 'minimum measured length')
        if high is not None:
            positive(high, 'maximum measured length')
            if high < low: raise ValueError('invalid measured profile range')
        samples = tuple(lengths)
        if not samples: raise ValueError('uncached-length samples cannot be empty')
        for length in samples:
            count(length, 'uncached length')
            if length and (length < low or (high is not None and length > high)):
                raise ValueError('uncached length outside measured profile range')
        self.lengths = tuple(length for length in samples if length)
        self.sample_count = len(samples)
        self.full_hit_fraction = 1 - len(self.lengths) / len(samples)
        self.has_misses = bool(self.lengths)

    @property
    def mean(self):
        if not self.has_misses: raise ValueError('all samples are complete cache hits')
        return fmean(self.lengths)

    def split(self, threshold):
        positive(threshold, 'threshold', zero=True)
        if not self.has_misses: raise ValueError('all samples are complete cache hits')
        long = [n for n in self.lengths if n > threshold]
        short = [n for n in self.lengths if n <= threshold]
        return (len(long) / len(self.lengths), fmean(long) if long else None,
                fmean(short) if short else None)
