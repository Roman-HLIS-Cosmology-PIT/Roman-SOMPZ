# pylint: disable=invalid-name
from __future__ import annotations

import enum


class DistributionType(enum.Enum):
    """ Keep track of a particular distribution represents """
    
    ad_hoc = 0
    posterior = 1
    likelihood = 2




