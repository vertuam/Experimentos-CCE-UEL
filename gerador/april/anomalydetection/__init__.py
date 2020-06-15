# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import inspect
import sys

from gerador.april.anomalydetection.autoencoder import *
from gerador.april.anomalydetection.basic import *
from gerador.april.anomalydetection.bezerra import *
from gerador.april.anomalydetection.binet.binet import *
from gerador.april.anomalydetection.boehmer import *
from gerador.april.anomalydetection.utils.binarizer import Binarizer
from gerador.april.anomalydetection.utils.result import AnomalyDetectionResult
from gerador.april.anomalydetection.warrender import *

# Lookup dict for AD abbreviations
AD = dict((ad.abbreviation, ad) for _, ad in inspect.getmembers(sys.modules[__name__], inspect.isclass)
          if hasattr(ad, 'abbreviation') and ad.abbreviation is not None)
