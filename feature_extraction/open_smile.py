import opensmile

from pathlib import Path
from typing import Union


class OpenSmileFeatureExtractor:
    def __init__(self, feature_set=opensmile.FeatureSet.eGeMAPSv02):
        self._lld_features_setting = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        self._functional_features_setting = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def run(self, path: Path, start=None, end=None):
        functional_features = self._functional_features_setting.process_file(path, start=start, end=end)

        return functional_features
