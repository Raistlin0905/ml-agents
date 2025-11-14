import psutil
from base_collector import FeatureCollector


class CPUCollector(FeatureCollector):
    @staticmethod
    def get_features():

        return {
            "host_cpu_cores": psutil.cpu_count(logical=True),
        }

    def collect_from_windows(self):
        features = self.get_features()

        return features

    def collect_from_mac(self):
        features = self.get_features()

        return features

    def collect_from_linux(self):
        features = self.get_features()

        return features
