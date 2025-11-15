from disk_collector import DiskCollector
from base_collector import FeatureCollector


class DiskAdapter(FeatureCollector):

    def collect_from_windows(self) -> dict:
        return DiskCollector().to_dict()

    def collect_from_mac(self) -> dict:
        return DiskCollector().to_dict()

    def collect_from_linux(self) -> dict:
        return DiskCollector().to_dict()
