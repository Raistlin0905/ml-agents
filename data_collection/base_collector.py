from abc import ABC, abstractmethod
import sys


class DataObject(ABC):
    pass


# watch that all the formats that come out of
# the functions (for the respective OSs) are the same
class DataCollector(ABC):
    # ONLY windows, mac and linux are supported
    SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION = [
        "win32",
        "darwin",
        "linux",
    ]  # refer to sys.platform docs

    @abstractmethod
    def collect_from_windows(self) -> dict:
        pass

    @abstractmethod
    def collect_from_mac(self) -> dict:
        pass

    @abstractmethod
    def collect_from_linux(self) -> dict:
        pass

    def collect_data(self) -> dict:
        handlers = {
            "win32": self.collect_from_windows,
            "darwin": self.collect_from_mac,
            "linux": self.collect_from_linux,
        }

        if sys.platform in handlers:
            return handlers[sys.platform]()
        else:
            raise NotImplementedError(
                "Data collection for feature is not supported for the following platform: "
                + sys.platform
            )
