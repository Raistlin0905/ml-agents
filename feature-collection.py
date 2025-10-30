from abc import ABC, abstractmethod
import sys

class Model_Feature(ABC):
    # ONLY windows, mac and linux are supported
    SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION = ['win32', 'darwin', 'linux'] #refer to sys.platform docs 
    @abstractmethod
    def collectFromWindows(self, csv_file_name):
        pass
    @abstractmethod
    def collectFromMac(self, csv_file_name):
        pass
    @abstractmethod
    def collectFromLinux(self, csv_file_name):
        pass
    def collectFeatureAndPutIn(self, csv_file_name):
        if sys.platform == self.SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION[0]:
            self.collectFromWindows(csv_file_name)
        elif sys.platform == self.SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION[1]:
            self.collectFromMac(csv_file_name)
        elif sys.platform == self.SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION[2]:
            self.collectFromLinux(csv_file_name)
        else:
            raise NotImplementedError("Data collection for feature is not supported for the following platform: " + sys.platform)
