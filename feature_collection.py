from abc import ABC, abstractmethod
import sys

class Data_Object(ABC):
    pass
# watch that all the formats that come out of 
# the functions (for the respective OSs) are the same
class Model_Feature_Collector(ABC):
    # ONLY windows, mac and linux are supported
    SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION = ['win32', 'darwin', 'linux'] #refer to sys.platform docs 
    @abstractmethod
    def collectFromWindows(self) -> Data_Object:
        pass
    @abstractmethod
    def collectFromMac(self) -> Data_Object:
        pass
    @abstractmethod
    def collectFromLinux(self) -> Data_Object:
        pass
    def collectFeature(self) -> Data_Object:
        if sys.platform == self.SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION[0]:
            return self.collectFromWindows()
        elif sys.platform == self.SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION[1]:
            return self.collectFromMac()
        elif sys.platform == self.SUPPORTED_OPERATING_SYSTEMS_FOR_DATA_COLLECTION[2]:
            return self.collectFromLinux()
        else:
            raise NotImplementedError("Data collection for feature is not supported for the following platform: " + sys.platform)
