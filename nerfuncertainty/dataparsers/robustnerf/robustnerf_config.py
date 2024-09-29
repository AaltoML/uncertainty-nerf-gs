
""""""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfuncertainty.dataparsers.robustnerf.robustnerf_dataparser import RobustnerfDataParserConfig

robustnerfDataparser = DataParserSpecification(config=RobustnerfDataParserConfig())