''' This file only serves the purpose of loading nerfbuster dataset for bayesrays and does not work for running nerfbuster mode. To run/test nerfbuster model please refer to nerfbuster repository '''

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfuncertainty.dataparsers.hh_blender.hh_blender_dataparser import HHBlenderDataParserConfig

hhBlenderDataparser = DataParserSpecification(config=HHBlenderDataParserConfig())