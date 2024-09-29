''' This file only serves the purpose of loading nerfbuster dataset for bayesrays and does not work for running nerfbuster mode. To run/test nerfbuster model please refer to nerfbuster repository '''

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfuncertainty.dataparsers.sparse_blender.sparse_blender_dataparser import SparseBlenderDataParserConfig

sparseBlenderDataparser = DataParserSpecification(config=SparseBlenderDataParserConfig())