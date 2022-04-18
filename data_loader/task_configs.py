# from .glue.datasets import MRPCDataset
from .mldoc import MLDocDataset
from .marc import MARCDataset
# from .ner import CONLL2003Dataset, PANXDataset
from .ner import PANXDataset
# from .argus import ARGUStanceDataset
from .pawsx import PAWSXDataset
from .NLI import XNLIDataset
from .pos import UDPOSDataset

# from sampled_data_loader.mldoc.mldoc import SampledMLDocDataset


task2dataset = {
    "udpos": UDPOSDataset,
}


task2labelsetsize = {                                  
    "udpos": -1,
}
