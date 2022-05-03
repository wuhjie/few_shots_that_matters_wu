
from .pos import UDPOSDataset

# from sampled_data_loader.mldoc.mldoc import SampledMLDocDataset


task2dataset = {
    "udpos": UDPOSDataset,
}


task2labelsetsize = {                                  
    "udpos": -1,
}
