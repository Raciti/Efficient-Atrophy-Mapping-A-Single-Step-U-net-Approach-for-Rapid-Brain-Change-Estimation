from monai.config import KeysCollection
from monai.transforms import MapTransform
import copy
import torch


class preprocessing(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        
        d = copy.deepcopy(data)
        d['images'] = torch.cat([data["immA"], data["immB"]], dim=0)  
        d['immGT'] = ((data['immGT'] * 1000) + 2000)
        del d['immA']
        del d['immB']
        return d 
    
class preprocessing_use(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        
        d = copy.deepcopy(data)
        d['images'] = torch.cat([data["immA"], data["immB"]], dim=0)  
        del d['immA']
        del d['immB']
        return d 
