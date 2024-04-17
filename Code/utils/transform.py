from monai.config import KeysCollection
from monai.transforms import MapTransform
import copy
import torch

# Trasformazione personalizzata
class preprocessing(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        
        d = copy.deepcopy(data)
        d['images'] = torch.cat([data["immA"], data["immB"]], dim=0) # Ritorna immagine concatenata e GT
        del d['immA']
        del d['immB']
        #print("KEYS:", d.keys())
        return d 