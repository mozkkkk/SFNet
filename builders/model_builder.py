from LCNet.model.LCNet import LCNet
from LWGANet.models.UNetFormer_lwganet import UNetFormer_lwganet_l1
from model.EDANet import EDANet
from model.FastICENet import FastICENet




def build_model(model_name, num_classes):
    if model_name == 'FastICENet':
        return FastICENet(classes=num_classes,output_aux=True)
    elif  model_name == 'EADNet':
        return EDANet(classes=num_classes)
    elif model_name == 'LCNet':
        return LCNet(classes=num_classes)
    elif model_name == 'LWGANet':
        return UNetFormer_lwganet_l1(num_classes=num_classes)

   