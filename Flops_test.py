import sys
import argparse
import time

import numpy as np
import torch

from builders.model_builder import build_model
from model.EDANet import EDANet
from model.FastICENet import FastICENet
from network.SFNet import SFNet

from tools.flops_counter.ptflops import get_model_complexity_info



pt_models = {

    'FastICENet': FastICENet,
    'EDANet': EDANet,

    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='deeplabv3plus_mobilenet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    with torch.cuda.device(args.device):
        if args.model == "SFNet":
            net = SFNet(3).cuda()
        else:
            net = build_model(args.model, num_classes=3).cuda()
        '''
        net = pt_models[args.model](backbone='STDCNet1446', n_classes=19).cuda()
        '''
        fpses = []
        flops, params = get_model_complexity_info(net, (3, 640, 1600),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  ost=ost)
        net.eval()
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                input_tensor = torch.randn((1, 3, 640, 1600)).cuda()
                output = net(input_tensor)
                speed_time = time.time() - start
                fpses.append(1/speed_time)
        print("fps:"+str(np.mean(fpses)))
        print('Flops: ' + flops)
        print('Params: ' + params)