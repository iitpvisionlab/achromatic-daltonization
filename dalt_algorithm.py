# import sys
# sys.path.append('dalt_methods/rgb_opt')
import os
import random
import glob
import argparse

import numpy as np
from PIL import Image
import torch

from utils import collate_fn, Dataset, simulation_func
from process import run_ma, run_o_ma

def parse_args():
    parser = argparse.ArgumentParser("Compute Dalt algorithm using watermark approach on set of images")
    parser.add_argument("-i", "--input", required=True, 
                        help="path to images to process")
    parser.add_argument("-o", "--output", required=True, 
                        help="path to output directory")
    parser.add_argument("-b", "--batch", required=False, 
                         default=11, type=int)
    parser.add_argument("-cvd", "--cvd_type", required=True,
                         help="type of dichromacy simulation",
                         choices=['PROTAN', 'DEUTAN', 'rg', 'by'])
    parser.add_argument("-m", "--method", required=True,
                         help="simple approx method",
                        choices=["approxMultiplicat"])
    parser.add_argument("-sn", "--sim_func_name", required=True,
                         choices=['Vienot', 'Farup'])
    parser.add_argument("-px", "--pixel_bias", required=False,
                        help="Bias of pixel for contrast calculation",
                        type=int, default=1)
    parser.add_argument("-sg", "--sign_guide", required=True,
                         help='variant of sign guide l - luminance, b - blindless',
                         choices=['l', 'b', 'lb', 'lbb', 's_l'])
    parser.add_argument("-avg", "--avg_ma", required=False,
                        type=float, default=0.8,
                         help="avg value filled for simple approximation")
    parser.add_argument("-e", "--epsilon", required=False,
                         help="loss function epsilon",
                         type=float, default=0.015)
    parser.add_argument("-lr", "--learning_rate", required=False,
                         type=float, default=0.00001)
    parser.add_argument("-n", "--num_epochs", required=False,
                         type=int, default=80000)
    parser.add_argument("-on", "--opt_name", required=False,
                        default='Adam',
                        help='name of optimizer',
                        choices=['Adam'])
    parser.add_argument("-s", "--stage", required=False,
                        default='ma',
                        help='stage of algorithm',
                        choices=['ma', 'o_ma'])
    parser.add_argument("-ma", "--ma_path", required=False,
                        default=None,
                        help="pickle format 'ma' result path")
    parser.add_argument("-cuda", "--cuda_num", required=False,
                        type=int, default=0, help="what gpu to use")

    args = parser.parse_args()

    if args.sim_func_name == "Farup":
        assert args.sign_guide in ['b', 'lb', 'lbb'], \
            'no such option for sign guide using Farup simulation'
    elif args.sim_func_name == "PROTAN":
        assert args.sign_guide in ['b', 'l', 'lb', 's_l'], \
            'no such option for sign guide using PROTAN simulation'
    return args


def _main(input_path, path_to_out_files, batch_size,
          cvdtype, method, sim_func_name, px_bias, sign_guide, eps,
          LR, num_epochs, opt_name, stage, ma_path, avg_v_fill):
    global H_MAX, W_MAX
    data = sorted(glob.glob(os.path.join(input_path, f"*")))

    # get max linear sizes
    hs = []
    ws = []
    for file in data:
        img = np.array(Image.open(file))
        hs.append(img.shape[0])
        ws.append(img.shape[1])

    h_max = max(hs)
    w_max = max(ws)

    dataset = Dataset(sorted(data), h_max, w_max)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, drop_last=False, collate_fn=collate_fn
    )

    sim_func = simulation_func(sim_func=sim_func_name, cvdtype=cvdtype, device=DEVICE)

    if stage == 'ma':
        print("Starting ma..")
        out_path = os.path.join(path_to_out_files, f'{sim_func_name}.{cvdtype}.lr_{LR}.{sign_guide}_ma.eps_{eps}.avg_v_fill_{avg_v_fill}.bias_{px_bias}')
        run_ma(dataloader, DEVICE, out_path, opt_name, eps, 
               num_epochs, sign_guide, avg_v_fill, sim_func, 
               sim_func_name, px_bias, cvdtype, len(dataset), h_max, 
               w_max, method=method, thresh=0, LR=LR)
    elif stage == 'o_ma':
        print("Starting o_ma..")
        out_path = os.path.join(path_to_out_files, f'{sim_func_name}.{cvdtype}.lr_{LR}.{sign_guide}_o_ma.eps_{eps}.avg_v_fill_{avg_v_fill}.bias_{px_bias}')
        run_o_ma(dataloader, DEVICE, ma_path, out_path, opt_name, eps,
            num_epochs, sim_func, sim_func_name, px_bias, len(dataset), h_max, w_max, LR=LR)


if __name__ == "__main__":
    global DEVICE, SEED
    SEED = int(random.random()*1000)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # to reproduce almost same result over and over
    torch.backends.cudnn.deterministic = True

    # getting arguments from terminal
    args = parse_args()
    
    DEVICE = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    _main(args.input, args.output, args.batch,
          args.cvd_type, args.method, args.sim_func_name, 
          args.pixel_bias, args.sign_guide, args.epsilon, 
          args.learning_rate, args.num_epochs, args.opt_name, 
          args.stage, args.ma_path, args.avg_ma)
