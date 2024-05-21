from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, GaussianModel
import torch
import os
import sys

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    gaussians = GaussianModel(args.sh_degree)
    # scene = Scene(args, gaussians)
    params, iter = torch.load(os.path.join(args.model_path, 'chkpnt30000.pth'))
    gaussians.restore(params, args, 'eval')
    
    print(gaussians.get_xyz.shape)
