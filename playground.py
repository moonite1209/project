from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
import torch
import torchvision
import os
import sys
import glob
import numpy as np

def gaussian_test():
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
    render_pkg = render(camera, gaussians, pipe, bg_color, opt)
    print(gaussians.get_xyz.shape)

def test():
    gt_npys = glob.glob('output/lerf_ovs/**/train/ours_None/gt_npy/*.npy',recursive=True)
    render_npys = glob.glob('output/lerf_ovs/**/train/ours_None/renders_npy/*.npy',recursive=True)
    print(gt_npys[-1], render_npys[-1])
    print(len(gt_npys), len(render_npys))
    for gt_npy in gt_npys:
        output_path=os.path.join(os.path.dirname(gt_npy),'../gt',os.path.basename(gt_npy).replace('npy','png'))
        output_path=os.path.abspath(output_path)
        gt_npy=torch.from_numpy(np.load(gt_npy)).permute(2,0,1)
        torchvision.utils.save_image((gt_npy-gt_npy.min())/(gt_npy.max()-gt_npy.min()), output_path)
    for render_npy in render_npys:
        output_path=os.path.join(os.path.dirname(render_npy),'../renders',os.path.basename(render_npy).replace('npy','png'))
        output_path=os.path.abspath(output_path)
        render_npy=torch.from_numpy(np.load(render_npy)).permute(2,0,1)
        torchvision.utils.save_image((render_npy-render_npy.min())/(render_npy.max()-render_npy.min()), output_path)


def main()->None:
    test()

if __name__ == '__main__':
    main()
