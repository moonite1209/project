from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
import torch
import torchvision
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    # plt.plot(gaussians.get_language_feature_3d[:,0].detach().cpu().numpy())
    # plt.plot(gaussians.get_language_feature_3d[:,1].detach().cpu().numpy())
    # plt.plot(gaussians.get_language_feature_3d[:,2].detach().cpu().numpy())
    # plt.show()
    fig = plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.scatter(gaussians.get_language_feature_3d[:,0].detach().cpu().numpy(),gaussians.get_language_feature_3d[:,1].detach().cpu().numpy(),gaussians.get_language_feature_3d[:,2].detach().cpu().numpy())
    plt.show()
    # render_pkg = render(camera, gaussians, pipe, bg_color, opt)
    print(gaussians.get_xyz.shape)

def save_image():
    gt_npys = glob.glob('output/lerf_ovs/**/train/ours_None/gt_npy/*.npy',recursive=True)
    render_npys = glob.glob('output/lerf_ovs/**/train/ours_None/renders_npy/*.npy',recursive=True)
    print(gt_npys[-1], render_npys[-1])
    print(len(gt_npys), len(render_npys))
    for gt_npy in gt_npys:
        output_path=os.path.join(os.path.dirname(gt_npy),'../gt',os.path.basename(gt_npy).replace('npy','png'))
        output_path=os.path.abspath(output_path)
        gt_npy=torch.from_numpy(np.load(gt_npy)).permute(2,0,1)
        torchvision.utils.save_image(gt_npy/2+0.5, output_path)
    for render_npy in render_npys:
        output_path=os.path.join(os.path.dirname(render_npy),'../renders',os.path.basename(render_npy).replace('npy','png'))
        output_path=os.path.abspath(output_path)
        render_npy=torch.from_numpy(np.load(render_npy)).permute(2,0,1)
        torchvision.utils.save_image(render_npy/2+0.5, output_path)

def strip():
    plys=glob.glob('output/lerf_ovs/**/point_cloud/ours_None/gt_npy/*.npy',recursive=True)

def save_ply():
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

    paths=glob.glob('output/lerf_ovs/*3d*',recursive=True)
    paths=['output/lerf_ovs/figurines_1','output/lerf_ovs/figurines_2','output/lerf_ovs/figurines_3',
           'output/lerf_ovs/ramen_1','output/lerf_ovs/ramen_2','output/lerf_ovs/ramen_3',
           'output/lerf_ovs/teatime_1','output/lerf_ovs/teatime_2','output/lerf_ovs/teatime_3',
           'output/lerf_ovs/waldo_kitchen_1','output/lerf_ovs/waldo_kitchen_2','output/lerf_ovs/waldo_kitchen_3']
    paths3d=['output/lerf_ovs/figurines_3d_1','output/lerf_ovs/figurines_3d_2','output/lerf_ovs/figurines_3d_3',
           'output/lerf_ovs/ramen_3d_1','output/lerf_ovs/ramen_3d_2','output/lerf_ovs/ramen_3d_3',
           'output/lerf_ovs/teatime_3d_1','output/lerf_ovs/teatime_3d_2','output/lerf_ovs/teatime_3d_3',
           'output/lerf_ovs/waldo_kitchen_3d_1','output/lerf_ovs/waldo_kitchen_3d_2','output/lerf_ovs/waldo_kitchen_3d_3']
    print(paths)
    for path in tqdm(paths):
        pth=os.path.join(path,'chkpnt30000.pth')
        ply=os.path.join(path,'point_cloud/iteration_30000/point_cloud.ply')
        gaussians = GaussianModel(args.sh_degree)
        params, iter = torch.load(pth)
        gaussians.restore(params, args, 'eval')
        gaussians.save_ply(ply)

def save_ply_with_similarity():
    parser = ArgumentParser(description="save relevancy script parameters")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--ae_ckpt_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default='ours')
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args(sys.argv[1:])

    pharses=('knife', 'yellow desk', 'refrigerator', 'cabinet', 'frog cup', 'plate')
    gaussians = GaussianModel(3)
    pc_path=os.path.join(args.model_path, 'point_cloud/iteration_30000/point_cloud.ply')
    gaussians.load_ply(pc_path,'ours')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = OpenCLIPNetwork(device)
    checkpoint = torch.load(args.ae_ckpt_path, map_location=device)
    codec = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    codec.load_state_dict(checkpoint)
    codec.eval()

    lf3=gaussians.get_language_feature_3d
    print(lf3.shape)
    lf=codec.decode(lf3)
    print(lf.shape)
    clip.set_positives(pharses)
    pharses_rel = clip.get_relevancy_pc(lf)
    print(len(pharses_rel))

    for index, item in enumerate(pharses):
        print(f'saving {os.path.join(os.path.basename(pc_path), f'relevancy_{item}.ply')}')
        gaussians.save_ply(os.path.join(os.path.basename(pc_path), f'relevancy_{item}.ply'), relevancy=pharses_rel[index].unsqueeze(-1).detach().cpu().numpy())

def main()->None:
    save_ply_with_similarity()

if __name__ == '__main__':
    main()
