# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    dataset_name = 'lerf_ovs/waldo_kitchen'
    FEATURE_DIM = 32
    MODEL_PATH = f'output/{dataset_name}' # f'./output/ablation/temp/{dataset_name}_3d_3' # 30000
    ae_ckpt_path = f'data/{dataset_name}/autoencoder/best_ckpt.pth' # f'./ckpt/{dataset_name}/best_ckpt.pth'
    save_path = './edit_output'
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]

    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000

    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {
            'scene': gaussian_model,
        }

        self.seg_score = None

        self.color_mapper = None

        self.device = torch.device('cuda')

        self.load_clip()
        self.load_codec()
        self.load_model()

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.roll_back = False
        self.reload_flag = False        # reload the whole scene / point cloud

        self.render_mode = 'rgb'

        self.save_flag = False
    def __del__(self):
        dpg.destroy_context()

    def load_clip(self):
        self.clip = OpenCLIPNetwork(self.device)

    def load_codec(self):
        self.codec = Autoencoder(self.opt.encoder_dims, self.opt.decoder_dims).to(self.device)
        self.codec.load_state_dict(torch.load(self.opt.ae_ckpt_path, map_location=self.device))
        self.codec.eval()

    def load_model(self):
        self.loaded = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH, mode = 'ours')
        self.engine['scene'].get_language_feature_3d[self.engine['scene'].get_language_feature_3d.isnan()] = 0 #TODO
        self.raw_language_feature = self.codec.decode(self.engine['scene'].get_language_feature_3d)
        self.do_pca()   # calculate self.color_mapper
        self.loaded = True
        print("loading model file done.")

    def query(self, sender, app_data, user_data):
        print('querying')
        query = dpg.get_value("_query")
        threshold = dpg.get_value("_threshold")
        gaussians = self.engine["scene"]

        self.clip.set_positives([query])
        relevancy = self.clip.get_relevancy_pc(self.raw_language_feature)[0]
        mask = relevancy > threshold
        gaussians.remove_points(~mask)
        self.raw_language_feature = self.raw_language_feature[mask]
        print(f'query {mask.sum()} points')

    def remove(self, sender, app_data, user_data):
        print('removing')
        query = dpg.get_value("_query")
        threshold = dpg.get_value("_threshold")
        gaussians = self.engine["scene"]

        self.clip.set_positives([query])
        relevancy = self.clip.get_relevancy_pc(self.raw_language_feature)[0]
        mask = relevancy > threshold
        gaussians.remove_points(mask)
        self.raw_language_feature = self.raw_language_feature[~mask]
        print(f'remove {mask.sum()} points')

    def reload(self, sender, app_data, user_data):
        self.load_model()

    def save(self, sender, app_data, user_data):
        self.engine['scene'].save_ply(os.path.join(self.opt.save_path, 'point_cloud.ply'))

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            # dpg.add_button(label="render_option", tag="_button_depth",
                            # callback=callback_depth)
            dpg.add_text("\nRender option: ", tag="render")
            dpg.add_radio_button(('RGB', 'Semantic', 'Overlay'), callback=lambda sender, app_data: setattr(self,'render_mode', app_data.lower()), default_value='RGB')
            
            dpg.add_text("\nEdit option: ", tag="edit")
            dpg.add_input_text(label="query", tag="_query")
            dpg.add_slider_float(label="threshold", default_value=0.6,
                                 min_value=0.0, max_value=1.0, tag="_threshold")
            
            dpg.add_text("\n")
            dpg.add_button(label="query", callback=self.query)
            dpg.add_button(label="remove", callback=self.remove)
            dpg.add_button(label="reload", callback=self.reload)
            dpg.add_button(label="save", callback=self.save)

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        
        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            self.mouse_pos = pos


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:setattr(self, 'moving', not self.moving))
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:setattr(self, 'moving', not self.moving))
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:setattr(self, 'moving_middle', not self.moving_middle))
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:setattr(self, 'moving_middle', not self.moving_middle))
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.loaded:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            segment=None,
            semantic=None,
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam

    def pca(self, X, n_components=3):
        pca=PCA(n_components=n_components)
        pca.fit(X.cpu())
        return pca
    

    def do_pca(self):
        sems = self.engine['scene'].get_language_feature_3d.clone()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.color_mapper = self.pca(sem_chosen, n_components=3)
        print("color mapper mat initialized !")


    @torch.no_grad()
    def fetch_data(self, view_camera):
        
        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, mode='ours')
        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0).cpu()

        rgb_score = img.clone()

        # --- semantic image --- #
        sems = scene_outputs['language_feature_3d'].permute(1, 2, 0).cpu()
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = sems #TODO torch.from_numpy(self.color_mapper.transform(sems.reshape(-1, sems.shape[-1]))).reshape(sems.shape[0], sems.shape[1], -1)
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        self.render_buffer = None
        render_num = 0
        if self.render_mode == 'rgb':
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode == 'semantic':
            self.render_buffer = sem_transed_rgb.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode == 'overlay':
            self.render_buffer = 0.6*sem_transed_rgb.cpu().numpy().reshape(-1) + 0.4*rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        self.render_buffer /= render_num

        dpg.set_value("_texture", self.render_buffer)


if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    # parser.add_argument('-m', '--model_path', type=str, default="./output/lerf_ovs/figurines_3d_3")
    parser.add_argument('-f', '--feature_iteration', type=int, default=10000)
    parser.add_argument('-s', '--scene_iteration', type=int, default=30000)

    args = parser.parse_args()

    opt = CONFIG()

    # opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.SCALE_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    opt.FEATURE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    gui = GaussianSplattingGUI(opt, gs_model)

    gui.render()