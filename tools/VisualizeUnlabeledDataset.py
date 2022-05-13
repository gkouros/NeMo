import sys
sys.path.append('../code')

import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import BboxTools as bbt
from lib.MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras
import argparse


device = 'cuda:0'
render_image_size = (672, 672)
image_size = (256, 672)

parser = argparse.ArgumentParser(description='Visualize Unlabeled Dataset')
parser.add_argument('--anno_path', default='../final_pred.npz', type=str, help='')
parser.add_argument('--img_path', default='../data/PASCAL3D+_release1.1', type=str, help='')
parser.add_argument('--cate', default='car', type=str, help='')
args = parser.parse_args()

anno_path = args.anno_path
img_path = args.img_path
cate = args.cate
mesh_path = '../data/PASCAL3D+_release1.1/CAD/%s/01.off' % cate


class CustomedCrop(object):
    def __init__(self, crop_size, tar_horizontal):
        self.crop_size = crop_size
        self.tar_horizontal = tar_horizontal

    def __call__(self, im):
        size_ = im.size
        out_size = (self.tar_horizontal, int(size_[1] / size_[0] * self.tar_horizontal),)
        img = np.array(im.resize(out_size))
        crop_box = bbt.box_by_shape(self.crop_size, bbt.full(img).center, ).shift((30, 0))
        cropped_img = crop_box.apply(img)
        return cropped_img


x3d, xface = load_off(mesh_path)

faces = torch.from_numpy(xface)

# TODO: convert verts
verts = torch.from_numpy(x3d)
verts = pre_process_mesh_pascal(verts)
cameras = OpenGLPerspectiveCameras(device=device, fov=12.0)
# cameras = PerspectiveCameras(focal_length=1.0 * 3000, principal_point=((render_image_size[0]/ 2, render_image_size[1]/ 2),), image_size=(render_image_size, ), device=device)

verts_rgb = torch.ones_like(verts)[None] * torch.Tensor([1, 0.85, 0.85]).view(1, 1, 3)  # (1, V, 3)
# textures = Textures(verts_rgb=verts_rgb.to(device))
textures = Textures(verts_features=verts_rgb.to(device))
meshes = Meshes(verts=[verts], faces=[faces], textures=textures)
meshes = meshes.to(device)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=render_image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=0
)
# We can add a point light in front of the object.
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights, cameras=cameras),
)

trans = CustomedCrop(image_size, 672)
annos = np.load(anno_path, allow_pickle=True)


suffix = os.listdir(img_path)[0].split('.')[-1]

for k in list(annos.keys())[0::10]:

    img_ = Image.open(img_path + '%s.%s' % (k, suffix))
    img = trans(img_)

    print(annos[k])
    distance_pred, theta_pred, elevation_pred, azimuth_pred, t0, t1 = \
        3.46957731,  0.32689595,  0.00502517,  1.5008601 , -0.06280591, 0.02838039
        # annos[k]

    # print(distance_pred)
    # Image.fromarray(img).show()
    C = camera_position_from_spherical_angles(distance_pred, elevation_pred, azimuth_pred, degrees=False, device=device)
    R, T = campos_to_R_T(C, torch.Tensor([theta_pred]), device=device, extra_trans=torch.Tensor([[t0, t1, 0]]).to(device))
    image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
    image = image[0, ..., :3].detach().squeeze().cpu().numpy()
    print(image.min())

    image = np.array((image / image.max()) * 255).astype(np.uint8)

    crop_box = bbt.box_by_shape(image_size, (render_image_size[0] // 2, render_image_size[1] // 2), image_boundary=render_image_size)

    image = crop_box.apply(image)

    mixed_image = (image * 0.6 + img * 0.4).astype(np.uint8)
    Image.fromarray(mixed_image).save('../Visuals1/%s.jpg' % k)

    break