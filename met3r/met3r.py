# import os
# os.environ['TORCH_HOME'] = '/BS/grl-masim-data/work/torch_models'

import torch
from torch import Tensor
from pathlib import Path
from torch.nn import Module
from jaxtyping import Float, Bool
from typing import Union, Tuple
from einops import rearrange, repeat

# Load featup
from featup.util import norm, unnorm

from yugo.python.lib.py_utils.point_cloud import PointCloud

# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.transforms import Transform3d

import sys
import os
import os.path as path
HERE_PATH = os.path.dirname(os.path.abspath(__file__))
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r'))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r/dust3r'))
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, 'mast3r')
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(MASt3R_LIB_PATH) and path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MASt3R_REPO_PATH)
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(f"mast3r and dust3r is not initialized, could not find: {MASt3R_LIB_PATH}.\n "
                    "Did you forget to run 'git submodule update --init --recursive' ?")
from dust3r.utils.geometry import xy_grid


def freeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()
    

class MEt3R(Module):

    def __init__(
        self, 
        img_size: int | None = 256, 
        use_norm: bool = True,
        feat_backbone: str = 'dino16',
        featup_weights: str | Path = 'mhamilton723/FeatUp',
        dust3r_weights: str | Path = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
        use_mast3r_dust3r: bool = True,
        use_mast3r_features: bool = False,
        use_featup: bool = True,
        use_dust3r_features: bool = False,
        **kwargs
    ) -> None:
        """Initialize MET3R

        Args:
            img_size (int, optional): Image size for rasterization. Set to None to allow for rasterization with the input resolution on the fly. Defaults to 224.
            use_norm (bool, optional): Whether to use norm layers in FeatUp. Refer to https://github.com/mhamilton723/FeatUp?tab=readme-ov-file#using-pretrained-upsamplers. Defaults to True.
            feat_backbone (str, optional): Feature backbone for FeatUp. Select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]. Defaults to "dino16".
            featup_weights (str | Path, optional): Weight path for FeatUp upsampler. Defaults to "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric".
            use_mast3r_dust3r (bool, optional): Set to True to use DUSt3R weights from MASt3R. Defaults to True.
            use_mast3r_features (bool, optional): Set to True to use MASt3R features instead of FeatUp. Defaults to False.
            upsample_features (bool, optional): Set to False to use the native features directly from backbone without featup and instead use nearest neighbor upsampling. Defaults to True.
        """
        super().__init__()
        self.img_size = img_size
        self.upsampler = torch.hub.load(featup_weights, feat_backbone, use_norm=use_norm)

        self.use_mast3r_dust3r = use_mast3r_dust3r
        self.use_mast3r_features = use_mast3r_features
        self.use_dust3r_features = use_dust3r_features
        self.use_featup = use_featup
        if use_mast3r_dust3r:
            # Load MASt3R
            from mast3r.model import AsymmetricMASt3R
            self.dust3r = AsymmetricMASt3R.from_pretrained(dust3r_weights)
        else:
            # Load DUSt3R model
            from dust3r.model import AsymmetricCroCo3DStereo
            self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(dust3r_weights)

        if self.img_size is not None:
            self.set_rasterizer(
                image_size=img_size,
                points_per_pixel=10,
                bin_size=0,
                **kwargs
            )

        self.compositor = AlphaCompositor()

        freeze(self.dust3r)
        freeze(self.upsampler)

    def set_rasterizer(
        self,
        image_size,
        points_per_pixel=10,
        bin_size=0,
        **kwargs
    ) -> None:
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
            **kwargs
        )

        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

    def _compute_canonical_point_map(
            self,
            images: Float[Tensor, 'b 2 c h w'],
            return_ptmps: bool=False) -> Float[Tensor, 'b h w 3']:
        
        # NOTE: Apply DUST3R to get point maps and confidence scores
        view1 = {'img': images[:, 0, ...], 'instance': ['']}
        view2 = {'img': images[:, 1, ...], 'instance': ['']}
        pred1, pred2 = self.dust3r(view1, view2)

        ptmps = torch.stack([pred1['pts3d'], pred2['pts3d_in_other_view']], dim=1).detach()
        conf = torch.stack([pred1['conf'], pred2['conf']], dim=1).detach()

        # NOTE: Get canonical point map using the confidences
        confs11 = conf.unsqueeze(-1) - 0.999
        canon = (confs11 * ptmps).sum(1) / confs11.sum(1)
        outputs = [canon]
        if return_ptmps:
            outputs.append(ptmps)
        return (*outputs, )

    @staticmethod
    def depth_to_pointmap(
        depth_map: torch.Tensor,
        camera_matrix: torch.Tensor) -> PointCloud:
        """
        Creates PointCloud object given the depth map and camera intrinsics.
        Optional RGB colors, camera pose and normals are supported.

        Args:
            depth_map: Depth map of shape [<height>, <width>] or [<height>, <width>, 1].
            camera_matrix: Pinhole camera intrinsics matrix of shape [4, 4].
            normal_map: Optional normal map of shape [<height>, <width>, 3].
            color_image: Optional color image of shape [height, width] or [height, width, 1] if grayscale
                and [height, width, 3] if RGB image.
            camera_pose: Optional camera extrinsics matrix of shape [4, 4].
        Rets:
            PointCloud object created from the given data.
        """
        # Validate depth map
        assert depth_map.dim() in [2, 3]

        if depth_map.dim() == 3:
            assert depth_map.shape[2] == 1
            depth_map = depth_map[:, :, 0]
        depth_map = depth_map.to(dtype=torch.float32)

        # Validate intrinsics matrix
        assert camera_matrix.shape == (4, 4)
        camera_matrix = camera_matrix.to(dtype=torch.float32)

        valid_mask = depth_map > 0  # [H, W]
        valid_mask = valid_mask.reshape(-1)  # [H*W]

        H, W = depth_map.shape
        inv_camera_matrix = torch.inverse(camera_matrix)

        # Create pixel grid
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x = x.unsqueeze(-1)  # [H, W, 1]
        y = y.unsqueeze(-1)  # [H, W, 1]
        grid = torch.cat([x, y], dim=-1)  # [H, W, 2]

        # Convert to homogenous coordinates
        grid = grid.reshape(-1, 2)  # [H*W, 2]
        grid = torch.cat([grid, torch.ones(H * W, 1), torch.ones(H * W, 1)], dim=-1)  # [H*W, 4]
        grid = grid.t()  # [4, H*W]
        grid = grid.to(dtype=torch.float32, device=depth_map.device)  # [4, H*W]

        # Project pixels into 3D world, onto the camera plane
        points_3d = torch.matmul(inv_camera_matrix, grid)  # [4, H*W]
        points_3d = points_3d.t()  # [H*W, 4]
        points_3d = points_3d.reshape(H, W, 4)  # [H, W, 4]

        # Project points from camera plane to real world points
        depth = depth_map.unsqueeze(-1)  # [H, W, 1]
        points_3d[:, :, :3] *= depth

        # Remove homogenous coordinate
        pointmap = points_3d[:, :, :3]

        return pointmap

    def _get_relative_pose(self, train_pose, ood_pose):
        """A relative transformation from train to ood camera coordinate system

        Returns:
            Rt_rel: relative rotation matrix
            tt_rel: relative translation vector
        """
        assert train_pose.shape == (4, 4)
        assert ood_pose.shape == (4, 4)
        train_pose_inv = torch.inverse(train_pose)
        rel_pose = ood_pose @ train_pose_inv
        return rel_pose

    def transform_pointmap(self, pointmap, pose):
        """Transform a pointmap using a pose matrix
        """
        assert pointmap.dim() == 4  # [B==1, H, W, 3]
        assert pointmap.shape[-1] == 3
        assert pose.shape == (4, 4)
        points = pointmap.reshape(-1, 3)  # [N, 3]
        points = torch.cat([points, torch.ones(points.shape[0], 1, device=pointmap.device)], dim=-1)  # [N, 4]
        points = points.t()  # [4, N]
        points = pose @ points  # [4, N]
        points = points.t()  # [N, 4]
        points = points[:, :3]  # [N, 3]
        return points.reshape(pointmap.shape[0], pointmap.shape[1], pointmap.shape[2], 3)  # [B==1, H, W, 3]

    def _canonical_point_map_from_depth(self, train_depth, ood_depth, K, train_pose, ood_pose):
        h, w = train_depth.shape[-2:]

        rel_pose = self._get_relative_pose(train_pose, ood_pose)

        # use the relative pose to get the point map for the ood image
        ptmps = torch.zeros(train_depth.shape[0], 2, h, w, 3).to(train_depth.device)
        camera_matrix = torch.eye(4).to(train_depth.device)
        camera_matrix[:3, :3] = K
        ptmps[:, 0, ...] = self.depth_to_pointmap(ood_depth.squeeze(), camera_matrix)
        ptmps[:, 1, ...] = self.depth_to_pointmap(train_depth.squeeze(), camera_matrix)

        ptmps[:, 1, ...] = self.transform_pointmap(ptmps[:, 1, ...], rel_pose)

        canon = ptmps[:, 0, ...]

        return canon, ptmps

    def render(
        self,
        point_clouds: Pointclouds,
        **kwargs
    ) -> Tuple[
            Float[Tensor, 'b h w c'],
            Float[Tensor, 'b 2 h w n']
        ]:
        """Adoped from Pytorch3D https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/renderer.html

        Args:
            point_clouds (pytorch3d.structures.PointCloud): Point cloud object to render

        Returns:
            images (Float[Tensor, "b h w c"]): Rendered images
            zbuf (Float[Tensor, "b k h w n"]): Z-buffers for points per pixel
        """
        with torch.autocast('cuda', enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf

    def forward(
        self,
        train_rgb: Float[Tensor, 'b 3 h w'],
        ood_rgb: Float[Tensor, 'b 3 h w'],
        return_overlap_mask: bool = False,
        return_score_map: bool = False,
        return_projections: bool = False,
        train_depth: Float[Tensor, 'b h w'] | None = None,
        ood_depth: Float[Tensor, 'b h w'] | None = None,
        K: Float[Tensor, 'b 3 3'] | None = None,
        train_pose: Float[Tensor, 'b 4 4'] | None = None,
        ood_pose: Float[Tensor, 'b 4 4'] | None = None,
        return_ptmps: bool = False
    ) -> Tuple[
            float,
            Bool[Tensor, 'b h w'] | None,
            Float[Tensor, 'b h w'] | None,
            Float[Tensor, 'b 2 c h w'] | None
        ]:

        """Forward function to compute MET3R
        Args:
            images (Float[Tensor, "b 2 c h w"]): Normalized input image pairs with values ranging in [-1, 1],
            return_overlap_mask (bool, False): Return 2D map overlapping mask
            return_score_map (bool, False): Return 2D map of feature dissimlarity (Unweighted)
            return_projections (bool, False): Return projected feature maps

        Return:
            score (Float[Tensor, "b"]): MET3R score which consists of weighted mean of feature dissimlarity
            mask (bool[Tensor, "b c h w"], optional): Overlapping mask
            feat_dissim_maps (bool[Tensor, "b h w"], optional): Feature dissimilarity score map
            proj_feats (bool[Tensor, "b h w c"], optional): Projected and rendered features
        """

        images = torch.zeros(1, 2, *train_rgb.shape).to(train_rgb.device)
        images[0, 0] = ood_rgb
        images[0, 1] = train_rgb
        *_, h, w = images.shape

        if K is not None or train_pose is not None or train_depth is not None:
            assert K is not None and train_pose is not None and train_depth is not None, \
                'K, pose, and depth_map must be provided together'

        # Set rasterization settings on the fly based on input resolution
        if self.img_size is None:
            raster_settings = PointsRasterizationSettings(
                image_size=(h, w),
                radius=0.01,
                points_per_pixel=10,
                bin_size=0
                )
            self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

        if train_depth is None:
            canon, ptmps = self._compute_canonical_point_map(images, return_ptmps=True)
        else:

            canon, ptmps = self._canonical_point_map_from_depth(train_depth, ood_depth, K, train_pose, ood_pose)

        if only_ptmps:
            return ptmps

        # Define principal point
        pp = torch.tensor([w /2 , h / 2], device=canon.device)
    
        # NOTE: Estimating fx and fy for a given canonical point map
        B, H, W, THREE = canon.shape
        assert THREE == 3

        # centered pixel grid
        pixels = xy_grid(W, H, device=canon.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
        canon = canon.flatten(1, 2)  # (B, HW, 3)

        if K is None:
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = canon.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.stack((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-2)[0]

            # Normalized focal length
            focal[..., 0] = 1 + focal[..., 0] / w
            focal[..., 1] = 1 + focal[..., 1] / h
            focal = repeat(focal, 'b c -> (b k) c', k=2)

        else:
            focal = torch.tensor([
                [K[0, 0], K[1, 1]],
                [K[0, 0], K[1, 1]]
            ]).to(dtype=torch.float32, device=canon.device)
            focal[..., 0] = 1 + focal[..., 0] / w
            focal[..., 1] = 1 + focal[..., 1] / h

        # NOTE: Getting high-resolution features from FeatUp
        b, k, *_ = images.shape
        images = rearrange(images, 'b k c h w -> (b k) c h w')
        images = (images + 1) / 2

        if self.use_featup:
            # Some feature backbone may result in a different resolution feature maps than the inputs
            hr_feat = self.upsampler(norm(images))
            hr_feat = torch.nn.functional.interpolate(hr_feat, (images.shape[-2:]), mode='bilinear')
            hr_feat = rearrange(hr_feat, '... c h w -> ... (h w) c')

        elif self.use_dust3r_features:
            # NOTE: Only for Ablation (Not to be used for measuring consistency)
            shape = torch.tensor(images.shape[-2:])[None].repeat(b, 1)
            images = rearrange(images, '(b k) c h w -> b k c h w', b=b, k=k)
            images = 2 * images - 1
            feat1, feat2, pos1, pos2 = self.dust3r._encode_image_pairs(images[:, 0], images[:, 1], shape, shape)
            images = (images + 1) / 2
            images = rearrange(images, 'b k c h w -> (b k) c h w')

            hr_feat = torch.stack([feat1, feat2], dim=1)
            hr_feat = rearrange(hr_feat, 'b k (h w) c -> (b k) c h w', h=14, w=14)
            hr_feat = torch.nn.functional.interpolate(hr_feat, (images.shape[-2:]), mode='bilinear')
            hr_feat = rearrange(hr_feat, '... c h w -> ... (h w) c')

        else:
            hr_feat = self.upsampler.model(norm(images))
            hr_feat = torch.nn.functional.interpolate(hr_feat, (images.shape[-2:]), mode='nearest')
            hr_feat = rearrange(hr_feat, '... c h w -> ... (h w) c')

        # NOTE: Unproject feature on the point cloud
        ptmps = rearrange(ptmps, 'b k h w c -> (b k) (h w) c', b=b, k=2)

        point_cloud = Pointclouds(points=ptmps, features=hr_feat)

        R = torch.eye(3)
        R[0, 0] *= -1
        R[1, 1] *= -1
        R = repeat(R, '... -> (b k) ...', b=b, k=2)
        T = torch.zeros(3,)
        T = repeat(T, '... -> (b k) ...', b=b, k=2)

        # Define Pytorch3D camera for projection
        cameras = PerspectiveCameras(device=ptmps.device, R=R, T=T, focal_length=focal)

        # Render via point rasterizer to get projected features
        with torch.autocast('cuda', enabled=False):
            rendering, zbuf = self.render(point_cloud, cameras=cameras, background_color=[-10000] * hr_feat.shape[-1])
        rendering = rearrange(rendering, '(b k) ... -> b k ...',  b=b, k=2)

        # Compute overlapping mask
        non_overlap_mask = (rendering == -10000)
        overlap_mask = (1 - non_overlap_mask.float()).prod(-1).prod(1)

        # Zero out regions which do not overlap
        rendering[non_overlap_mask] = 0.0

        # Mask for weighted sum
        mask = overlap_mask

        # NOTE: Uncomment for incorporating occlusion masks along with overlap mask
        # zbuf = rearrange(zbuf, "(b k) ... -> b k ...",  b=b, k=2)
        # closest_z = zbuf[..., 0]
        # diff = (closest_z[:, 0, ...] - closest_z[:, 1, ...]).abs()
        # mask = (~(diff > 0.5) * (closest_z != -1).prod(1)) * mask

        # Get feature dissimilarity score map
        feat_dissim_maps = 1 - (rendering[:, 1, ...] * rendering[:, 0, ...]).sum(-1) / (torch.linalg.norm(rendering[:, 1, ...], dim=-1) * torch.linalg.norm(rendering[:, 0, ...], dim=-1) + 1e-3)
        # Weight feature dissimilarity score map with computed mask
        feat_dissim_weighted = (feat_dissim_maps * mask).sum(-1).sum(-1) / (mask.sum(-1).sum(-1) + 1e-6)

        outputs = [feat_dissim_weighted]
        if return_overlap_mask:
            outputs.append(mask)

        if return_score_map:
            outputs.append(feat_dissim_maps)

        if return_projections:
            outputs.append(rendering)

        if return_ptmps:
            outputs.append(ptmps)

        return (*outputs, )
