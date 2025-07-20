import os
import os.path as osp
import PIL
import cv2
import numpy as np
import json 
import h5py
import torch
from PIL import Image

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, is_good_type, transpose_to_landscape
from dust3r.datasets.utils import cropping
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.utils.image import imread_cv2

import pdb

def cam_to_opencv(cam_T: np.ndarray):
    y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cam_T = y_z_swap.dot(cam_T)
    # revert x-axis
    cam_T[:3, 0] *= -1

    # opengl camera to opencv camera
    R = cam_T[:3, :3]
    T = cam_T[:3, 3]
    R[:, 1] *= -1
    R[:, 2] *= -1

    cam_T_world = np.eye(4)
    cam_T_world[:3, :3] = R
    cam_T_world[:3, 3] = T

    
    return cam_T_world


class ModFront3DV3(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, json_root=None, input_n=2, target_n=1, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        # assert self.split == 'train'
        # self.loaded_data = self._load_data()
        self.dataset_label = '3DFront'


        # print(input_n, target_n, self.num_views)
        super().__init__(*args, **kwargs)

        self.input_n = input_n
        self.target_n = target_n

        self.num_views = input_n + target_n

        if json_root is None:
            scene_json = osp.join(self.ROOT, f'selected_seqs_{self.split}.json')
        else:
            scene_json = osp.join(json_root, f'selected_seqs_{self.split}.json')

        with open(scene_json, 'r') as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0 and not k.endswith("_modified")}
            self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                           for k2, v2 in v.items()}
                

        self.scene_list = list(self.scenes.keys())
        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.scenes)
        

    def _get_views(self, idx, resolution, rng):

        # image_idx1, image_idx2 = self.pairs[idx]
        obj, instance = self.scene_list[idx]

        image_pool = self.scenes[(obj, instance)]

        # random choice self.input_n images
        imgs_idxs = np.random.choice(image_pool, self.num_views, replace=False)

        intrinsics = np.load(osp.join(self.ROOT, "cam_K.npy")).astype(np.float32)

        views = []
        for view_idx in imgs_idxs:
            scene_dir = osp.join(self.ROOT, obj)

            local_idx = len(views)
            view_label = f'input_{local_idx}' if local_idx < self.input_n else f'target_{local_idx-self.input_n}'

            # cutomized loading for rgb and depth
            basename = str(view_idx) + ".hdf5"
            
            # print(basename)
            h5_path = osp.join(scene_dir, instance, basename)
            with h5py.File(h5_path) as f:
                rgb_image = Image.fromarray(np.array(f["colors"])[:,::-1])
            mod_h5_path = osp.join(scene_dir + "_modified", instance, basename)
            with h5py.File(mod_h5_path) as f:
                depthmap = np.array(f["depth"])[:, ::-1]
                depthmap[depthmap == 10000000000.0] = 0
                camera_pose = np.array(f["cam_Ts"]).astype(np.float32)
                x, y, z = camera_pose[:3, 3]
                camera_pose = cam_to_opencv(camera_pose)
                camera_pose[:3, 3] = [x, z, y]
                mod_rgb_image = Image.fromarray(np.array(f["colors"])[:, ::-1])

            """ TODO: Add this back, but also for mod_rgb_image
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            """

            view_data = dict(
                img=rgb_image,
                mod_img=mod_rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=obj + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
                view_label=view_label,
            )
            views.append(view_data)

        assert len(views) == self.num_views, f"Expected {self.num_views} views, but got {len(views)}, image_pool:{image_pool}, imgs_idxs: {imgs_idxs}"
        return views
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])
            view["mod_img"] = self.transform(view["mod_img"])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    from dust3r.viz import SceneVizRerun

    dataset = ModFront3DV3(split='train', ROOT="../../renderings", resolution=224, aug_crop=0, json_root=None, input_n=2, target_n=1)

    import rerun as rr
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        viz = SceneVizRerun()
        rr.connect_grpc(flush_timeout_sec=15)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
        poses = [views[view_idx]['camera_pose'] for view_idx in range(2)]
        cam_size = 0.3
        for view_idx in range(len(views)):
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            mod_colors = rgb(views[view_idx]['mod_img'])

            viz.add_pointcloud(pts3d, f'img{idx}_{view_idx}', mod_colors, valid_mask)

            print("views[view_idx]", views[view_idx]['camera_intrinsics'][0, 0])
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size,
                           imsize=valid_mask.shape[:2])

            rr.log(f'world/cams/img{view_idx}/mask', rr.SegmentationImage(valid_mask.astype(np.uint8)))
        # break

        print(views[0].keys())
        global_pts3d_xyz = views[0]['global_pts3d_xyz']
        global_pts3d_rgb = views[0]['global_pts3d_rgb']
        global_pts3d_mask = views[0]['global_pts3d_mask']
 
        # pdb.set_trace()
        print('mask', np.mean(global_pts3d_mask))

        print(global_pts3d_xyz.shape)
        global_pts3d_rgb = global_pts3d_rgb
        print('pts3d_color', global_pts3d_rgb.max(), global_pts3d_rgb.min())
        print(global_pts3d_mask.shape)
        viz.add_pointcloud(global_pts3d_xyz, f'global_pts3d_xyz_masked', global_pts3d_rgb, global_pts3d_mask)

        global_pts3d_mask = np.ones_like(global_pts3d_mask, dtype=np.float32)
        viz.add_pointcloud(global_pts3d_xyz, f'global_pts3d_xyz', global_pts3d_rgb, global_pts3d_mask)

        viz.show()
        
        pdb.set_trace()