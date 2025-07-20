import os
import os.path as osp
import cv2
import numpy as np
import json 
import h5py

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
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


class Front3DV3(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, json_root=None, input_n=2, target_n=0, **kwargs):
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
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
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
                rgb_image = np.array(f["colors"])[:,::-1]
                depthmap = np.array(f["depth"])[:, ::-1]
                depthmap[depthmap == 10000000000.0] = 0
                camera_pose = np.array(f["cam_Ts"]).astype(np.float32)
                x, y, z = camera_pose[:3, 3]
                camera_pose = cam_to_opencv(camera_pose)
                camera_pose[:3, 3] = [x, z, y]

            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            view_data = dict(
                img=rgb_image,
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


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    from dust3r.viz import SceneVizRerun

    dataset = Front3DV3(split='train', ROOT="../../renderings", resolution=224, aug_crop=0, json_root=None, input_n=2, target_n=0)

    import rerun as rr
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        viz = SceneVizRerun()
        rr.connect_grpc(flush_timeout_sec=15)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
        poses = [views[view_idx]['camera_pose'] for view_idx in range(len(views))]
        cam_size = 0.3
        for view_idx in range(len(views)):
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            print("view_color", colors.max(), colors.min())

            viz.add_pointcloud(pts3d, f'img{idx}_{view_idx}', colors, valid_mask)

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