import os
import h5py
import numpy as np
import cv2
import torch
import tqdm
from dust3r.mod_training import loss_of_one_batch
from dust3r.modified_model import EditSceneModel
from dust3r.utils.device import collate_with_cat, to_cpu
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def extract_hdf5(file_path: str):
    with h5py.File(file_path) as f:
        return {
            "img": np.array(f["colors"]),
            "depth": np.array(f["depth"]),
            "cam_Ts": np.array(f["cam_Ts"]).astype(np.float32)
        }

def load_image_from_hdf5(file_path: str, idx: int, img_size: int = 224) -> dict:
    with h5py.File(file_path) as f:
        img = np.array(f["colors"])
    orig_img = img
    h, w = img.shape[:2]
    min_len = min(h, w)
    img = cv2.resize(img, dsize=None, fx=min_len/img_size, fy=min_len/img_size)
    top = (h - min_len) // 2
    left = (w - min_len) // 2
    img = ImgNorm(img[top:top+img_size, left:left+img_size])
    out_dict = {
        "img": img[None],
        "mod_img": img[None],
        "orig": orig_img,
        "true_shape": np.array([img.shape[1:]], dtype=np.int32),
        "idx": idx,
        "instance": str(idx)
    }
    return out_dict

@torch.no_grad()
def inference(pairs, mod_view, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    pairs = [(*p, mod_view) for p in pairs]
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2, img3 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2, img3 in pairs]
    shapes3 = [img3['img'].shape[-2:] for img1, img2, img3 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2) and all(shapes3[0] == s for s in shapes3) 

if __name__ == '__main__':
    device = 'cpu'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "checkpoints/dust3r_demo_224_mod_2/checkpoint-best.pth" # "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
    scene_dir = "../../renderings/0a9c667d-033d-448c-b17c-dc55e6d3c386"
    mod_scene_dir = scene_dir + "_modified"
    room_dir = "SecondBedroom-5159/king-sizebed" # "DiningRoom-11628/diningchair"
    file_names = ["0.hdf5", "1.hdf5", "2.hdf5"]
    model = EditSceneModel.from_pretrained(model_name).to(device)
    images = [
        load_image_from_hdf5(os.path.join(scene_dir, room_dir, file_names[0]), 0),
        load_image_from_hdf5(os.path.join(scene_dir, room_dir, file_names[1]), 1),
        load_image_from_hdf5(os.path.join(scene_dir, room_dir, file_names[2]), 2),
    ]
    mod_images = [
        load_image_from_hdf5(os.path.join(mod_scene_dir, room_dir, file_names[0]), 0),
        load_image_from_hdf5(os.path.join(mod_scene_dir, room_dir, file_names[1]), 1),
        load_image_from_hdf5(os.path.join(mod_scene_dir, room_dir, file_names[2]), 2),
    ]
    mod_info = [
        extract_hdf5(os.path.join(mod_scene_dir, room_dir, file_names[0])),
        extract_hdf5(os.path.join(mod_scene_dir, room_dir, file_names[1])),
        extract_hdf5(os.path.join(mod_scene_dir, room_dir, file_names[2])),
    ]

    pairs = make_pairs(images[:-1], scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, mod_images[-1], model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.im_poses
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    scene.compute_global_alignment(init="mst")

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    # poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    import rerun as rr
    rr.init("Visualize Modified Dust3R")
    rr.connect_grpc("rerun+http://127.0.0.1:19876/proxy")
    print(pred1["pts3d"].detach().cpu().numpy().reshape(-1, 3).shape)
    rr.log("world/pts1", rr.Points3D(
        pred1["pts3d"].detach().cpu().numpy().reshape(-1, 3),
        colors=mod_images[0]["orig"].reshape(-1, 3)
    ))
    rr.log("world/pts2", rr.Points3D(
        pred2["pts3d_in_other_view"].detach().cpu().numpy().reshape(-1, 3),
        colors=mod_images[1]["orig"].reshape(-1, 3)
    ))
    rr.log("mod_img", rr.Image(mod_images[-1]["orig"]))
    # rr.log("pts2", rr.Points3D(pred2["pts3d_in_other_view"].detach().cpu().numpy().reshape(-1, 3)))
    # rr.log("pts2", rr.Points3D(pred2["pts3d"].detach().cpu().numpy().reshape(-1, 3)))
    # visualize reconstruction
    # scene.show()