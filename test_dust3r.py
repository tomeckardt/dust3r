import h5py
import numpy as np
import cv2
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_image_from_hdf5(file_path: str, idx: int, img_size: int = 224) -> dict:
    with h5py.File(file_path) as f:
        img = np.array(f["colors"])
    h, w = img.shape[:2]
    min_len = min(h, w)
    img = cv2.resize(img, dsize=None, fx=min_len/img_size, fy=min_len/img_size)
    top = (h - min_len) // 2
    left = (w - min_len) // 2
    img = ImgNorm(img[top:top+img_size, left:left+img_size])
    out_dict = {
        "img": img[None],
        "true_shape": np.array([img.shape[1:]], dtype=np.int32),
        "idx": idx,
        "instance": str(idx)
    }
    return out_dict

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "checkpoints/dust3r_demo_224_lr_2/checkpoint-final.pth" # "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    """
    images = load_images([
        f'data/co3d_subset_processed/book/119_13962_28926/images/frame{str(10*i+1).zfill(6)}.jpg' for i in range(10)
    ], size=224)
    """
    images = [
        load_image_from_hdf5(f"../../renderings/0a9c667d-033d-448c-b17c-dc55e6d3c386/DiningRoom-11628/diningchair/{str(70+i)}.hdf5", i)
        for i in range(2)
    ]
    """
    images = [
        load_image_from_hdf5(f"../../renderings/0a9c667d-033d-448c-b17c-dc55e6d3c386/MasterBedroom-8583/singlebed/{str(17+i)}.hdf5", i)
        for i in range(2)
    ]
    """

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    import rerun as rr
    rr.init("Test Dust3R")
    rr.connect_grpc()
    for i, (pts, focal, pose, img) in enumerate(zip(pts3d, focals, poses, imgs), 1):
        rr.log(f"world/pts{i}", rr.Points3D(
            pts.reshape(-1, 3).detach().cpu().numpy(),
            colors=img.reshape(-1, 3)
        ))
        pose = pose.detach().cpu().numpy()
        rr.log(f'world/cams/img{i}', 
            rr.Pinhole(
                resolution=(224, 224),
                focal_length=focal.item(),
                camera_xyz=rr.ViewCoordinates.RDF, 
                image_plane_distance=0.02,
            ))

        rr.log(
                f"world/cams/img{i}",
                rr.Transform3D(translation=pose[:3,3], mat3x3=pose[:3,:3]),
            )
        
        rr.log(f'world/cams/img{i}', rr.Image(img))

    # visualize reconstruction
    # scene.show()