import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import time
import numpy as np

from face_reconstruction import face_model
from utils import align_img, align_face_affine_along_roll
from io_ import visualize
from utils import rotation_matrix_to_rpy, visualize_rpy_on_image


def get_lm3d_std():
    lm3d_std = np.array([[-0.31148657, 0.29036078, 0.13377953],
                                [0.30979887, 0.28972036, 0.13179526],
                                [0.0032535, -0.04617932, 0.55244243],
                                [-0.25216928, -0.38133916, 0.22405732],
                                [0.2484662, -0.38128236, 0.22235769]], dtype=np.float32)
    return lm3d_std     
    

def crop_based_five_landmarks(image, landmarks, lm3d_std):
    H = image.shape[0]
    landmarks = np.array(landmarks).astype(np.float32)
    landmarks[:, -1] = H - 1 - landmarks[:, -1]
    trans_params, im, lm, _ = align_img(image, landmarks, lm3d_std)
    return trans_params, im

    
    
def prepare_recon_face(roi, size_=224, landmarks=None):
    # get RGB
    dst = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) 
    
    if landmarks is None:
        # check Size  224
        if (dst.shape[0] != size_) or (dst.shape[1] != size_):
            org_shape = roi.shape
            dst = cv2.resize(dst, (size_, size_))
            print("[INFO]resized {} TO {}".format(org_shape, src.shape))
    
    # normailize
    
    # 
    trans_params = None
    if landmarks is not None:
        trans_params, dst = crop_based_five_landmarks(dst, landmarks, get_lm3d_std())
    
    return dst, trans_params

 

def init_recon():
    args = {"ldm68": True, "ldm106": False, "ldm106_2d": False, "ldm134": False, "seg": False, "seg_visible": False, "useTex": False, "extractTex": False, "backbone_recon": "mbnetv3", "onnx_resource":"own"}
    
    recon_model = face_model(args)
    return recon_model, args


if __name__=='__main__':
    #0. init model
    recon_model, args = init_recon()
    
    #1. input - prepare
    imgpath = "testimgs/3_det.jpg"
    srcimg = cv2.imread(imgpath)
    im, trans_params = prepare_recon_face(srcimg)
    
    #2. forward [ 82.60232047, 130.82607263]
    a = time.time()
    results = recon_model.forward(im)
    b = time.time()
    ldm68_results = results["ldm68"]    
    print("[DEBUG]forward all : {}\nlandmarks68 : {}\ntype : {} ; shape : {}".format(results, ldm68_results, type(ldm68_results), ldm68_results.shape))

    my_visualize = visualize(results, args)    
    img_name = os.path.splitext(os.path.basename(imgpath))[0]
    save_path = os.path.join(os.getcwd(), 'results_recon' + args["backbone_recon"] + "_" + args["onnx_resource"] + "_seg-{}".format(args["seg"]), img_name)
    os.makedirs(save_path, exist_ok=True)
    
    my_visualize.visualize_and_output(trans_params, srcimg, save_path, img_name)
    print("[INFO]save at : {}".format(save_path))
    
    
    #affine_file_name = img_name + "_affine_" 
    #aligned_face, aligned_landmarks, valid_flags = align_face_affine_along_roll(srcimg, ldm68_results)
    #my_visualize.visualize_and_output(trans_params, aligned_face, save_path, affine_file_name)
    
    r, p, y = rotation_matrix_to_rpy(results["rot"])
    print("[DEBUG]trans : {}, {}, {}\nrot : {}, {}, {}\ntransform : {}, {}, {}".format(type(results["trans"]), results["trans"].shape, results["trans"],type(results["rot"]), results["rot"].shape, results["rot"], type(results["transform"]), results["transform"].shape, results["transform"]))
    
    centre_index = 30
    centre_point = ldm68_results[0][centre_index, :]
    centre_point = (int(centre_point[0]), int(centre_point[1]))
    print("[INFO]R:{}, P:{}, Y:{}".format(r, p, y))
    img_axis = visualize_rpy_on_image(srcimg, r, p, y, center=centre_point)
    cv2.imwrite("{}.jpg".format(os.path.join(save_path, img_name + "_rpy_")), img_axis)
    
    



