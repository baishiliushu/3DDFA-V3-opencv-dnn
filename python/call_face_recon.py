import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import time
import numpy as np

from face_reconstruction import face_model
from utils import align_img
from io_ import visualize


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
    return recon_model


if __name__=='__main__':
    #0. init model
    recon_model = init_recon()
    
    #1. input - prepare
    imgpath = "testimgs/3_det.jpg"
    srcimg = cv2.imread(imgpath)
    im, trans_params = prepare_recon_face(srcimg)
    
    #2. forward [ 82.60232047, 130.82607263]
    a = time.time()
    results = recon_model.forward(im)
    b = time.time()
    ldm68_results = results["ldm68"]
    print("[DEBUG]forward all : {}\nlandmarks68 : {}".format(results, ldm68_results))


