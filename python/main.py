import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import time
from det_face_landmarks import retinaface
from face_reconstruction import face_model
from io_ import visualize


if __name__=='__main__':
    # args = {"iscrop": True, "detector": "retinaface", "ldm68": True, "ldm106": True, "ldm106_2d": True, "ldm134": True, "seg": True, "seg_visible": True, "useTex": True, "extractTex": True, "backbone": "resnet50"}  #resnet50 mbnetv3 others own
    args = {"iscrop": True, "detector": "retinaface", "ldm68": True, "ldm106": False, "ldm106_2d": False, "ldm134": False, "seg": False, "seg_visible": False, "useTex": False, "extractTex": False, "backbone": "resnet50", "backbone_recon": "mbnetv3", "onnx_resource":"own"}
    imgpath = "testimgs/3.jpg"
    
    facebox_detector = retinaface()
    recon_model = face_model(args)

    srcimg = cv2.imread(imgpath)
    
    a = time.time()
    trans_params, im = facebox_detector.detect(srcimg)
    results = recon_model.forward(im)
    b = time.time()

    print("[DEBUG]forward : {}".format(results))
    my_visualize = visualize(results, args)
    
    img_name = os.path.splitext(os.path.basename(imgpath))[0]
    save_path = os.path.join(os.getcwd(), 'results_' + args["backbone_recon"] + "_" + args["onnx_resource"] + "_seg-{}".format(args["seg"]), img_name)
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(imgpath.split(".")[0] + "_det" + ".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    my_visualize.visualize_and_output(trans_params, srcimg, save_path, img_name)
    print("[INFO]save at : {}".format(save_path))
    print(f'#### one image Total cost time: {(b-a):.3f}s')
    
