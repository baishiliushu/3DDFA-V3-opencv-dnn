import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import time
import numpy as np


from face_reconstruction import face_model
from utils import align_img, align_face_affine_along_roll
from io_ import visualize
from utils import rotation_matrix_to_rpy, rpy_to_maxtirx_no_tr, visualize_rpy_on_image


def make_affine_result_visual(results, aligned_landmarks):
    results_affine = results
    aligned_landmarks = aligned_landmarks[0]
    aligned_landmarks[:, 1] = srcimg.shape[0] - 1 - aligned_landmarks[:,1]
    aligned_landmarks = aligned_landmarks[np.newaxis, ...]
    
    results_affine["ldm68"] = aligned_landmarks
    return results_affine

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


def letterbox_image(image, target_size, padding_value=127):
    """
    对图像进行letterbox padding处理（保持宽高比，补边像素值127）
    
    Args:
        image: 输入图像（BGR格式，OpenCV读取）
        target_size: 目标尺寸（正方形，如416）
        padding_value: 补边像素值（默认127）
    
    Returns:
        处理后的图像（尺寸为target_size x target_size）
    """
    # 确保输入是三通道图像
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    h, w = image.shape[:2]
    
    # 计算缩放比例（保持宽高比）
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建空白图像（目标尺寸，填充指定值）
    padded = np.full((target_size, target_size, 3), padding_value, dtype=np.uint8)
    
    # 计算粘贴位置（居中）
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # 将缩放后的图像粘贴到空白图像中心
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded
    
def prepare_recon_face(roi, size_=224, landmarks=None):
    # get RGB
    roi = letterbox_image(roi, size_)
    dst = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) 
    
    if landmarks is None:
        # check Size  224
        if (dst.shape[0] != size_) or (dst.shape[1] != size_):
            org_shape = roi.shape
            dst = cv2.resize(dst, (size_, size_))
            print("[INFO]resized {} TO {}".format(org_shape, dst.shape))
    
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

import cv2
import numpy as np

def draw_text_on_image(img, text, font_scale_factor=0.015, min_font_scale=0.5, max_font_scale=0.5):
    """
    在图片左上角添加文字，自动适配图片尺寸
    
    参数:
    img: OpenCV格式的图片矩阵 (BGR)
    text: 要显示的字符串
    font_scale_factor: 字体比例因子 (默认0.015，可调)
    min_font_scale: 最小字体大小 (默认0.5)
    max_font_scale: 最大字体大小 (默认2.0)
    
    返回:
    带文字的图片 (BGR格式，与输入格式一致)
    """
    # 深拷贝原图
    
    h, w = img.shape[:2]
    
    # 计算动态字体大小 (基于图片最小边)
    font_scale = min(w, h) * font_scale_factor
    font_scale = max(min_font_scale, min(font_scale, max_font_scale))
    
    # 确定文字颜色 (单通道/三通道处理)
    if len(img.shape) == 2:  # 灰度图
        color = 255
    else:  # BGR图
        color = (255, 255, 255)  # 白色
    
    # 获取文字尺寸 (用于精准定位)
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    
    # 计算文字左上角位置 (确保不超出边界)
    x = 15  # 水平偏移
    y = 30 + text_height  # 垂直偏移 (文字顶部在y=30处)
    
    # 保护边界 (确保文字不超出图片)
    y = min(y, h - 10)
    x = min(x, w - text_width - 5)
    
    # 添加文字 (使用抗锯齿)
    cv2.putText(
        img, text, (x, y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        color, 
        1, 
        cv2.LINE_AA
    )
    
    return img


if __name__=='__main__':
    #0. init model
    recon_model, args = init_recon()
    
    #1. input - prepare
    imgpath = "testimgs/rpy/yaw+45-small.jpg"
    #TODO: argv    
    print("[INFO]run on : {}".format(imgpath))
    
    srcimg = cv2.imread(imgpath)
    
    im, trans_params = prepare_recon_face(srcimg)
    im_rpy_display = im.copy() 
    im_rpy_display = cv2.cvtColor(im_rpy_display, cv2.COLOR_RGB2BGR)
    
    #2. forward [ 82.60232047, 130.82607263]
    a = time.time()
    results = recon_model.forward(im)
    b = time.time()
    ldm68_results = results["ldm68"]    
    print("[DEBUG]forward all : {}\nlandmarks68 : {}\ntype : {} ; shape : {}".format(results, ldm68_results, type(ldm68_results), ldm68_results.shape))

    my_visualize = visualize(results, args)    
    img_name = os.path.splitext(os.path.basename(imgpath))[0]
    save_path = os.path.join(os.getcwd(), '2-padding-results_recon-' + args["backbone_recon"] + "_" + args["onnx_resource"] + "_seg-{}".format(args["seg"]), img_name)
    os.makedirs(save_path, exist_ok=True)
    
    #my_visualize.visualize_and_output(trans_params, srcimg, save_path, img_name)
    print("[INFO]save at : {}".format(save_path))
    
    
    affine_file_name = img_name + "_wrap" 
    affine_face, aligned_landmarks, valid_flags = align_face_affine_along_roll(srcimg, ldm68_results)
    results_affine = make_affine_result_visual(results, aligned_landmarks)
    affine_visualize = visualize(results_affine, args)
    affine_visualize.visualize_and_output(None, affine_face, save_path, affine_file_name)
    
    r, p, y = rotation_matrix_to_rpy(results["rot"]) # for debug rotation matrix
    
    print("[DEBUG]trans : {}, {}, {},\n \ntransform : {}, {}, {}\nrot : {}, {} , {}".format(type(results["trans"]), results["trans"].shape, results["trans"], results["rot"], type(results["transform"]), results["transform"].shape, results["transform"],type(results["rot"]), results["rot"].shape, results["rot"]))
    
    rpy_based_model = [results["xyz_rpy"][0], results["xyz_rpy"][1], results["xyz_rpy"][2]] 
    print("[DEBUG]R:{}, P:{}, Y:{} v.s. rpy_based_model-> R:{}, P:{}, Y:{} ({})".format(r, p, y, rpy_based_model[0], rpy_based_model[1], rpy_based_model[2], type(rpy_based_model[0])))
    
    centre_index = 30
    centre_point = ldm68_results[0][centre_index, :]
    centre_point = (int(centre_point[0]), int(centre_point[1]))
    #rot_matrix_visual = rpy_to_maxtirx_no_tr(rpy_based_model[0], rpy_based_model[1], rpy_based_model[2])
    rot_matrix_visual = results["rot"]

    img_axis = visualize_rpy_on_image(srcimg, rot_matrix_visual)
    cv2.imwrite("{}.jpg".format(os.path.join(save_path, img_name + "_rpy_")), img_axis)
    
    
    degree_scale = 1.0 #180.0 / 3.14159265358979323846

    pyr_pai = [results["xyz_pyr"][0]*degree_scale, results["xyz_pyr"][1]*degree_scale, results["xyz_pyr"][2]*degree_scale]
    # results["xyz_rpy"][0]    
    rpy_text = "R:{},P:{},Y:{}".format(f"{float(pyr_pai[2]):.1f}", f"{float(pyr_pai[0]):.1f}", f"{float(pyr_pai[1]):.1f}")
    im_rpy_display = draw_text_on_image(im_rpy_display, rpy_text)
    cv2.imwrite("{}.jpg".format(os.path.join(save_path, img_name + "_rpy-text_")), im_rpy_display)
    
    



