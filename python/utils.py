import numpy as np
import cv2
from itertools import product
from math import ceil


def priorbox(min_sizes, steps, clip, image_size):
    feature_maps = [[ceil(image_size[0] / step),
                     ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = np.array(anchors, dtype=np.float32).reshape((-1, 4))
    if clip:
        output = np.clip(output, 0, 1)
    return output

def decode(loc, priors, variances):
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ), axis=1)
    
    boxes[:, :2] -= boxes[:, 2:] * 0.5
    # boxes[:, 2:] += boxes[:, :2]
    return boxes  ###（x, y, w, h）


def decode_landm(pre, priors, variances):
    return np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ), axis=1)


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[0]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp, [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # _, k = cv2.solve(A.T@A, A.T@b, cv2.DECOMP_CHOLESKY)   ////等价的， c++的solve(A.t()*A, A.t()*b, k, DECOMP_CHOLESKY);

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    h0, w0 = img.shape[:2]
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = int(left + target_size)
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = int(up + target_size)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img = img[up:below, left:right]
    dsth, dstw = int(target_size), int(target_size)
    if img.shape[0] < dsth:
        img = cv2.copyMakeBorder(img, 0, dsth - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    if img.shape[1] < dstw:
        img = cv2.copyMakeBorder(img, 0, 0, 0, dstw - img.shape[1], cv2.BORDER_CONSTANT, value=0)

    if mask is not None:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = mask[up:below, left:right]
        if mask.shape[0] < dsth:
            mask = cv2.copyMakeBorder(mask, 0, dsth - mask.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
        if mask.shape[1] < dstw:
            mask = cv2.copyMakeBorder(mask, 0, 0, 0, dstw - mask.shape[1], cv2.BORDER_CONSTANT, value=0)

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --numpy.array  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --numpy.array  (target_size, target_size)
    
    Parameters:
        img                --numpy.array  (raw_H, raw_W, 3)
        lm                 --numpy.array  (5, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --numpy.array  (raw_H, raw_W, 3)
    """

    h0, w0  = img.shape[:2]
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p, lm3D)
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    # trans_params = np.array([w0, h0, s, t[0], t[1]])
    trans_params = np.array([w0, h0, s, t[0][0], t[1][0]])
    print("[DEBUG]type: {}, shape : {}".format(type(img_new), img_new.shape))
    return trans_params, img_new, lm_new, mask_new

def process_uv(uv_coords, uv_h = 224, uv_w = 224):
    uv_coords[:,0] = uv_coords[:,0] * (uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1] * (uv_h - 1)
    # uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def get_expend_box(x1, y1, x2, y2, enlarge_ratio, height, width):
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    cx = (x2 + x1) * 0.5
    cy = (y2 + y1) * 0.5
    sz = max(h, w) * enlarge_ratio

    x1 = cx - sz * 0.5
    y1 = cy - sz * 0.5
    trans_x1 = x1
    trans_y1 = y1
    x2 = x1 + sz
    y2 = y1 + sz

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return dy, edy, dx, edx, sz, trans_x1, trans_y1

def face_affine(src, lm_points, average_face_3D = None):
    
    return

def align_face_affine_along_roll(face_img, landmarks_68, desired_left_eye=(0.35, 0.35), 
                                    desired_face_width=224, desired_face_height=None):
    """
    使用仿射变换对齐人脸图像并同时变换关键点
    
    参数:
        face_img: 输入的人脸ROI图像 (BGR格式)
        landmarks_68: 人脸68个关键点坐标 (形状为(1, 68, 2)或(68, 2)的numpy数组)
        desired_left_eye: 期望的左眼在输出图像中的相对位置 (x,y)
        desired_face_width: 输出图像的宽度
        desired_face_height: 输出图像的高度(如果为None，则与宽度相同)
    
    返回:
        aligned_face: 对齐后的人脸图像
        aligned_landmarks: 变换后的关键点(形状与输入相同，包含合法性标记)
        valid_flags: 每个关键点是否合法的布尔数组(True表示合法)
    """
    if desired_face_height is None:
        desired_face_height = desired_face_width
    
    # 转换关键点形状为(68, 2)
    orig_shape = landmarks_68.shape
    if landmarks_68.ndim == 3 and landmarks_68.shape[0] == 1:
        landmarks = landmarks_68[0]  # 从(1, 68, 2)变为(68, 2)
    elif landmarks_68.shape == (68, 2):
        landmarks = landmarks_68
    else:
        raise ValueError(f"Landmarks shape must be (1, 68, 2) or (68, 2), got {landmarks_68.shape}")
    
    # 检查关键点是否有效(非负)
    valid_flags = np.all(landmarks >= 0, axis=1)
    
    # 获取左右眼中心坐标(只使用有效的关键点)
    left_eye_indices = slice(36, 42)
    right_eye_indices = slice(42, 48)
    
    left_eye_points = landmarks[left_eye_indices][valid_flags[left_eye_indices]]
    right_eye_points = landmarks[right_eye_indices][valid_flags[right_eye_indices]]
    
    if len(left_eye_points) < 3 or len(right_eye_points) < 3:
        raise ValueError("Not enough valid eye landmarks for alignment")
    
    left_eye_center = left_eye_points.mean(axis=0).astype("float")
    right_eye_center = right_eye_points.mean(axis=0).astype("float")
    
    # 计算两眼之间的角度
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 计算期望的两眼距离
    desired_right_eye_x = 1.0 - desired_left_eye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_face_width
    scale = desired_dist / dist
    
    # 获取两眼中心的中点
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) * 0.5,
                   (left_eye_center[1] + right_eye_center[1]) * 0.5)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    # 更新平移分量
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    
    # 应用仿射变换到图像
    (w, h) = (desired_face_width, desired_face_height)
    aligned_face = cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # 准备变换关键点
    aligned_landmarks = np.zeros_like(landmarks)
    
    # 变换每个关键点并更新合法性
    for i in range(68):
        if not valid_flags[i]:
            continue  # 保持原有的无效状态
            
        x, y = landmarks[i]
        transformed_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        transformed_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        
        # 检查变换后的点是否在图像范围内
        if (0 <= transformed_x < w) and (0 <= transformed_y < h):
            aligned_landmarks[i] = [transformed_x, transformed_y]
        else:
            valid_flags[i] = False
            aligned_landmarks[i] = [-1, -1]  # 标记为无效
    
    # 恢复原始形状(1, 68, 2)
    if orig_shape[0] == 1:
        aligned_landmarks = aligned_landmarks[np.newaxis, ...]
    
    return aligned_face, aligned_landmarks, valid_flags


def rpy_to_maxtirx_no_tr(roll, pitch, yaw):
    # 将 RPY 角度转换为弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # 计算旋转矩阵 (RPY 顺序: Z yaw -> Y pitch -> X roll)
    # 计算旋转矩阵 (RPY 顺序: X pitch -> Y yaw -> Z roll)
    #TODO: order rotation
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R = Rz @ Ry @ Rx
    print("[DEBUG]rotated based rpy-degree : {}, shape {}\n{}".format(type(R), R.shape, R))
    return R

def rotation_matrix_to_rpy(R, to_degree = True):
    print("[DEBUG]--maxtirx {} to euler-angle".format(R.shape))
    if R.shape == (1, 3, 3):
        print("[INFO]original rot-matrix in results has been TRANSPONSED")
        R = np.transpose(R, (0, 2, 1))
        R = R.reshape(3, 3)
    else:
        R = np.transpose(R)
    
    print("[DEBUG]++maxtirx {} to euler-angle".format(R.shape))
    # 计算Pitch (theta)
    #pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    # 计算Yaw (psi)
    #yaw = np.arctan2(R[1, 0], R[0, 0])
    # 计算Roll (phi)
    #roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll = np.arctan2(R[1, 0], R[0, 0])
    if to_degree:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
    return roll, pitch, yaw

def rpy_to_rotation_maxtrix(x, y, z):
    # x -> roll, y -> pitch, z -> yaw
    batch_size = 1
    ones = np.ones([batch_size, 1])
    zeros = np.zeros([batch_size, 1])
    
    rot_x = np.concatenate([
        ones, zeros, zeros,
        zeros, np.cos(x), -np.sin(x), 
        zeros, np.sin(x), np.cos(x)
    ], axis=1).reshape([batch_size, 3, 3])
    
    rot_y = np.concatenate([
        np.cos(y), zeros, np.sin(y),
        zeros, ones, zeros,
        -np.sin(y), zeros, np.cos(y)
    ], axis=1).reshape([batch_size, 3, 3])

    rot_z = np.concatenate([
        np.cos(z), -np.sin(z), zeros,
        np.sin(z), np.cos(z), zeros,
        zeros, zeros, ones
    ], axis=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    rot = np.transpose(rot, (0, 2, 1))
    return rot

def transform_matrix_transpose():
    #TODO: TRG = ORG * rot ; ORG = rot[Tr] * TRG

    return rot


def orientation_max_map_2D(unit_3D, R, cx = 112, cy = 112):
    # 应用旋转矩阵并转换为 2D 坐标
    if R.shape == (1, 3, 3):
        R = R.reshape(3, 3)
        #np.expand_dims(R, axis=1)
    rotated_axes =  unit_3D @ R #rotated_axes = R @ unit_3D.T
    print("[DEBUG]rotated_axes : {}, shape {}\n{}".format(type(rotated_axes), rotated_axes.shape, rotated_axes))
    x_end = (int(rotated_axes[0, 0].astype(int) + cx), int(rotated_axes[1, 0].astype(int) + cy)) #x_end
    y_end = (int(rotated_axes[0, 1].astype(int) + cx), int(rotated_axes[1, 1].astype(int) + cy)) #y_end
    z_end = (int(rotated_axes[0, 2].astype(int) + cx), int(rotated_axes[1, 2].astype(int) + cy)) #z_end
    
    return x_end, y_end, z_end

def orientation_jacobian_map_2D(unit_3D, R, size_H = 224, size_W = 224, trans_x = 0, trans_y = 0, trans_z = 0):
    unit_3D[..., -1] = (10.0 - unit_3D[..., -1])
    # unit_3D = unit_3D[..., :1] / unit_3D[..., 1:]
    R = np.eye(3)
    print("INPUT : {}\n{}\norts : {}\n{}".format(R.shape, R, unit_3D.shape, unit_3D))
    trans = np.array([[trans_x, trans_y, trans_z]], dtype="double")


    camera_matrix = np.array([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], dtype=np.float32).reshape((3, 3)) #.T
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion 
    print("[DEBUG]dist {} {}\npersc_proj : {} \n{}".format(dist_coeffs, dist_coeffs.shape, camera_matrix.shape, camera_matrix))
    imgpts, jac = cv2.projectPoints(unit_3D, R, trans, camera_matrix, dist_coeffs)
    print("[DEBUG]end-points {} \n{}\njacobian {}\n{}".format(imgpts.shape, imgpts, jac, jac.shape))
    return tuple(imgpts[0].astype(int).ravel()), tuple(imgpts[1].astype(int).ravel()), tuple(imgpts[2].astype(int).ravel())

def visualize_rpy_on_image(src, R, scale=1, thickness=1, center=None, text_offset=5, font_scale = 0.43, text_thickness = 1, RPYs=[]):
    """
    在输入图像上可视化 RPY 欧拉角（支持自定义原点）
    
    参数:
        src (np.ndarray): 输入图像 (BGR 格式)
        R(np.ndarry <1*3*3> or <3*3, by rpy>): rotaion matrix (has been transposed, A ROT B -> matA @ AtoB.T)
        scale (int): 坐标轴长度
        thickness (int): 线条粗细
        center (tuple): 坐标轴原点 (x, y)，默认图像中心
        text_offset (int): 文本与箭头的像素偏移
    
    返回:
        np.ndarray: 可视化后的图像
    """
    img = src.copy()
    h, w = img.shape[:2]
    
    # 设置原点（默认图像中心）
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = center
    
    # 定义 3D 坐标轴端点 (X, Y, Z)
    axes = np.float32([
        [1, 0, 0],  # X 轴 (红色)
        [0, -1, 0],  # Y 轴 (绿色)
        [0, 0, 1]   # Z 轴 (蓝色)
    ]) * scale

    print("xyz org: ({}, {}) , ({}, {}), ({}, {})".format(axes.T[0, 0].astype(int), axes.T[1, 0].astype(int),
axes.T[0, 1].astype(int),axes.T[1, 1].astype(int), axes.T[0, 2].astype(int), axes.T[1, 2].astype(int)))

    #x_end, y_end, z_end = orientation_rot_map_2D(axes, R, cx, cy)
    x_end, y_end, z_end = orientation_jacobian_map_2D(axes, R)
    print("x: {} ({}), y: {}({}), z:{}({})".format(x_end, type(x_end), y_end, type(y_end), z_end, type(z_end)))
    # 绘制坐标轴
    cv2.line(img, (cx, cy), x_end, (0, 0, 255), thickness)  # X: 红色 (Pitch)
    cv2.line(img, (cx, cy), y_end, (0, 255, 0), thickness)  # Y: 绿色 (Yaw)
    cv2.line(img, (cx, cy), z_end, (255, 0, 0), thickness)  # Z: 蓝色 (Roll)
    
    # 在箭头附近标注 RPY 名称和角度°
    font = cv2.FONT_HERSHEY_SIMPLEX
    roll_deg = -720
    pitch_deg = -720
    yaw_deg = -720
    if len(RPYs) == 3:
        roll_deg = RPYs[0]
        pitch_deg = RPYs[1]
        yaw_deg = RPYs[2]
    else:
        roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy(R)
        print("[DEBUG]visual rpy: {}, {}, {}".format(roll_deg, pitch_deg, yaw_deg))
    
    cv2.putText(img, f"P:{ceil(pitch_deg)}", 
                (x_end[0] - text_offset, x_end[1] + text_offset), 
                font, font_scale, (0, 0, 255), text_thickness)
    cv2.putText(img, f"Y:{ceil(yaw_deg)}", 
                (y_end[0] + text_offset, y_end[1] + text_offset), 
                font, font_scale, (0, 255, 0), text_thickness)
    cv2.putText(img, f"R:{ceil(roll_deg)}", 
                (z_end[0] + text_offset, z_end[1] - text_offset), 
                font, font_scale, (255, 0, 0), text_thickness)
    
    return img
 

