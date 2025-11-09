import numpy as np
from .renderer import create_renderer
# from renderer import create_renderer
import open3d as o3d
from utils import *
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time


class Render:
    def __init__(self, K, image_size, model_path, model_name=None, type=None):
        self.model_name = model_name
        self.K = K
        self.image_size = image_size
        self.type = type
        self.renderer = create_renderer(*(self.image_size), 'vispy', self.type, 'phong')
        # self.model_fn = model_path
        self.model_pcd = o3d.io.read_point_cloud(model_path)  # 模型点云
        self.renderer.add_object(0, model_path)  # 给renderer设置模型

    def RenderSilhouette(self, PCO):
        # time0 = time.time()
        res = self.renderer.render_object(0, PCO[:3, :3], PCO[:3, 3].T, self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])
        # print(f'渲染一次耗时 {time.time() - time0}s')
        if self.type == 'mask':
            grey = cv2.cvtColor(res['mask'], cv2.COLOR_RGB2GRAY)
        elif self.type == 'rgb':
            grey = cv2.cvtColor(res['rgb'], cv2.COLOR_RGB2GRAY)
        elif self.type == 'depth':
            grey = res['depth']
        return grey

    def RenderMutiImgbyCameraPosition(self, camera_position, degree_step):
        """
        为给定的camera_position渲染多角度mask，让mask大致均匀的出现在图像的所有位置
        :param camera_position: (r,θ,φ)
        :param degree_step: 旋转度数步长
        :return:
        """
        # 模型中心d
        model_center = np.mean(np.asarray(self.model_pcd.points),axis=0)
        # 相机光心的xyz坐标    (√)
        c_x , c_y, c_z = Sph2Car(*camera_position)
        print(c_x,c_y,c_z)
        # 相机z轴方向向量（object坐标系下）  (√)
        dir_z = model_center - np.array([c_x, c_y, c_z])
        dir_z = dir_z / np.linalg.norm(dir_z)
        # 叉乘求x轴，y轴的方向向量     (√)
        dir_x = np.cross(dir_z, np.array([0,1,0]))
        dir_x = dir_x / np.linalg.norm(dir_x)
        dir_y = np.cross(dir_z, dir_x)
        dir_y = dir_y / np.linalg.norm(dir_y)
        # R_camera表示物体坐标系下相机初始位姿的旋转矩阵
        R_camera = np.column_stack((dir_x,dir_y,dir_z))

        # ------------------------------ 开始渲染 ------------------------------ #
        current_file = os.path.abspath(__file__)
        fp = os.path.dirname(current_file)
        gfp = os.path.dirname(fp)
        # 存放渲染数据集的根文件夹名称
        root_render_dataset_dirname = "render_mask_dataset"
        root_render_dataset_path = os.path.join(gfp, root_render_dataset_dirname)
        if not os.path.exists(root_render_dataset_path):
            os.makedirs(root_render_dataset_path)
        curModel_mask_dataset_path = os.path.join(root_render_dataset_path, self.model_name)
        # 渲染当前所给模型的mask数据集
        if not os.path.exists(curModel_mask_dataset_path):
            os.makedirs(curModel_mask_dataset_path)


        # 因为根据camera position渲染mask，需要考虑的是相机的纯旋转
        # 但是OpenGL的渲染函数是通过PCO来渲染的，所以需要确定给定相机位姿POC的情况下，相机能拍到物体的旋转角度范围
        # 采取的判断方式是，先用大步长探索rx,ry的边界条件，因为z轴旋转不会导致拍不到物体所以不考虑rz的取值范围
        # 判定x的旋转取值范围
        for x in range(0, 90, degree_step):
            xrotation = Rotation.from_euler(seq='x', angles=x, degrees=True)
            xR = xrotation.as_matrix()
            xR_camera = R_camera @ xR
            xPOC = GetTMatFromXYZR(c_x, c_y, c_z, xR_camera)
            xPCO = np.linalg.inv(xPOC)
            xmask = self.RenderSilhouette(PCO=xPCO)
            if CheckContourInMask(xmask) == False:
                x_ub = x - degree_step
                break

        # 判定y的旋转取值范围
        for y in range(0, 90, degree_step):
            yrotation = Rotation.from_euler(seq='y', angles=y, degrees=True)
            yR = yrotation.as_matrix()
            yR_camera = R_camera @ yR
            yPOC = GetTMatFromXYZR(c_x, c_y, c_z, yR_camera)
            yPCO = np.linalg.inv(yPOC)
            ymask = self.RenderSilhouette(PCO=yPCO)
            if CheckContourInMask(ymask) == False:
                y_ub = y - degree_step
                break


        for x in tqdm(range(-x_ub, x_ub, 3)):
            for y in range(-y_ub, y_ub, 3):
                for z in range(0, 360, degree_step * 6):
                    rotation = Rotation.from_euler(seq='xyz', angles=np.array([x, y, z]), degrees=True)
                    R_obj = rotation.as_matrix()
                    iR_camera = R_camera @ R_obj
                    POC = GetTMatFromXYZR(c_x, c_y, c_z, iR_camera)
                    PCO = np.linalg.inv(POC)
                    mask = self.RenderSilhouette(PCO=PCO)
                    if CheckContourInMask(mask) == True:
                        mask_fp = os.path.join(curModel_mask_dataset_path, GetPoseStrFromT(np.linalg.inv(PCO)) + '.png')
                        cv2.imwrite(mask_fp, mask)

        print("camera pose: {}, 共生成了 {} 个mask".format(','.join(map(str, [c_x , c_y, c_z])), len(os.listdir(curModel_mask_dataset_path))))




def get_T_from_position(r, theta, phi):
    '''
    根据 camera 在 object 坐标系下的 position, 求出相机z轴指向物体中心时, 物体在相机坐标系下的 T 矩阵
    :param r:
    :param theta:
    :param phi:
    :return:
    '''
    # 初始位置为粗分类确定, 朝向为相机z轴指向物体中心
    c_x, c_y, c_z = Sph2Car(r, theta, phi)
    # 相机z轴方向向量（object坐标系下）  (√)
    dir_z = np.array([0, 0, 0]) - np.array([c_x, c_y, c_z])
    dir_z = dir_z / np.linalg.norm(dir_z)
    # 叉乘求x轴，y轴的方向向量     (√)
    dir_x = np.cross(dir_z, np.array([0, 1, 0]))
    dir_x = dir_x / np.linalg.norm(dir_x)
    dir_y = np.cross(dir_z, dir_x)
    dir_y = dir_y / np.linalg.norm(dir_y)
    # R_camera表示物体坐标系下相机初始位姿的旋转矩阵
    R_camera = np.column_stack((dir_x, dir_y, dir_z))
    return np.linalg.inv(GetTMatFromXYZR(c_x, c_y, c_z, R_camera))


def get_T_from_XYZposition(x, y, z):
    '''
    根据 camera 在 object 坐标系下的 xyz坐标表示的position, 求出相机z轴指向物体中心时, 物体在相机坐标系下的 T 矩阵
    :param x:
    :param y:
    :param z:
    :return:
    '''
    dir_z = np.array([0, 0, 0]) - np.array([x, y, z])
    dir_z = dir_z / np.linalg.norm(dir_z)
    # 叉乘求x轴，y轴的方向向量     (√)
    dir_x = np.cross(dir_z, np.array([0, 1, 0]))
    dir_x = dir_x / np.linalg.norm(dir_x)
    dir_y = np.cross(dir_z, dir_x)
    dir_y = dir_y / np.linalg.norm(dir_y)
    # R_camera表示物体坐标系下相机初始位姿的旋转矩阵
    R_camera = np.column_stack((dir_x, dir_y, dir_z))
    return np.linalg.inv(GetTMatFromXYZR(x, y, z, R_camera))






from siou import SIoU



# if __name__ == "__main__":
#     image_size = (640,480)
#     K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
#     # img = cv2.imread('/home/zc/SRL-pose/render_mask_dataset/1.png')
#
#     Render = Render(model_name='cat', K=K, image_size=image_size, model_path='D:\研究生学习\研1上学习\图像合成技术\张驰-2023244242\model.ply')
#     # Pco = GetTMatFrom6DoF(0,0,0,0,0,0)
#     # print(Pco)
#     # Poc = GetTMatFrom6DoF(0.0,-0.0,-0.0,157.40259867023542,11.32162754436606,6.944207872662179)
#     # Pco = np.linalg.inv(Poc)
#     # print(Pco)
#     # x,y,z,rx,ry,rz = Get6DoFFromTMat(Pco)
#     # print('{},{},{},{},{},{}'.format(x,y,z,rx,ry,rz))
#     # render_img = Render.RenderSilhouette(Pco)
#     # plt_imshow(Cvtopencv2plt_imgMat(render_img))
#     # Render.RenderMutiImgbyCameraPosition((500,20,30),degree_step=10)
#     Pco = get_T_from_position(520, 60, 50)
#     mask_ref = Render.RenderSilhouette(Pco)
#     Poc = np.linalg.inv(Pco)
#     Poc = Poc @ GetTMatFrom6DoF(0,0,0,15,17,0)
#     mask_cur_1 = Render.RenderSilhouette(np.linalg.inv(Poc))
#
#     Pco_delta = GetTMatFrom6DoF(0,0,0,0,0,50)
#     Pco_t = Pco @ Pco_delta
#     mask_cur_2 = Render.RenderSilhouette(Pco_t)
#
#     cv2.imshow('mask_ref', mask_ref)
#     cv2.imshow('mask_cur_ZhuanXiangJi', mask_cur_1)
#     cv2.imshow('mask_cur_ZhuanWuTi', mask_cur_2)
#     print('相机转的siou', SIoU(mask_cur_1, mask_ref, K, 1024, 'icp'))
#     print('物体转的siou', SIoU(mask_cur_2, mask_ref, K, 1024, 'icp'))
#     cv2.waitKey()


