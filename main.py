import renderer
import numpy as np
import cv2
import time
import pyrr
import open3d as o3d

def GetR(rx,ry,rz)->np.ndarray:
    rx = np.array(pyrr.Matrix44.from_x_rotation(rx/180*3.14159))[:3,:3]
    ry = np.array(pyrr.Matrix44.from_y_rotation(ry/180*3.14159))[:3,:3]
    rz = np.array(pyrr.Matrix44.from_z_rotation(rz/180*3.14159))[:3,:3]
    tmp = rz.dot(ry).dot(rx)
    return tmp

#%% 

import open3d as o3d
import numpy as np

# 添加这个函数
def pointcloud_to_mesh(input_file, output_file):
    # 加载点云
    # pcd = o3d.io.read_point_cloud(input_file)
    #
    # # 估算法线
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=10))
    # pcd.orient_normals_consistent_tangent_plane(100)

    # # 泊松重建
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     pcd, depth=8, scale=1.1
    # )
    #
    # # 过滤低质量面
    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    #
    # 保存结果
    # # 加载 PLY 文件
    mesh = o3d.io.read_triangle_mesh(input_file)
    # 估算法线
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_file, mesh)
    return mesh

if __name__ == '__main__':
    input_file = "jixiebi.ply"
    # output_file = "temp.ply"
    # output_file = "droid_1 (1).ply"
    output_file = "jixiebi.ply"
    # output_file = "model.ply"

    # 使用新函数进行点云转mesh
    # mesh = pointcloud_to_mesh(input_file, output_file)
    #
    # # 加载 PLY 文件
    # mesh = o3d.io.read_triangle_mesh(input_file)
    # # 估算法线
    # mesh.compute_vertex_normals()
    # # 添加面（示例：假设有3个顶点，添加一个三角面）
    # if len(mesh.vertices) >= 3:
    #     mesh.triangles.append([0, 1, 2])
    #     mesh.triangle_normals = o3d.utility.Vector3dVector(np.zeros((len(mesh.triangles), 3)))
    # # 保存带法线和面的 PLY 文件
    # o3d.io.write_triangle_mesh(output_file, mesh)


    # ren_rgb = renderer.create_renderer(4096,3072,'vispy','rgb','phong') # 已经经过魔改，现在只出mask，这些光照模型是没有用上的
    ren_rgb = renderer.create_renderer(4096,3072,'vispy','rgb','flat') # 已经经过魔改，现在只出mask，这些光照模型是没有用上的
    ren_rgb.add_object(0,output_file)

    #%%
    st = time.time()
    for i in range(1):
        # R = GetR(40,50,40)
        R = GetR(90,0,0)
        res = ren_rgb.render_object(0,R,np.array([0,0,0.5]),1957.289,1957.043,2048.868,1524.94)
    print('time: ',time.time()-st)
    img = res['rgb']
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("limingyan.png", img_bgr)
    cv2.imwrite("limingyan_flat.png", img_bgr)
    # cv2.imwrite("limingyan.png",img)
    # cv2.imshow("hhh.png",img)
    # cv2.waitKey(0)

    # %%