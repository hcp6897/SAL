from __future__ import print_function

import os
import sys
sys.path.append('../code')

import json
import argparse
import trimesh
from trimesh.sample import sample_surface
import numpy as np
from scipy.spatial import cKDTree

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

import utils.general as utils


def as_mesh(scene_or_mesh):
    r"""Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    
    return mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help="Path of unzipped d-faust scans.")
    parser.add_argument('--split', type=str, default='../confs/splits/dfaust/train_all_every5.json', help="Split file name.")
    parser.add_argument('--shapeindex', type=int, required=True, help="Shape index to be preprocessed. -1 for all")
    parser.add_argument('--skip', action="store_true", default=False)
    parser.add_argument('--sigma', type=float, default=0.2)

    opt = parser.parse_args()

    with open(opt.split, "r") as f:
        train_split = json.load(f)

    shapeindex = opt.shapeindex
    global_shapeindex = 0

    for human, scans in train_split['scans'].items():
        for pose, shapes in scans.items():
            source = '{0}/{1}/{2}/{3}'.format(opt.datapath, 'scans', human, pose)
            output = opt.datapath + '_processed'
            utils.mkdir_ifnotexists(output)
            utils.mkdir_ifnotexists(os.path.join(output, human))
            utils.mkdir_ifnotexists(os.path.join(output, human, pose))

            for shape in shapes:
                if (shapeindex == global_shapeindex or shapeindex == -1):
                    print ("The {} has been found.".format(shape))
                    output_file = os.path.join(output, human, pose, shape)
                    print (output_file)
                    
                    if (not opt.skip ) or (not os.path.isfile(output_file + '_dist_triangle.npy')):
                        print ('loading : {0}'.format(os.path.join(source, shape)))
                        mesh = trimesh.load(os.path.join(source, shape) + '.ply')
                        
                        """采样点云
                        这段代码从网格中采样250000个点，并将这些点中心化（即减去中心点坐标），最后进行缩放（虽然缩放因子为1，实际没有缩放效果）。
                        这种操作通常用于点云数据的预处理，使得点云数据在空间中居中，便于后续的处理或分析。
                        """
                        sample = sample_surface(mesh, 250000)   # 250000 points
                        center = np.mean(sample[0], axis=0)
                        pnts = sample[0]
                        pnts = pnts - np.expand_dims(center, axis=0)
                        scale = 1
                        pnts = pnts/scale
                        
                        """采样三角面片
                        对网格中的每个三角面片进行归一化处理，使其相对于网格中心点进行平移，并通过缩放因子 scale 进行缩放.
                        处理后的三角面片存储在 triangles 列表中，以便后续使用.
                        这种处理通常用于将网格数据标准化，以便在不同的算法或模型中使用.
                        """
                        triangles = []  # triangle soups
                        for tri in mesh.triangles:
                            a = Point_3((tri[0][0] - center[0])/scale, (tri[0][1] - center[1])/scale, (tri[0][2] - center[2])/scale)   # 三角面片的第一个顶点
                            b = Point_3((tri[1][0] - center[0])/scale, (tri[1][1] - center[1])/scale, (tri[1][2] - center[2])/scale)   # 三角面片的第二个顶点
                            c = Point_3((tri[2][0] - center[0])/scale, (tri[2][1] - center[1])/scale, (tri[2][2] - center[2])/scale)   # 三角面片的第三个顶点
                            triangles.append(Triangle_3(a, b, c)) # 将处理后的三个顶点 a, b, c 组成一个新的三角面片 Triangle_3，并将其添加到 triangles 列表中.
                        
                        # AABB tree for triangle soups
                        tree = AABB_tree_Triangle_3_soup(triangles) 

                        """
                        每个点定义两个不同的噪声标准差：局部噪声标准差 sigmas (可以理解为该点的局部密度或噪声水平) 和 全局噪声标准差 sigmas_big.
                        """
                        sigmas = []
                        ptree = cKDTree(pnts) # 构建点云的KDtree
                        i = 0
                        # 计算每个点的51个最近邻点(K-nearest neighbors)的距离
                        for p in np.array_split(pnts, 100, axis=0):     # 将点云分成100个部分进行处理
                            d = ptree.query(p, 51)  # 计算查询点的KNN（K=51）距离值，返回两个数组：第一个数组是距离，第二个数组是索引. 
                            sigmas.append(d[0][:, -1])  # 对于每个点，提取第51个最近邻距离，来估计局部密度或噪声水平. 

                            i = i+1

                        # 局部噪声标准差 sigmas
                        sigmas = np.concatenate(sigmas)
                        # 全局噪声标准差 sigmas_big，对所有点都是相同的.
                        sigmas_big = opt.sigma * np.ones_like(sigmas)

                        # 保存点云
                        np.save(output_file + '.npy', pnts)

                        # 使用两个不同的噪声标准差（sigmas 和 sigmas_big）来对原始点集 pnts 进行扰动，然后将扰动后的点集拼接在一起.
                        sample = np.concatenate(
                            [pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape), 
                             pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)], 
                            axis=0)

                        # 计算一组查询点（sample）到最近三角形网格的距离
                        dists = []
                        for np_query in sample:
                            cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double), np_query[2].astype(np.double))
                            
                            cp = tree.closest_point(cgal_query)  # 计算查询点到最近三角面片上的点 cp
                            cp = np.array([cp.x(), cp.y(), cp.z()])  # 将 cp 转换为 numpy 数组
                            dist = np.sqrt(((cp - np_query)**2).sum(axis=0))  # 计算 np_query 到 cp 的距离

                            dists.append(dist)  # 将距离添加到 dists 列表中

                        dists = np.array(dists)  # 将 dists 列表转换为 numpy 数组
                        np.save(output_file + '_dist_triangle.npy',
                                np.concatenate([sample, np.expand_dims(dists, axis=-1)], axis=-1))  # 将查询点 sample和对应的距离 dists 合并为一个数组.

                        np.save(output_file + '_normalization.npy', 
                                {"center":center, "scale":scale})  # 将归一化参数 center 和 scale 保存为一个字典，并保存为 npy 文件.

                global_shapeindex = global_shapeindex + 1

    print ("end!")