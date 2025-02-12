import torch.utils.data as data
from utils.general import *
import trimesh
from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import utils.general as utils


class ReconDataSet(data.Dataset):

    def __init__(self, split: str, dataset_path: str, dist_file_name: str):
        """初始化 Recon 数据集.

        Args:
            split (str):  数据集划分（未使用）.
            dataset_path (str):  3D模型文件路径.
            dist_file_name (str): 距离场文件名后缀（未使用）.
        """
        model = trimesh.load(dataset_path)

        # 如果模型是点云，则直接使用顶点数据; 否则，对模型进行采样以获取点云数据.
        if not isinstance(model, trimesh.PointCloud):
            model = utils.as_mesh(trimesh.load(dataset_path))
            if model.faces.shape[0] > 0:
                self.points = sample_surface(model, 250000)[0]
            else:
                self.points = model.vertices
        else:
            self.points = model.vertices

        # 将点云数据归一化到单位球面上.
        self.points = self.points - self.points.mean(0, keepdims=True)
        scale = np.abs(self.points).max()
        self.points = self.points / scale

        # 计算点云数据的距离场.
        sigmas = []

        ptree = cKDTree(self.points)

        for p in tqdm(np.array_split(self.points, 10, axis=0)):
            d = ptree.query(p, np.int(51.0))
            sigmas.append(d[0][:, -1])

        sigmas = np.concatenate(sigmas)
        sigmas = np.tile(sigmas, [250000 // sigmas.shape[0]])
        sigmas_big = 1.0 * np.ones_like(sigmas)
        pnts = np.tile(self.points, [250000 // self.points.shape[0], 1])

        sample = np.concatenate(
            [
                pnts
                + np.expand_dims(sigmas, -1)
                * np.random.normal(0.0, 1.0, size=pnts.shape),
                pnts
                + np.expand_dims(sigmas_big, -1)
                * np.random.normal(0.0, 1.0, size=pnts.shape),
            ],
            axis=0,
        )

        dists = []

        for np_query in tqdm(sample):
            dist = ptree.query(np_query)[0]
            dists.append(dist)

        dists = np.array(dists)
        self.dists = np.concatenate([sample, np.expand_dims(dists, axis=-1)], axis=-1)
        self.npyfiles_mnfld = [dataset_path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """获取数据集中指定索引的数据.

        从points和dists中随机采样128^2个点, 并返回采样后的点云数据、距离场数据和索引.

        Args:
            index (int): 数据索引.
        """
        point_set_mnlfld = torch.from_numpy(self.points).float()

        sample_non_mnfld = torch.from_numpy(self.dists).float()

        random_idx = (torch.rand(128**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx)

        random_idx = (torch.rand(128**2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld, 0, random_idx)

        return point_set_mnlfld, sample_non_mnfld, index

    def __len__(self):
        return 1
