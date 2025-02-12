import os
import logging
import torch
import torch.utils.data as data

from utils.general import *
import utils.general as utils


class DFaustDataSet(data.Dataset):

    def __init__(
        self, split: dict, dataset_path: str, dist_file_name: str, with_gt: bool = False
    ):
        r"""初始化 DFaust 数据集."""
        base_dir = dataset_path

        self.npyfiles_mnfld = self.get_instance_filenames(base_dir, split)
        self.npyfiles_dist = self.get_instance_filenames(
            base_dir, split, dist_file_name
        )

        if with_gt:
            # Used only for evaluation
            self.normalization_files = self.get_instance_filenames(
                base_dir, split, "_normalization"
            )
            self.gt_files = self.get_instance_filenames(
                utils.concat_home_dir("datasets/dfaust/scripts"), split, "", "obj"
            )
            self.scans_files = self.get_instance_filenames(
                utils.concat_home_dir("datasets/dfaust/scans"), split, "", "ply"
            )
            self.shapenames = [x.split("/")[-1].split(".obj")[0] for x in self.gt_files]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        point_set_mnlfld = torch.from_numpy(np.load(self.npyfiles_mnfld[index])).float()

        sample_non_mnfld = torch.from_numpy(np.load(self.npyfiles_dist[index])).float()

        random_idx = (torch.rand(128**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx)

        random_idx = (torch.rand(128**2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld, 0, random_idx)

        return point_set_mnlfld[:, :3], sample_non_mnfld, index

    def __len__(self):
        return len(self.npyfiles_mnfld)

    def get_instance_filenames(
        self, base_dir: str, split: dict, ext: str = "", format: str = "npy"
    ) -> list[str]:
        r"""根据split配置生成所有数据文件的完整路径.

        返回包含所有有效文件路径的列表 `npyfiles`.

        Args:
            base_dir (str): 数据集根目录.
            split (dict): 数据集划分配置，是一个嵌套字典结构.
            ext (str, optional): 文件扩展名后缀，默认为空字符串.
            format (str, optional): 文件格式，默认为 "npy".

        Examples:
            >>> base_dir = "datasets/dfaust"
            >>> split = {"train": {"human": {"subject1": ["shape1", "shape2"]}}}
            >>> ext = "_dist"
            >>> format = "npy"
            >>> get_instance_filenames(base_dir, split, ext, format)
            [
                "datasets/dfaust/human/subject1/shape1_dist.npy",
                "datasets/dfaust/human/subject1/shape2_dist.npy"
            ]
        """
        npyfiles = []
        i = 0
        for dataset in split:
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    j = 0
                    for shape in split[dataset][class_name][instance_name]:
                        instance_filename = os.path.join(
                            base_dir,
                            class_name,
                            instance_name,
                            shape + "{0}.{1}".format(ext, format),
                        )

                        if not os.path.isfile(instance_filename):
                            logging.error(
                                'Requested non-existent file "'
                                + instance_filename
                                + "' {0} , {1}".format(i, j)
                            )
                            i = i + 1
                            j = j + 1
                        npyfiles.append(instance_filename)

        return npyfiles
