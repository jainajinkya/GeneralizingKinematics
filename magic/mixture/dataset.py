import os

import h5py
import numpy as np
import torch
from GeneralizingKinematics.magic.noise_models import Distractor
from torch.utils.data import Dataset


# %%
from magic.mixture.utils import change_frames


class MixtureDataset(Dataset):
    def __init__(self,
                 ntrain,
                 root_dir,
                 bounds=[],
                 n_dof=1,
                 normalize=False,
                 transform=None,
                 test=False,
                 keep_columns=[],
                 one_columns=[],
                 preserve_labels=False,
                 distractor_prob=0.0):
        super(MixtureDataset, self).__init__()
        '''
            shapes and kinematics and stuff
        '''
        self.root_dir = root_dir
        # self.labels_frame = pd.read_csv(os.path.join(root_dir, 'params.csv'), header=None)
        # self.raw_labels = self.labels_frame.values
        self.labels_frame = None
        self.raw_labels = self.get_raw_labels()
        self.length = ntrain
        self.n_dof = n_dof
        self.transform = transform

        if not test:
            if normalize:
                if bounds == []:
                    self.compute_bounds()
                    np.save(os.path.join(root_dir, 'bounds.npy'), self.bounds)
                else:
                    self.bounds = bounds
                print('normalizing.')
                self.normalize_labels()
            else:
                self.bounds = []

        self.raw_torch = torch.from_numpy(self.raw_labels).float()
        axes = self.get_axes()
        radii = self.get_radii()
        configs = self.get_angles()
        geoms = self.get_geometry()
        poses = self.get_poses()

        # concatenate all the quantities into one labels datastructure
        self.full_labels = torch.cat((axes,
                                      configs,
                                      radii,
                                      geoms,
                                      poses), dim=1)

        if keep_columns == []:
            # compute which columns are constant zeros or constant ones
            x = self.full_labels
            self.zero_columns = [i for i in range(x.shape[1]) if not x[:, i].byte().any()]
            self.one_columns = [i for i in range(x.shape[1]) if x[:, i].byte().all()]
            self.keep_columns = [i for i in range(x.shape[1]) if
                                 (i not in self.zero_columns) and (i not in self.one_columns)]
        else:
            self.keep_columns = keep_columns
            self.one_columns = one_columns

        # throw out constant columns for stability!
        print('full labels shape', self.full_labels.shape)
        self.labels = self.full_labels[:, self.keep_columns]
        print('labels shape', self.labels.shape)

        # set up distractor in the background
        self.distractor_prob = distractor_prob
        self.distractor = Distractor(trans_weight=0.0,
                                     rotate=False)

        print("keep_columns: ", self.keep_columns)
        print("one_columns: ", self.one_columns)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # load depth
        # path = os.path.join(self.root_dir, 'depth' + str(idx).zfill(6) + '.pt')
        # depth = torch.load(path)
        # if self.labels_frame is None:
        #     self.labels_frame = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')

        obj_idx, obj_data_idx = self.idx_to_h5py_dataset_idx(idx)
        with h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r') as h5File:
            depth = torch.from_numpy(h5File['obj_' + str(obj_idx).zfill(6)]['depth_imgs'][obj_data_idx])

        # random other depth image
        if np.random.rand() < self.distractor_prob:
            other_idx = np.random.randint(self.length)
            # path2 = os.path.join(self.root_dir, 'depth' + str(other_idx).zfill(6) + '.pt')
            # depth2 = torch.load(path2)
            other_obj_idx, other_data_idx = self.idx_to_h5py_dataset_idx(other_idx)
            with h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r') as h5File:
                depth2 = torch.from_numpy(h5File['obj_' + str(other_obj_idx).zfill(6)]['depth_imgs'][other_data_idx])
            depth = self.distractor(depth, depth2)

        depth = depth.unsqueeze(0).float()

        if self.transform is not None:
            depth = self.transform(depth)

        label = self.labels[idx]
        depth = torch.cat((depth, depth, depth))
        sample = {'depth': depth,
                  'label': label}
        return sample

    def get_axes(self):
        # print('Loading axes.')
        if self.n_dof == 1:
            axes = self.raw_torch.narrow(1, 5, 7)
        else:
            axes = self.raw_torch.narrow(1, 5, 14)
        # print(axes.size())
        return axes

    def get_radii(self):
        # print('Loading radii.')
        if self.n_dof == 1:
            radii = self.raw_torch.narrow(1, 12, 3)
        else:
            radii = self.raw_torch.narrow(1, 19, 6)
        # print(radii.size())
        return radii

    def get_geometry(self):
        geom = self.raw_torch.narrow(1, 1, 4)
        return geom

    def get_poses(self):
        if self.n_dof == 1:
            xyz = self.raw_torch.narrow(1, 15, 3).view(-1, 3)
            quat = self.raw_torch.narrow(1, 18, 4).view(-1, 4)
        else:
            xyz = self.raw_torch.narrow(1, 25, 3).view(-1, 3)
            quat = self.raw_torch.narrow(1, 28, 4).view(-1, 4)
        poses = torch.cat((xyz, quat), 1)
        return poses

    def get_angles(self):
        if self.n_dof == 1:
            angles = self.raw_torch.narrow(1, 22, 1).view(-1, 1)
        else:
            angles = self.raw_torch.narrow(1, 32, 2).view(-1, 2)
        return angles

    def compute_bounds(self):
        self.bounds = np.zeros((self.raw_labels.shape[1], 2))
        for d in range(self.raw_labels.shape[1]):
            dmin = min(self.raw_labels[:, d])
            dmax = max(self.raw_labels[:, d])
            self.bounds[d, 0] = dmin
            self.bounds[d, 1] = dmax

    def normalize_dim(self, data, bounds, dimension):
        dmax, dmin = bounds[dimension, 1], bounds[dimension, 0]
        data[:, dimension] = (data[:, dimension] - dmin) / (dmax - dmin)

    def normalize_labels(self):
        for d in range(self.raw_labels.shape[1]):
            dmin = self.bounds[d, 0]
            dmax = self.bounds[d, 1]
            if (dmin == 0 and dmax == 0) or (dmin == 1 and dmax == 1) or (dmin == dmax):
                pass
            else:
                self.normalize_dim(self.raw_labels, self.bounds, d)

    def get_raw_labels(self):
        # if self.labels_frame is None:
        raw_labels = []

        with h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r') as h5File:
            for obj in h5File.values():
                embeds = np.array(obj['embedding_and_params'])
                ##### Changing axis to local coordinates #####
                axis_in_world = embeds[5:12]
                obj_pose_in_world = embeds[-7:]
                axis_in_obj = change_frames(obj_pose_in_world, axis_in_world)
                embeds[5:12] = axis_in_obj
                #############################################
                q_vals = np.array(obj['q'])
                for q in q_vals:
                    row = np.concatenate((embeds, q))
                    raw_labels.append(row)
        return np.array(raw_labels)

    def idx_to_h5py_dataset_idx(self, idx, imgs_per_obj=16):
        obj_idx = int(idx / imgs_per_obj)
        obj_data_idx = idx % imgs_per_obj
        return obj_idx, obj_data_idx
