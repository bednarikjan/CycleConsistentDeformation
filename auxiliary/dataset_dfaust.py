# 3rd party
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Project files.
import cyccon.auxiliary.normalize_points as normalize_points
import cyccon.auxiliary.useful_losses as loss

# Python std.
from timeit import default_timer as timer
import jblib.file_sys as jbfs
import os


class PointsTensorDataset(torch.utils.data.Dataset):
    def __init__(self, points_tensor, n_points=0, resample_points=False,
                 normalization_matrices=None, orientation_matrices=None):
        self.points_tensor = points_tensor
        if n_points==0:
            n_points = self.points_tensor.shape[1]
        # randomly sample n_points from points available in the tensor
        assert(self.points_tensor.shape[1] >= n_points)
        # some normalization code needs to be tweaked to enable arbitrary
        # dims... TODO: support arbitrary dims
        assert(self.points_tensor.shape[2] == 3)
        self.resample_points = resample_points
        self.n_points = n_points
        self.normalization_matrices = normalization_matrices
        assert(self.normalization_matrices == None or
               self.normalization_matrices.shape[0] ==
               self.points_tensor.shape[0])
        self.orientation_matrices = orientation_matrices
        assert(self.normalization_matrices == None or
               self.orientation_matrices.shape[0] ==
               self.points_tensor.shape[0])

    def __len__(self):
        return self.points_tensor.shape[0]

    def __getitem__(self, idx):
        if self.resample_points:
            sampled_vertices = np.random.choice(
                range(0, self.points_tensor.shape[1]), self.n_points)
        else:
            sampled_vertices = np.arange(self.n_points, dtype=int)
        if self.orientation_matrices is None \
                and self.normalization_matrices is None:
            return self.points_tensor[idx][sampled_vertices]
        #
        # if dataset needs normalization
        if self.orientation_matrices is not None:
            orientation_matrix = np.eye(4,4)
            orientation_matrix[:3, :3] = self.orientation_matrices[idx]
            if self.normalization_matrices is not None:
                object_normalization = np.matmul(
                    orientation_matrix, self.normalization_matrices[idx])
            else:
                object_normalization = orientation_matrix
        elif self.normalization_matrices is not None:
            object_normalization = self.normalization_matrices[idx]
        else:
            assert(False)
        #
        normalized_point_cloud = self.points_tensor[idx][sampled_vertices]
        ones = np.ones([1, normalized_point_cloud.shape[0]])
        normalized_point_cloud = object_normalization.dot(
            np.concatenate([normalized_point_cloud.T, ones])).T[:, 0:3]
        #
        return normalized_point_cloud


class DatasetDFAUSTTriplets(torch.utils.data.Dataset):
    path_ds = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_all_10k.npy'
    path_seq2inds = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_all_10k_seq2ind.npz'
    path_reg_m = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_m.hdf5'
    path_reg_f = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_f.hdf5'
    path_areas = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_areas_all.npy'
    path_neighbors = '/cvlabdata2/cvlab/datasets_jan/dfaust/knn_neighbors'

    def __init__(self, num_pts=2500, subjects='all', sequences='all',
                 subj_seq=None, mode='random', mode_params=None,
                 resample_pts=True, z_rot_range=40, knn=False, num_neighbors=20,
                 get_single_shape=False):
        super(DatasetDFAUSTTriplets)

        # Store options.
        self._z_rot_range = z_rot_range
        self._get_single_shape = get_single_shape
        self._knn = knn
        self._knn_neighbs = num_neighbors

        # Get pcloud ds and seq2id mapping.
        pts_tensor = np.load(DatasetDFAUSTTriplets.path_ds, mmap_mode='r')
        self._pcloud_ds = PointsTensorDataset(
            pts_tensor, n_points=num_pts, resample_points=resample_pts)
        seq2id = np.load(DatasetDFAUSTTriplets.path_seq2inds)

        subjects_all, sequences_all = zip(
            *[it.split('_', maxsplit=1) for it in seq2id.keys()])
        subjects_all = np.sort(np.unique(subjects_all)).tolist()
        sequences_all = np.sort(np.unique(sequences_all)).tolist()

        # Normalize format of selected subjects and sequences.
        if subj_seq is None:
            assert subjects is not None and sequences is not None
            if isinstance(subjects, str):
                assert subjects == 'all'
                subjects = subjects_all
            if isinstance(sequences, str):
                assert sequences == 'all'
                sequences = sequences_all
            subj_seq = {sub: sequences for sub in subjects}
        else:
            if subjects is not None or sequences is not None:
                print('[WARNING]: Using "subj_seq" argument to gather data '
                      'sequences even though "subjects" and/or "sequences" '
                      'are not None (they will be ignored).')
            assert isinstance(subj_seq, dict)

        self.dataset_string_args = self.ds2str(
            num_pts, subj_seq, resample_pts, knn, num_neighbors)

        # Get indices of selected data samples.
        self._inds_all = []
        self._reg_keys_inds = []
        seq_start_inds = [0]
        for subj in sorted(list(subj_seq.keys())):
            for seq in sorted(subj_seq[subj]):
                k = f"{subj}_{seq}"
                if k not in seq2id:
                    print(f"[WARNING]: Requested sequence {k} is no "
                           "part of the dataset and will be ingnored.")
                    continue
                fr, to = seq2id[k]
                self._inds_all.extend(list(range(fr, to)))
                self._reg_keys_inds.extend([(k, i) for i in range(to - fr)])
                seq_start_inds.append(seq_start_inds[-1] + to - fr)
        self._seq_start_inds = np.array(seq_start_inds)
        assert len(self._inds_all) == len(self._reg_keys_inds)

        # Get mode.
        assert mode in ('random', 'within_seq', 'neighbors')
        self._mode = mode
        self._mode_params = mode_params
        if mode == 'neighbors':
            assert 'max_frames' in mode_params

        # Get kNNs.
        if self._knn:
            path_neighbs_inds = jbfs.jn(
                DatasetDFAUSTTriplets.path_neighbors,
                self.dataset_string_args + '_inds.npy')
            path_neighbs_dists = jbfs.jn(
                DatasetDFAUSTTriplets.path_neighbors,
                self.dataset_string_args + '_dists.npy')

            if not (os.path.exists(path_neighbs_inds) and
                    os.path.exists(path_neighbs_dists)):
                self._knn_inds, self._knn_dists = \
                    self._compute_nearest_neighbors_graph(
                        path_neighbs_inds, path_neighbs_dists)
            else:
                self._knn_inds = np.load(path_neighbs_inds)
                self._knn_dists = np.load(path_neighbs_dists)

            # self._knn_inds = np.load(path_neighbs_inds) \
            #     if os.path.exists(path_neighbs_inds) \
            #     else self._compute_nearest_neighbors_graph(
            #         path_neighbs_inds, path_neighbs_dists)

    def ds2str(self, num_pts, subj_seq, resample_pts, knn, num_neihbors):
        keys = sorted(list(subj_seq.keys()))
        ss_str = ''
        for k in keys:
            ss_str += k
            ss_str += '_' + '-'.join(subj_seq[k])
        return f"N{num_pts}_{ss_str}_res{('F', 'T')[resample_pts]}_" \
               f"knn{('F', 'T')[knn]}-{num_neihbors}"

    def _compute_nearest_neighbors_graph(self, out_path_inds, out_path_dists):
        print('Computing kNN graph')
        ts = timer()

        # Get all points.
        smpls_all = []
        for i, ind in enumerate(self._inds_all):
            print(f"\rLoading ds {i + 1}/{len(self._inds_all)}", end='')
            smpls_all.append(self._pcloud_ds[ind])
        pts = np.stack(smpls_all, axis=0).\
            reshape((len(smpls_all), -1)) # (num_samples, num_pts * 3)

        # Train and predict kNN.
        print('\nComputing nearest neighbors.')
        self._knn_neighbs = min(self._knn_neighbs, len(self._inds_all))
        nbrs = NearestNeighbors(
            n_neighbors=self._knn_neighbs, algorithm='ball_tree',
            metric=loss.NN_metric, n_jobs=4).fit(pts)
        # knn_inds = nbrs.kneighbors(pts)[1]
        knn_dists, knn_inds = nbrs.kneighbors(pts)

        # Save indices.
        np.save(out_path_inds, knn_inds)
        np.save(out_path_dists, knn_dists)
        print(f"Finished in {(timer() - ts) / 60.} minutes.")
        return knn_inds, knn_dists

    def __len__(self):
        return len(self._inds_all)

    def __getitem__(self, idx):
        # Get ind range of the idx1.
        it = np.sum(idx >= self._seq_start_inds)
        fr1, to1 = self._seq_start_inds[it - 1:it + 1]

        inds = [idx]
        if not self._get_single_shape:
            if self._knn:
                inds.extend(self._knn_inds[[idx, idx], np.random.randint(
                    0, self._knn_neighbs, (2,))].tolist())

                # debug
                # print(inds)
            else:
                # Get idx2 of the second sample in the pair.
                if self._mode == 'random':
                    smpl_fr, smpl_to = 0, len(self)
                elif self._mode == 'within_seq':
                    smpl_fr, smpl_to = fr1, to1
                elif self._mode == 'neighbors':
                    mf = self._mode_params['max_frames']
                    smpl_fr, smpl_to = \
                        np.maximum(fr1, idx - mf), np.minimum(to1, idx + mf + 1)

                attempts = 0
                while True:
                    if len(inds) == 3:
                        break
                    cand_idx = np.random.randint(smpl_fr, smpl_to)
                    if cand_idx not in inds:
                        inds.append(cand_idx)
                    attempts += 1
                    if attempts > 100:
                        print(f"[ERROR]: Finding a three distinct indices "
                              f"failed after 100 attempts, returning a triplet "
                              f"containing two or three identical indices.")
                        break

        # Get pts and inds.
        pts = torch.stack([torch.from_numpy(
            self._pcloud_ds[self._inds_all[i]]) for i in inds], dim=0)

        # Add zeros insted of normals (expected format by Cycle Consistency).
        pts_mock_norm = torch.cat(
            [pts, torch.zeros(pts.shape[:2] + (4, ),
                              dtype=torch.float32)], dim=2)

        # Generate a rotation matrix.
        rm = normalize_points.uniform_rotation_axis_matrix(
            axis=1, range_rot=self._z_rot_range)

        smpls = (pts_mock_norm[0], 0, rm, 0,
                 pts_mock_norm[1], 0, rm, 0,
                 pts_mock_norm[2], 0, rm, 0) if not self._get_single_shape \
            else (pts_mock_norm[0], 0, rm, 0)

        return smpls
