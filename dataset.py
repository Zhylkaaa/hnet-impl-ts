import os
import random 

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L

import augmentations


class ECGPretrainDataset(Dataset):
    def __init__(self, data_path: str, multichannel: bool = False, randomize_leads: bool = False):
        self.data_path = data_path
        self.data = h5py.File(data_path, "r")
        self.leads = [k for k in self.data.keys() if k not in ['cum_seq_len', 'date', 'patient_id', 'study_id']]
        self.cum_seq_len = self.data["cum_seq_len"][()]
        self.multichannel = multichannel
        self.randomize_leads = randomize_leads
        self.pad_to_max = multichannel
        self.noise_data_mat = sio.loadmat('DATA_noises_real.mat')

    def __len__(self):
        return len(self.data["cum_seq_len"])
    
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries (e.g. file descriptors).
        del state["data"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Restore hdf5 dataset.
        self.data = h5py.File(self.data_path)
    
    def _process_ecg(self, lead: str, start_idx: int, end_idx: int):
        ecg = self.data[lead][start_idx:end_idx]
        ecg = self.impute(ecg)
        ecg = self.normalize(ecg)
        ecg = augmentations.ecg_positive_augmentation(ecg, lead, 500, self.noise_data_mat)
        return ecg[:, None]
    
    def impute(self, ecg: np.ndarray):
        ecg[np.isnan(ecg)] = 0
        return ecg

    def normalize(self, ecg: np.ndarray):
        return (ecg - ecg.mean()) / (ecg.std() + 1e-6)

    # # TODO: Implement more augmentations
    # def augment(self, ecg: np.ndarray):
    #     # 50% no augmentation
    #     if random.random() < 0.5:
    #         ecg = ecg + np.random.normal(0, 0.01, ecg.shape)
    #     return ecg

    # Assumes uniform length
    def __getitem__(self, idx):
        start_idx = self.cum_seq_len[idx - 1] if idx > 0 else 0
        end_idx = self.cum_seq_len[idx]
        if self.multichannel:
            if self.randomize_leads:
                k_a = random.randint(1, len(self.leads))
                k_b = random.randint(1, len(self.leads))
            else:
                k_a = len(self.leads)
                k_b = len(self.leads)
        else:
            k_a = 1
            k_b = 1
        leads_a = random.sample(self.leads, k_a)
        leads_b = random.sample(self.leads, k_b)
        ecgs_a = [self._process_ecg(lead, start_idx, end_idx) for lead in leads_a]
        ecgs_b = [self._process_ecg(lead, start_idx, end_idx) for lead in leads_b]
        
        ecgs_a = np.stack(ecgs_a, axis=0)
        if self.pad_to_max and k_a < len(self.leads):
            ecgs_a = np.concatenate([ecgs_a, np.zeros((len(self.leads) - k_a, ecgs_a.shape[1], 1))], axis=0)
        
        ecgs_b = np.stack(ecgs_b, axis=0)
        if self.pad_to_max and k_b < len(self.leads):
            ecgs_b = np.concatenate([ecgs_b, np.zeros((len(self.leads) - k_b, ecgs_b.shape[1], 1))], axis=0)

        input_mask = torch.ones(2, len(self.leads) if self.pad_to_max else max(k_a, k_b), dtype=torch.bool)
        input_mask[0, :k_a] = False
        input_mask[1, :k_b] = False
        ecgs = np.concatenate([ecgs_a, ecgs_b], axis=0)
        return torch.from_numpy(ecgs).float(), input_mask, [leads_a, leads_b]


def collate_fn(batch):
    batch, input_mask, leads = zip(*batch)
    all_leads = [example_leads for sublist in leads for example_leads in sublist]
    input_mask = torch.concatenate(input_mask, dim=0)
    batch = torch.concatenate(batch, dim=0)

    N = input_mask.shape[0]
    labels = torch.zeros(N, dtype=torch.long)
    idxs = torch.arange(N, dtype=torch.long)
    labels[idxs] = idxs - (2 * (idxs % 2) - 1)
    return batch, labels, input_mask, all_leads


class EGCDatamodule(L.LightningDataModule):
    def __init__(self, base_path: str, batch_size: int, num_workers: int, multichannel: bool = False, randomize_leads: bool = False):
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.multichannel = multichannel
        self.randomize_leads = randomize_leads
        # self.train_dataset = ECGPretrainDataset(os.path.join(self.base_path, "train_data.hdf5"), multichannel=self.multichannel)
        self.train_dataset = ECGPretrainDataset(os.path.join(self.base_path, "data.hdf5"), multichannel=self.multichannel, randomize_leads=self.randomize_leads)
        self.val_dataset = ECGPretrainDataset(os.path.join(self.base_path, "val_data.hdf5"), multichannel=self.multichannel, randomize_leads=self.randomize_leads)
        self.test_dataset = ECGPretrainDataset(os.path.join(self.base_path, "test_data.hdf5"), multichannel=self.multichannel, randomize_leads=self.randomize_leads)
        
        
        self.kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "multiprocessing_context": "spawn" if self.num_workers > 0 else None,
            "collate_fn": collate_fn,
            'persistent_workers': self.num_workers > 0,
            'pin_memory': True,
            'prefetch_factor': 2,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.kwargs, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, **self.kwargs)
    
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, **self.kwargs)
