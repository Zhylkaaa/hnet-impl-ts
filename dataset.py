import os
import random 

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L


class ECGPretrainDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = h5py.File(data_path, "r")
        self.leads = [k for k in self.data.keys() if k not in ['cum_seq_len', 'date', 'patient_id', 'study_id']]
        self.cum_seq_len = self.data["cum_seq_len"][()]

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
        ecg = self.augment(ecg)
        return ecg
    
    def impute(self, ecg: np.ndarray):
        ecg[np.isnan(ecg)] = 0
        return ecg

    def normalize(self, ecg: np.ndarray):
        return (ecg - ecg.mean()) / (ecg.std() + 1e-6)

    # TODO: Implement more augmentations
    def augment(self, ecg: np.ndarray):
        # 50% no augmentation
        if random.random() < 0.5:
            ecg = ecg + np.random.normal(0, 0.01, ecg.shape)
        return ecg

    # Assumes uniform length
    def __getitem__(self, idx):
        start_idx = self.cum_seq_len[idx - 1] if idx > 0 else 0
        end_idx = self.cum_seq_len[idx]
        
        lead_a = random.choice(self.leads)
        lead_b = random.choice(self.leads)

        ecg_a = self._process_ecg(lead_a, start_idx, end_idx)
        ecg_b = self._process_ecg(lead_b, start_idx, end_idx)
        ecgs = np.stack([ecg_a, ecg_b], axis=0)

        return torch.from_numpy(ecgs).float()


def collate_fn(batch):
    batch = torch.concatenate(batch, dim=0)
    labels = torch.zeros(batch.shape[0], dtype=torch.long)
    idxs = torch.arange(batch.shape[0], dtype=torch.long)
    labels[idxs] = idxs - (2 * (idxs % 2) - 1)
    return batch, labels


# class EGCDatamodule(L.LightningDataModule):
class EGCDatamodule:
    def __init__(self, base_path: str, batch_size: int, num_workers: int):
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = ECGPretrainDataset(os.path.join(self.base_path, "train_data.hdf5"))
        self.val_dataset = ECGPretrainDataset(os.path.join(self.base_path, "val_data.hdf5"))
        self.test_dataset = ECGPretrainDataset(os.path.join(self.base_path, "test_data.hdf5"))
        
        self.kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "multiprocessing_context": "spawn" if self.num_workers > 0 else None,
            "collate_fn": collate_fn,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.kwargs)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.kwargs)