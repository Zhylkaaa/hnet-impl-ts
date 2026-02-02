import os
import random 
import pickle

import h5py
import numpy as np
import wfdb
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import ast

import lightning as L

import augmentations


class ECGPretrainDataset(Dataset):
    def __init__(self, data_path: str, multichannel: bool = False, randomize_leads: bool = False, mask_probability: float = 0.2, mask_range: list[float] = (0.1, 0.3), subsample_sequence: int = None):
        self.data_path = data_path
        self.data = h5py.File(data_path, "r")
        self.leads = [k for k in self.data.keys() if k not in ['cum_seq_len', 'date', 'patient_id', 'study_id']]
        self.cum_seq_len = self.data["cum_seq_len"][()]
        self.multichannel = multichannel
        self.randomize_leads = randomize_leads
        self.pad_to_max = multichannel
        self.noise_data_mat = sio.loadmat('DATA_noises_real.mat')
        self.mask_probability = mask_probability
        self.mask_range = mask_range
        self.subsample_sequence = subsample_sequence

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
        if self.subsample_sequence:
            ecg = ecg[:self.subsample_sequence]
        ecg = self.impute(ecg)
        ecg = augmentations.ecg_positive_augmentation(ecg, lead, 500, self.noise_data_mat, p=0)
        ecg = self.normalize(ecg)
        if random.random() < self.mask_probability:
            ecg_masked, noise_mask = augmentations.zero_masking(ecg, mask_ratio=random.uniform(*self.mask_range))
            noise_mask[noise_mask == 1] = ecg[noise_mask == 1]
            ecg = ecg_masked
        else:
            noise_mask = np.zeros_like(ecg)
        return ecg[:, None], noise_mask[:, None]
    
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
        ecgs_a, noise_masks_a = zip(*[self._process_ecg(lead, start_idx, end_idx) for lead in leads_a])
        ecgs_b, noise_masks_b = zip(*[self._process_ecg(lead, start_idx, end_idx) for lead in leads_b])
        
        ecgs_a = np.stack(ecgs_a, axis=0)
        if self.pad_to_max and k_a < len(self.leads):
            ecgs_a = np.concatenate([ecgs_a, np.zeros((len(self.leads) - k_a, ecgs_a.shape[1], 1))], axis=0)
            noise_masks_a = np.concatenate([noise_masks_a, np.zeros((len(self.leads) - k_a, noise_masks_a.shape[1], 1))], axis=0)
        ecgs_b = np.stack(ecgs_b, axis=0)
        if self.pad_to_max and k_b < len(self.leads):
            ecgs_b = np.concatenate([ecgs_b, np.zeros((len(self.leads) - k_b, ecgs_b.shape[1], 1))], axis=0)
            noise_masks_b = np.concatenate([noise_masks_b, np.zeros((len(self.leads) - k_b, noise_masks_b.shape[1], 1))], axis=0)
        input_mask = torch.ones(2, len(self.leads) if self.pad_to_max else max(k_a, k_b), dtype=torch.bool)
        input_mask[0, :k_a] = False
        input_mask[1, :k_b] = False
        ecgs = np.concatenate([ecgs_a, ecgs_b], axis=0)
        noise_masks = np.concatenate([noise_masks_a, noise_masks_b], axis=0)
        return torch.from_numpy(ecgs).float(), input_mask, [leads_a, leads_b], torch.from_numpy(noise_masks).float()


def collate_fn(batch):
    batch, input_mask, leads, noise_masks = zip(*batch)
    all_leads = [example_leads for sublist in leads for example_leads in sublist]
    input_mask = torch.concatenate(input_mask, dim=0)
    batch = torch.concatenate(batch, dim=0)
    noise_masks = torch.concatenate(noise_masks, dim=0)
    N = input_mask.shape[0]
    labels = torch.zeros(N, dtype=torch.long)
    idxs = torch.arange(N, dtype=torch.long)
    labels[idxs] = idxs - (2 * (idxs % 2) - 1)
    return batch, labels, input_mask, all_leads, noise_masks


class EGCDatamodule(L.LightningDataModule):
    def __init__(
        self, base_path: str, batch_size: int, num_workers: int, multichannel: bool = False, 
        randomize_leads: bool = False, mask_probability: float = 0.2, mask_range: list[float] = (0.1, 0.3),
        subsample_sequence: int = None
    ):
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.multichannel = multichannel
        self.randomize_leads = randomize_leads
        # self.train_dataset = ECGPretrainDataset(os.path.join(self.base_path, "train_data.hdf5"), multichannel=self.multichannel)
        self.train_dataset = ECGPretrainDataset(
            os.path.join(self.base_path, "data.hdf5"), 
            multichannel=self.multichannel, 
            randomize_leads=self.randomize_leads, 
            mask_probability=mask_probability, 
            mask_range=mask_range, 
            subsample_sequence=subsample_sequence
        )
        # self.val_dataset = ECGPretrainDataset(os.path.join(self.base_path, "val_data.hdf5"), multichannel=self.multichannel, randomize_leads=self.randomize_leads, mask_probability=mask_probability, mask_range=mask_range)
        # self.test_dataset = ECGPretrainDataset(os.path.join(self.base_path, "test_data.hdf5"), multichannel=self.multichannel, randomize_leads=self.randomize_leads, mask_probability=mask_probability, mask_range=mask_range)
        
        
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


def load_form_split(base_path, split_name, use_lead='II'):
    data = pd.read_csv(os.path.join(base_path, f"ptbxl_form_{split_name}.csv"))
    y = data.iloc[:, -19:].values.astype(int)
    filenames = data.filename_hr
    X = []
    for file in filenames:
        signal, meta = wfdb.rdsamp(os.path.join(base_path, file))
        if use_lead:
            lead_id = meta['sig_name'].index(use_lead)
            X.append(signal[:, lead_id:lead_id+1].T)
        else:
            X.append(signal.T)
    return np.array(X), y

def load_rhythm_split(base_path, split_name, use_lead='II'):
    data = pd.read_csv(os.path.join(base_path, f"ptbxl_rhythm_{split_name}.csv"))
    y = data.iloc[:, -12:].values.astype(int)
    filenames = data.filename_hr
    X = []
    for file in filenames:
        signal, meta = wfdb.rdsamp(os.path.join(base_path, file))
        if use_lead:
            lead_id = meta['sig_name'].index(use_lead)
            X.append(signal[:, lead_id:lead_id+1].T)
        else:
            X.append(signal.T)
    return np.array(X), y

def load_diagnostic_split(base_path, split_name, use_lead='II'):
    data = pd.read_csv(os.path.join(base_path, f"ptbxl_diagnostic_{split_name}.csv"))
    y = data.iloc[:, -44:].values.astype(int)
    filenames = data.filename_hr
    X = []
    for file in filenames:
        signal, meta = wfdb.rdsamp(os.path.join(base_path, file))
        if use_lead:
            lead_id = meta['sig_name'].index(use_lead)
            X.append(signal[:, lead_id:lead_id+1].T)
        else:
            X.append(signal.T)
    return np.array(X), y

def load_subdiagnostic_split(base_path, split_name, use_lead='II'):
    data = pd.read_csv(os.path.join(base_path, f"ptbxl_subdiagnostic_{split_name}.csv"))
    y = data.iloc[:, -23:].values.astype(int)
    filenames = data.filename_hr
    X = []
    for file in filenames:
        signal, meta = wfdb.rdsamp(os.path.join(base_path, file))
        if use_lead:
            lead_id = meta['sig_name'].index(use_lead)
            X.append(signal[:, lead_id:lead_id+1].T)
        else:
            X.append(signal.T)
    return np.array(X), y

def load_supdiagnostic_split(base_path, split_name, use_lead='II'):
    data = pd.read_csv(os.path.join(base_path, f"ptbxl_supdiagnostic_{split_name}.csv"))
    y = data.iloc[:, -5:].values.astype(int)
    filenames = data.filename_hr
    X = []
    for file in filenames:
        signal, meta = wfdb.rdsamp(os.path.join(base_path, file))
        if use_lead:
            lead_id = meta['sig_name'].index(use_lead)
            X.append(signal[:, lead_id:lead_id+1].T)
        else:
            X.append(signal.T)
    return np.array(X), y

# Define all subsets and their loading functions
SUBSETS = {
    'form': load_form_split,
    'rhythm': load_rhythm_split,
    'diagnostic': load_diagnostic_split,
    'subdiagnostic': load_subdiagnostic_split,
    'supdiagnostic': load_supdiagnostic_split
}


def load_raw_data_ptbxl(df, sampling_rate, base_path, use_lead='II'):
    os.makedirs("dataset/ptbxl", exist_ok=True)
    if sampling_rate == 100:
        if os.path.exists(f"dataset/ptbxl/raw100_{use_lead}.npy"):
            data = np.load(f"dataset/ptbxl/raw100_{use_lead}.npy", allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(base_path, f)) for f in df.filename_lr]
            meta = data[0][1]
            if use_lead:
                lead_id = meta['sig_name'].index(use_lead)
                data = [signal[:, lead_id:lead_id+1].T for signal, _ in data]
            else:
                data = [signal.T for signal, _ in data]
            data = np.array(data)
            pickle.dump(data, open(f"dataset/ptbxl/raw100_{use_lead}.npy", "wb"), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(f"dataset/ptbxl/raw500_{use_lead}.npy"):
            data = np.load(f"dataset/ptbxl/raw500_{use_lead}.npy", allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(base_path, f)) for f in df.filename_hr]
            meta = data[0][1]
            if use_lead:
                lead_id = meta['sig_name'].index(use_lead)
                data = [signal[:, lead_id:lead_id+1].T for signal, _ in data]
            else:
                data = [signal.T for signal, _ in data]
            data = np.array(data)
            pickle.dump(data, open(f"dataset/ptbxl/raw500_{use_lead}.npy", "wb"), protocol=4)
    return data


def compute_label_aggregations(df, base_path, ctype):
    df["scp_codes_len"] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(os.path.join(base_path, "scp_statements.csv"), index_col=0)

    if ctype in ["diagnostic", "subdiagnostic", "superdiagnostic"]:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == "diagnostic":
            df["diagnostic"] = df.scp_codes.apply(aggregate_all_diagnostic)
            df["diagnostic_len"] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == "subdiagnostic":
            df["subdiagnostic"] = df.scp_codes.apply(aggregate_subdiagnostic)
            df["subdiagnostic_len"] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == "superdiagnostic":
            df["superdiagnostic"] = df.scp_codes.apply(aggregate_diagnostic)
            df["superdiagnostic_len"] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == "form":
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        df["form"] = df.scp_codes.apply(aggregate_form)
        df["form_len"] = df.form.apply(lambda x: len(x))
    elif ctype == "rhythm":
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        df["rhythm"] = df.scp_codes.apply(aggregate_rhythm)
        df["rhythm_len"] = df.rhythm.apply(lambda x: len(x))
    elif ctype == "all":
        df["all_scp"] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def load_dataset(base_path, sampling_rate, use_lead='II'):
    # load and convert annotation data
    Y = pd.read_csv(os.path.join(base_path, "ptbxl_database.csv"), index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, base_path, use_lead)

    return X, Y


def select_data(XX, YY, ctype, min_samples):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == "diagnostic":
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == "subdiagnostic":
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["subdiagnostic_len"] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == "superdiagnostic":
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["superdiagnostic_len"] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == "form":
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["form_len"] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == "rhythm":
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["rhythm_len"] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == "all":
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["all_scp_len"] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # # save LabelBinarizer
    # with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
    #     pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb


class PTBXLEGCDataset(Dataset):
    def __init__(self, base_path: str, subset: str, split: str, use_lead: str = 'II', subsample_sequence: int = None):
        self.base_path = base_path
        self.subset = subset
        self.split = split
        self.use_lead = use_lead
        self.subsample_sequence = subsample_sequence
        #print("Loading PTB-XL ECG dataset...")
        # self.data, self.labels = SUBSETS[self.subset](self.base_path, self.split, self.use_lead if self.use_lead != 'all' else None)
        # #print("Loaded PTB-XL ECG dataset")
        # self._filter_data()
        self.data, self.raw_labels = load_dataset(
            self.base_path, 500, use_lead=self.use_lead if self.use_lead != 'all' else None
        )
        self.labels = compute_label_aggregations(
            self.raw_labels, self.base_path, self.subset
        )
        self.data, df, self.labels, _ = select_data(
            self.data, self.labels, self.subset, 0
        )

        if self.split == 'train':
            self.data = self.data[df.strat_fold <= 8]
            self.labels = self.labels[df.strat_fold <= 8]
        elif self.split == 'val':
            self.data = self.data[df.strat_fold == 9]
            self.labels = self.labels[df.strat_fold == 9]
        elif self.split == 'test':
            self.data = self.data[df.strat_fold == 10]
            self.labels = self.labels[df.strat_fold == 10]

        if self.subsample_sequence:
            self.data = self.data[:, :self.subsample_sequence]
            
    def _filter_data(self):
        # apparently there are examples with no label at all
        mask = self.labels.sum(axis=1) > 0
        self.data = self.data[mask]
        self.labels = self.labels[mask]

    def __len__(self):
        return len(self.data)
    
    def impute(self, ecg: np.ndarray):
        ecg[np.isnan(ecg)] = 0
        return ecg

    def normalize(self, ecg: np.ndarray):
        return (ecg - ecg.mean(axis=1, keepdims=True)) / (ecg.std(axis=1, keepdims=True) + 1e-6)

    # Assumes uniform length
    def __getitem__(self, idx):
        ecg = self.data[idx]
        ecg = self.impute(ecg)
        ecg = self.normalize(ecg)
        return torch.from_numpy(ecg).float().unsqueeze(-1), torch.from_numpy(self.labels[idx]).float()


def collate_ptbxl_fn(batch):
    batch, labels = zip(*batch)
    batch = torch.stack(batch, dim=0)
    labels = torch.stack(labels, dim=0)
    return batch, labels


class PTBXLEGCDatamodule(L.LightningDataModule):
    def __init__(self, base_path: str, subset: str, batch_size: int, num_workers: int, use_lead: str = 'II', subsample_sequence: int = None):
        super().__init__()
        self.base_path = base_path
        self.subset = subset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_lead = use_lead
        self.train_dataset = PTBXLEGCDataset(self.base_path, subset=subset, split='train', use_lead=self.use_lead, subsample_sequence=subsample_sequence)
        self.val_dataset = PTBXLEGCDataset(self.base_path, subset=subset, split='val', use_lead=self.use_lead, subsample_sequence=subsample_sequence)
        self.test_dataset = PTBXLEGCDataset(self.base_path, subset=subset, split='test', use_lead=self.use_lead, subsample_sequence=subsample_sequence)
        
        
        self.kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "multiprocessing_context": "spawn" if self.num_workers > 0 else None,
            "collate_fn": collate_ptbxl_fn,
            'persistent_workers': self.num_workers > 0,
            'pin_memory': True,
            'prefetch_factor': 2,
        }
    
    @property
    def num_labels(self):
        return self.train_dataset.labels.shape[1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.kwargs)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.kwargs)
