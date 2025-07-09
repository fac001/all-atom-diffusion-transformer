
"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from typing import Callable, List, Optional

import periodictable
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)


class ZINC(InMemoryDataset):

    def __init__(
        self,
        root: str,
        pickle_path: str, 
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.pickle_path = pickle_path
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [file for file in os.listdir(self.pickle_path) if file.endswith('.pkl')]

    @property
    def processed_file_names(self) -> List[str]:
        return ['zinc.pt']

    # def download(self) -> None:
    #     pass

    def process(self) -> None:

        data_list = []

        for file in os.listdir(self.pickle_path):

            if not file.endswith('.pkl'): continue

            file_path = os.path.join(self.pickle_path, file)
            data_list.extend(self.process_file(file_path))

        self.save(data_list, os.path.join(self.root, "processed/zinc.pt"))

    def process_file(self, file_path):
        df = pd.read_pickle(file_path, compression="gzip")
        data = df.apply(self.process_row, axis=1).to_list()
        return data


    # ZINC adapted for ADiT 
    def process_row(self, row, removeHs=False):

        z = torch.tensor(list(map(lambda x: periodictable.elements.symbol(x).number, row['atom_types']))) #convert atomic symbol into atomic number

        atoms_to_keep = torch.ones_like(z, dtype=torch.bool)
        num_atoms = row['lig_natoms']
        if removeHs:
            atoms_to_keep = z != 1
            num_atoms = atoms_to_keep.sum().item()

        coords = torch.FloatTensor(row['lig_coords'])
        
        return Data(
            id=row['pdbid'],
            atom_types=z[atoms_to_keep],
            pos=coords[atoms_to_keep],
            frac_coords=torch.zeros_like(coords[atoms_to_keep]),
            cell=torch.zeros((1, 3, 3)),
            lattices=torch.zeros(1, 6),
            lattices_scaled=torch.zeros(1, 6),
            lengths=torch.zeros(1, 3),
            lengths_scaled=torch.zeros(1, 3),
            angles=torch.zeros(1, 3),
            angles_radians=torch.zeros(1, 3),
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
            spacegroup=torch.zeros(1, dtype=torch.long),  # null spacegroup
            token_idx=torch.arange(num_atoms),
            dataset_idx=torch.tensor([1], dtype=torch.long),  # 1 --> indicates non-periodic/molecule
        )

# data = ZINC(root = "./src/data/components", pickle_path= ".")
# print(len(data))
# print(data.get(0))
