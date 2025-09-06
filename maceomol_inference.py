import os
import sys
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from ase import Atoms
from mace.calculators import mace_omol
from typing import Dict
import time

class MACEOMOL_Inference:
    def __init__(self, model_path: str, h5_path: str, ds_name: str):
        self.h5_path = h5_path
        self.ds_name = ds_name
        self.data_dict = self.extract_input_from_h5()
        self.calc = mace_omol(model=model_path, device='cuda')
        self.calc.energy_units_to_eV

    def extract_input_from_h5(self, chunk_size: int = 1000) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        data_dict = {}
        with h5py.File(self.h5_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            num_keys = len(keys)

            for start in tqdm(range(0, num_keys, chunk_size), desc="Extracting HDF5 data in chunks"):
                end = min(start + chunk_size, num_keys)
                for key in keys[start:end]:
                    coord = h5_file[key]['coord'][:]
                    numbers = h5_file[key]['numbers'][:]
                    charge = h5_file[key]['charge'][:]
                    charge0 = h5_file[key]['charge0'][:]
                    charge1 = h5_file[key]['charge1'][:]
                    geom_id = h5_file[key]['geom_id'][:]
                    natoms0 = h5_file[key]['natoms0'][:]
                    natoms1 = h5_file[key]['natoms1'][:]
                    energy_int = h5_file[key]['energy_int'][:]

                    data_dict[key] = {}

                    unique_types = list(set(zip(natoms0, natoms1)))

                    for n0, n1 in unique_types:
                        type_key = f"({n0},{n1})"
                        
                        indices = [i for i, (x0, x1) in enumerate(zip(natoms0, natoms1)) if x0 == n0 and x1 == n1]
                        selected_coord = coord[indices]
                        selected_numbers = numbers[indices]
                        selected_charge = charge[indices]
                        selected_charge0 = charge0[indices]
                        selected_charge1 = charge1[indices]
                        selected_geom_id = geom_id[indices]
                        selected_energy_int = energy_int[indices]

                        data_dict[key][type_key] = {
                            'dimer': {
                                'coord': selected_coord,
                                'numbers': selected_numbers,
                                'charge': selected_charge,
                                'geom_id': selected_geom_id,
                                'ref_energy_int': selected_energy_int
                            },
                            'mol0': {
                                'coord': selected_coord[:, :n0, :],
                                'numbers': selected_numbers[:, :n0],
                                'charge': selected_charge0,
                            },
                            'mol1': {
                                'coord': selected_coord[:, n0:n0+n1, :],
                                'numbers': selected_numbers[:, n0:n0+n1],
                                'charge': selected_charge1,
                            }
                        }
        return data_dict

    def calculate_energies(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        energies = []
        for coord, numbers in zip(data['coord'], data['numbers']):
            mol = self.create_molecule(coord, numbers)
            ## Set charge and spin for the system
            #mol.info["charge"] = 1.0      # +1 charge
            #mol.info["spin"] = 1.0        # spin multiplicity
            mol.calc = self.calc
            energy = mol.get_potential_energy()
            energies.append(energy)
        return np.array(energies)

    def create_molecule(self, coord, numbers) -> Atoms:
        return Atoms(numbers=numbers, positions=coord)

    def run_inference(self) -> Dict[str, pd.DataFrame]:
        start_time = time.time()
        interaction_energies = []
        for group_name, group_data in tqdm(self.data_dict.items(), desc="Running MACE-OMOL model inference"):
            for dimer_type, data in group_data.items():
                e_dim = self.calculate_energies(data['dimer'])
                e_mol0 = self.calculate_energies(data['mol0'])
                e_mol1 = self.calculate_energies(data['mol1'])
                
                if len(e_dim) != len(e_mol0) or len(e_dim) != len(e_mol1):
                    raise ValueError(f"Energy arrays have mismatched shapes: {e_dim.shape}, {e_mol0.shape}, {e_mol1.shape}")

                interaction_energy = (e_dim - e_mol0 - e_mol1) * 23.0609

                df = pd.DataFrame({
                    'geom_id': data['dimer']['geom_id'],
                    'dimer_type': dimer_type,
                    'pred_dimer_energy': e_dim,
                    'pred_mol0_energy': e_mol0,
                    'pred_mol1_energy': e_mol1,
                    'pred_energy_int': interaction_energy,
                    'ref_energy_int': data['dimer']['ref_energy_int']
                })
                df['group'] = group_name
                interaction_energies.append(df)
        
        end_time = time.time()
        print(f"MACE-OMOL inference time for {self.ds_name}: {end_time - start_time:.2f} seconds")
        return pd.concat(interaction_energies, ignore_index=True)

    def save_results(self, final_df: pd.DataFrame):
        os.makedirs("outputs", exist_ok=True)
        final_df.to_csv(f'outputs/{self.__class__.__name__}_{self.ds_name}_intE.csv', index=False)

if __name__ == "__main__":
    model_path, h5_path, ds_name = sys.argv[1:4]

    maceomol_inference = MACEOMOL_Inference(model_path, h5_path, ds_name)
    interaction_energies = maceomol_inference.run_inference()
    maceomol_inference.save_results(interaction_energies)
