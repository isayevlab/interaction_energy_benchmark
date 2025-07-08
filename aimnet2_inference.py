import os
import sys
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import Dict
import time
from torch.amp import autocast              # for mixed precision

class AIMNET2_Inference:
    BATCH_SIZE = 1000                       # adjust this for optimal GPU usage

    def __init__(self, model_path: str, h5_path: str, ds_name: str):
        self.model = torch.jit.load(model_path).cuda()  
        self.h5_path = h5_path
        self.ds_name = ds_name
        self.data_dict = self.extract_input_from_h5()

    def extract_input_from_h5(self, chunk_size: int = 1000) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        data_dict = {}
        with h5py.File(self.h5_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            num_keys = len(keys)

            for start in tqdm(range(0, num_keys, chunk_size), desc="Extracting HDF5 data in chunks"):
                end = min(start + chunk_size, num_keys)
                for key in keys[start:end]:
                    coord = torch.tensor(h5_file[key]['coord'][:], dtype=torch.float16, requires_grad=False).cuda(non_blocking=True)
                    numbers = torch.tensor(h5_file[key]['numbers'][:], requires_grad=False).cuda(non_blocking=True)
                    charge = torch.tensor(h5_file[key]['charge'][:], requires_grad=False).cuda(non_blocking=True)
                    charge0 = torch.tensor(h5_file[key]['charge0'][:], requires_grad=False).cuda(non_blocking=True)
                    charge1 = torch.tensor(h5_file[key]['charge1'][:], requires_grad=False).cuda(non_blocking=True)
                    geom_id = torch.tensor(h5_file[key]['geom_id'][:], requires_grad=False).cuda(non_blocking=True)
                    natoms0 = torch.tensor(h5_file[key]['natoms0'][:], requires_grad=False).cuda(non_blocking=True)
                    natoms1 = torch.tensor(h5_file[key]['natoms1'][:], requires_grad=False).cuda(non_blocking=True)
                    energy_int = torch.tensor(h5_file[key]['energy_int'][:], requires_grad=False).cuda(non_blocking=True)

                    data_dict[key] = {}

                    unique_types = list(set(zip(natoms0.cpu().numpy(), natoms1.cpu().numpy())))

                    for n0, n1 in unique_types:
                        type_key = f"({n0},{n1})"

                        indices = [i for i, (x0, x1) in enumerate(zip(natoms0.cpu().numpy(), natoms1.cpu().numpy())) if x0 == n0 and x1 == n1]

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

    def model_inference(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        
        with torch.no_grad(), autocast(device_type='cuda'):                     
            output = self.model(data)       
        energy = output['energy'].detach().cpu().numpy().flatten() * 23.0609    
        return energy

    def batch_model_inference(self, data_list: Dict[str, torch.Tensor]) -> np.ndarray:
        
        # perform batched inference for faster performance, ensuring async data transfers and mixed precision
        energies = []
        for i in range(0, len(data_list['coord']), self.BATCH_SIZE):
            batch_data = {
                'coord': data_list['coord'][i:i+self.BATCH_SIZE].cuda(non_blocking=True),  
                'numbers': data_list['numbers'][i:i+self.BATCH_SIZE].cuda(non_blocking=True),
                'charge': data_list['charge'][i:i+self.BATCH_SIZE].cuda(non_blocking=True)
            }
            batch_energy = self.model_inference(batch_data)
            energies.append(batch_energy)
        return np.concatenate(energies)

    def run_inference(self) -> Dict[str, pd.DataFrame]:
        torch.cuda.synchronize()                            # ensure all operations are done before timing
        start_time = time.time()
        interaction_energies = []
        for group_name, group_data in tqdm(self.data_dict.items(), desc="Running AIMNet2 model inference"):
            for dimer_type, data in group_data.items():
                e_dim = self.batch_model_inference(data['dimer'])
                e_mol0 = self.batch_model_inference(data['mol0'])
                e_mol1 = self.batch_model_inference(data['mol1'])

                if len(e_dim) != len(e_mol0) or len(e_dim) != len(e_mol1):
                    raise ValueError(f"Energy arrays have mismatched shapes: {e_dim.shape}, {e_mol0.shape}, {e_mol1.shape}")

                interaction_energy = (e_dim - e_mol0 - e_mol1)

                df = pd.DataFrame({
                    'geom_id': data['dimer']['geom_id'].cpu().numpy(),  
                    'dimer_type': dimer_type,
                    'pred_dimer_energy': e_dim,
                    'pred_mol0_energy': e_mol0,
                    'pred_mol1_energy': e_mol1,
                    'pred_energy_int': interaction_energy,
                    'ref_energy_int': data['dimer']['ref_energy_int'].cpu().numpy()
                })
                df['group'] = group_name
                interaction_energies.append(df)

        torch.cuda.synchronize()                            # ensure all GPU tasks are done before stopping the time
        end_time = time.time()

        print(f"AIMNet2 inference time for {self.ds_name}: {end_time - start_time:.2f} seconds")
        return pd.concat(interaction_energies, ignore_index=True)
    
    def save_results(self, final_df: pd.DataFrame):
        os.makedirs("outputs", exist_ok=True)
        final_df.to_csv(f'outputs/{self.__class__.__name__}_{self.ds_name}_intE.csv', index=False)

if __name__ == "__main__":
    
    torch.set_grad_enabled(False)
    model_path, h5_path, ds_name = sys.argv[1:4]

    aimnet2_inference = AIMNET2_Inference(model_path, h5_path, ds_name)
    interaction_energies = aimnet2_inference.run_inference()
    aimnet2_inference.save_results(interaction_energies)
