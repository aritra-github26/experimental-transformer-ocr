"""HDF5 dataset merger for handwriting recognition datasets

To run this script, use the following command:
cd src
python data/merger.py --input_dir data/ --output_file data/merged/merged.hdf5
"""

import os
import h5py
import argparse
from tqdm import tqdm

def merge_hdf5_files(input_dir, output_file):
    """
    Merge multiple HDF5 files into a single file while preserving all data.
    
    Args:
        input_dir (str): Directory containing HDF5 files to merge
        output_file (str): Path to output merged HDF5 file
    """
    # Get list of HDF5 files in input directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5') or f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    print(f"Found {len(h5_files)} HDF5 files to merge")
    
    def print_h5_structure(h5_file, name):
        """Print structure of HDF5 file"""
        print(f"\nStructure of {name}:")
        def print_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        with h5py.File(h5_file, 'r') as f:
            f.visititems(print_group)

    # First analyze input files
    for file_name in h5_files:
        file_path = os.path.join(input_dir, file_name)
        print_h5_structure(file_path, file_name)

    # First collect all unique dataset paths and their shapes
    dataset_info = {}
    for file_name in h5_files:
        file_path = os.path.join(input_dir, file_name)
        with h5py.File(file_path, 'r') as f:
            def collect_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if name not in dataset_info:
                        dataset_info[name] = {
                            'shapes': [obj.shape],
                            'dtype': obj.dtype,
                            'total_size': obj.shape[0]
                        }
                    else:
                        # Verify shape compatibility
                        if dataset_info[name]['shapes'][0][1:] != obj.shape[1:]:
                            raise ValueError(f"Incompatible shapes for {name}: {dataset_info[name]['shapes'][0]} vs {obj.shape}")
                        dataset_info[name]['shapes'].append(obj.shape)
                        dataset_info[name]['total_size'] += obj.shape[0]
            f.visititems(collect_info)

    # Create merged file with pre-allocated datasets
    with h5py.File(output_file, 'w') as h5_out:
        # Create all groups first
        for path in dataset_info.keys():
            group_path = os.path.dirname(path)
            if group_path and group_path not in h5_out:
                h5_out.create_group(group_path)
        
        # Create datasets with total size
        for path, info in dataset_info.items():
            new_shape = (info['total_size'],) + info['shapes'][0][1:]
            h5_out.create_dataset(
                path,
                shape=new_shape,
                dtype=info['dtype'],
                chunks=True,
                compression="gzip"
            )
        
        # Now copy data into pre-allocated space
        offset = {path: 0 for path in dataset_info.keys()}
        for file_name in tqdm(h5_files, desc="Merging files"):
            file_path = os.path.join(input_dir, file_name)
            with h5py.File(file_path, 'r') as h5_in:
                for path in dataset_info.keys():
                    if path in h5_in:
                        data = h5_in[path][()]
                        h5_out[path][offset[path]:offset[path]+data.shape[0]] = data
                        offset[path] += data.shape[0]

    print(f"Successfully merged datasets into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge HDF5 datasets')
    parser.add_argument('--input_dir', required=True, help='Directory containing HDF5 files to merge')
    parser.add_argument('--output_file', required=True, help='Output HDF5 file path')
    args = parser.parse_args()
    
    merge_hdf5_files(args.input_dir, args.output_file)
