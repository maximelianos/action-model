import h5py
import numpy as np
from pathlib import Path

class LossLogger:
    def __init__(self, path: str, overwrite: bool = False):
        self.path = Path(path)
        self.data = {}  # In-memory storage: {key: {'t': [times], 'values': [values]}}
        
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if file exists and not overwriting
        if self.path.exists() and not overwrite:
            self._load_from_file()
        elif overwrite and self.path.exists():
            self.path.unlink()
    
    def _load_from_file(self):
        """Load existing data from h5 file into memory"""
        with h5py.File(self.path, 'r') as f:
            for key in f.keys():
                times = f[key]['t'][:]
                values = f[key]['values'][:]
                self.data[key] = {'t': times.tolist(), 'values': values.tolist()}
    
    def append(self, entry):
        """
        Append a value to history.
        
        entry = {
            "t": 1,
            "train": 0.1,
            "val": 0.2
        }
        
        Dictionary fields:
            t (int): time step,
            train (float): loss value,
            val (float): another loss value
        
        Arguments:
            entry (dict): contains time step t and loss values at this time step

        Returns:
            None
        """
        if 't' not in entry:
            raise ValueError("Entry must contain 't' (timestep)")
        
        timestep = entry['t']
        
        # Store in memory
        for key, value in entry.items():
            if key == 't':
                continue
            
            if key not in self.data:
                self.data[key] = {'t': [], 'values': []}
            
            self.data[key]['t'].append(timestep)
            self.data[key]['values'].append(value)
    
    def save(self):
        """Save in-memory data to h5 file"""
        with h5py.File(self.path, 'w') as f:
            for key, data in self.data.items():
                group = f.create_group(key)
                group.create_dataset('t', data=np.array(data['t']))
                group.create_dataset('values', data=np.array(data['values']))
    
    def get(self, key: str):
        """
        Get time array and loss values array.
        
        Arguments:
            key (str): name of the loss
        
        Returns:
            tuple[np.ndarray, np.ndarray]
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in loss data")
        
        times = np.array(self.data[key]['t'])
        values = np.array(self.data[key]['values'])
        
        return times, values

if __name__ == "__main__":
    if Path("loss.h5").exists():
        Path("loss.h5").unlink()

    loss = LossLogger(path="loss.h5", overwrite=False) # load content from disk, if no overwrite
    loss.append({
        "t": 0,
        "train": 0.125,
        "val": 0.18
    })
    loss.append({
        "t": 1,
        "train": 0.1,
    })
    loss.append({
        "t": 2,
        "val": 0.2
    })
    t, y = loss.get("train") # both are numpy arrays of same length - the time array and loss values array
    assert np.allclose(t, [0, 1])
    assert np.allclose(y, [0.125, 0.1])
    
    t, y = loss.get("val") # both are numpy arrays of same length - the time array and loss values array
    assert np.allclose(t, [0, 2])
    assert np.allclose(y, [0.18, 0.2])
    
    for i in range(10000):
        loss.append({
            "t": i,
            "train": 1/(i+1),
        })
    
    # Save to file
    loss.save()