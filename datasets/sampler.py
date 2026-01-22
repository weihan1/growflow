from torch.utils.data import Sampler
import torch
import random

class NeuralODEDataSampler_MixedInit(Sampler):
    """
    Mixed initial params sampler for neural ode, generate contiguous samples. 
    Samples multiple non-overlapping initial conditions each with seq_length timesteps.
    """
    def __init__(self, data_source, my_indices, seq_length, num_initial_conditions):
        self.data_source = data_source
        if isinstance(my_indices, torch.Tensor):
            self.indices = my_indices.tolist()
        else:
            self.indices = my_indices  
        
        self.total_timesteps = len(self.indices)
        self.seq_length = seq_length
        self.num_initial_conditions = num_initial_conditions
        
    def __iter__(self):
        while True:
            # Sample multiple non-overlapping starting points
            valid_starts = list(range(0, self.total_timesteps - self.seq_length + 1))
            
            # Randomly sample starting points without replacement
            start_indices = torch.randperm(len(valid_starts))[:self.num_initial_conditions].tolist()
            start_points = [valid_starts[i] for i in sorted(start_indices)]
            
            # Generate all sequences and flatten into single list
            all_timesteps = []
            for start_t in start_points:
                sequence = [self.indices[i] for i in range(start_t, start_t + self.seq_length)]
                all_timesteps.extend(sequence)
            
            # Yield flattened list of timesteps
            yield from all_timesteps
            
    def __len__(self):
        return 1000000

class NeuralODEDataSampler(Sampler):
    """
    Custom dataloader sampler 
    Convert indices to python indices
    """
    def __init__(self, data_source, my_indices, shuffle=False):
        self.data_source = data_source
        if isinstance(my_indices, torch.Tensor):
            self.indices = my_indices.tolist()
                
        else:
            self.indices = my_indices  

        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

class InfiniteNeuralODEDataSampler(Sampler):
    """
    Infinite sampler that endlessly yields indices from the provided set
    If shuffle is set to False, then if my list is [1,2,3,4,5,6,7] with batch size of 2,
    it will always sample [1,2], [3,4], [5,6], [7,1] (wrapping around)
    """
    def __init__(self, data_source, my_indices, shuffle=False):
        self.data_source = data_source
        if isinstance(my_indices, torch.Tensor):
            self.indices = my_indices.tolist()
        else:
            self.indices = my_indices
        self.shuffle = shuffle
        
    def __iter__(self):
        while True:  # Create an infinite iterator
            if self.shuffle:
                # Create a copy of indices to avoid modifying the original
                indices = self.indices.copy() if hasattr(self.indices, 'copy') else self.indices[:]
                random.shuffle(indices)
                yield from indices
            else:
                yield from self.indices
    
    def __len__(self):
        # This isn't strictly accurate for an infinite sampler,
        # but is needed for some PyTorch internals
        return 1000000  # A large number

