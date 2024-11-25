import datasets
from datasets import load_dataset
from typing import Optional



class DataLoader:
    def __init__(self, dataset_name : str, dataset_seed : int, dataset_samples : int, test_size: float):
        self.dataset_name : str = dataset_name
        self.dataset_seed : int = dataset_seed
        self.dataset_samples : int = dataset_samples
        self.test_size : float = test_size
    def load_dataset_samples(self):
             
            dataset = load_dataset(self.dataset_name, split="all")
            dataset =  dataset.shuffle(seed=self.dataset_seed).select(range(self.dataset_samples)) 
            dataset = dataset.train_test_split(seed = self.dataset_seed, test_size = self.test_size)
            return dataset
    
