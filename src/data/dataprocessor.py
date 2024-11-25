from datasets import load_dataset
import datasets
from get_logging import logger_object  # Adjust the module path accordingly

logger = logger_object()

class DatasetProcessor:
    def __init__(self, dataset_template_column_a : str, dataset_template_column_b : str, dataset_tokenize : bool, compute_n_proc : int):
        self.dataset_template_column_a : str = dataset_template_column_a
        self.dataset_template_column_b : str = dataset_template_column_b
        self.dataset_tokenize : bool = dataset_tokenize
        self.compute_n_proc : int = compute_n_proc


    def apply_chat_template(self,dataset, tokenizer, config):
        
        logger.debug('applying the chat template to dataset')
        def format_chat_template(row):


            row_json = [
                {"role" : "user", "content" : row[self.dataset_template_column_a]},
                {"role" : "assistant", "content" : row[self.dataset_template_column_b]}
            ]
            row["text"] = tokenizer.apply_chat_template(row_json, tokenize = self.dataset_tokenize)
            return row

        dataset = dataset.map(format_chat_template,num_proc=self.compute_n_proc,)
        
        return dataset



class DatasetParitioner:
    def __init__(self, dataset_test_size : float):
        self.dataset_test_size : float = dataset_test_size


    def apply_train_test_split(self, dataset):
        logger.debug('Splitting dataset into train and test')
        dataset = dataset.train_test_split(test_size=self.dataset_test_size)
        return dataset
    

    