import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from trl import setup_chat_format
from huggingface_hub import login
from datasets import load_dataset
from src.data.dataloader import  DataLoader
from src.utils.get_logging import logger_object
from sentence_transformers import SentenceTransformer
from src.model.model_artifacts import ModelArtifactLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logger_object()
torch.cuda.empty_cache()

# Load evaluation metrics
rouge = evaluate.load("rouge")
# Load a Sentence-BERT model for cosine similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


class Evaluator:
    def __init__(self, model_path : str, model_type : str):
        self.model_path : str = model_path
        self.model_type :str = model_type
        

    def load_model_artifacts(self):
        model_object = ModelArtifactLoader(self.model_path, self.model_type ) # path to base and finetuned model
        model, tokenizer = model_object.load_artifacts()
        return model, tokenizer

#model, tokenizer = load_model_artifacts(model_path = model_path)
    def generate_inference(self, input_string : str, model : AutoModelForCausalLM, tokenizer : AutoTokenizer, do_sample : bool = True, max_new_tokens : int = 200, temperature : float = 0.0, top_k : int = 20, top_p : float = 0.90, repetition_penalty : float = 1.5):
        messages = [{"role": "user", "content": input_string}]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty = repetition_penalty)
        result = outputs[0]["generated_text"]
        return result.split('>assistant\n')[1]


    def compute_metrics(self, generated: str, reference: str):
        # ROUGE
        rouge_score = rouge.compute(predictions=[generated], references=[reference])
        
        
        # SBERT Cosine Similarity
        gen_emb = sbert_model.encode(generated)
        ref_emb = sbert_model.encode(reference)
        cosine_sim = np.dot(gen_emb, ref_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb))
        
        return {
            "rouge": rouge_score,
            "sbert_cosine_similarity": cosine_sim
        }
    
    def evaluate_pipeline(self, model_type, dataset, model, tokenizer, do_sample, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
        metrics_list = []
        
        total_samples = len(dataset['test'])
        for index in tqdm(range(total_samples)):
            reference = dataset['test'][index]['Doctor']
            generated = self.generate_inference(input_string = dataset['test'][index]['Patient'], model = model, tokenizer = tokenizer, do_sample = do_sample, max_new_tokens = max_new_tokens, temperature = temperature, top_k = top_k, top_p = top_p, repetition_penalty = repetition_penalty)
            metrics = self.compute_metrics(generated = generated, reference = reference)
            metrics_list.append(metrics)

        sbert_cosine_similarity_scores = []
        
        rogue_score = []
        for i in range(len(metrics_list)):
            result = metrics_list[i]
            
            sbert_cosine_similarity_scores.append(result['sbert_cosine_similarity'])
            rogue_score.append(result['rouge']['rouge1'])    

        
        average_rogue_score = np.mean(rogue_score)
        average_sbert_cosine_similarity_scores = np.mean(sbert_cosine_similarity_scores)

        logger.debug(f'{model_type} evaluation results')
        logger.debug(f' average rogue score : {average_rogue_score} | average sbert cosine similarity score : {average_sbert_cosine_similarity_scores}')


        return average_rogue_score, average_sbert_cosine_similarity_scores
    
if __name__ == "__main__":

    finetuned_model_path = '../../../Medical-LLM-Finetuning/finetuned-llama-3-8b-chat-doctor-merged/'
    base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name = "ruslanmv/ai-medical-chatbot"
    max_new_tokens = 200
    temperature = 0.0
    top_k = 20
    top_p = 0.90
    repetition_penalty = 1.5
    do_sample = False

    test_size = 0.01
    dataset_seed = 412
    dataset_samples = 10000
    logger.debug('Loading dataset & Splitting')
    data_object = DataLoader(dataset_name = dataset_name, dataset_seed = dataset_seed, dataset_samples = dataset_samples, test_size = test_size)
    dataset = data_object.load_dataset_samples()

    logger.debug('Evaluating Base Model')
    e_base = Evaluator(model_path = base_model_path, model_type="base")
    base_model, base_tokenizer = e_base.load_model_artifacts()
    average_rogue_score_base, average_sbert_cosine_similarity_scores_base = e_base.evaluate_pipeline(model_type="base", dataset = dataset, model = base_model, tokenizer = base_tokenizer, do_sample = do_sample, max_new_tokens = max_new_tokens, temperature = temperature, top_k = top_k, top_p = top_p, repetition_penalty = repetition_penalty)

    logger.debug('Evaluating Finetuned Model')
    e_finetuned = Evaluator(model_path = finetuned_model_path, model_type = "finetuned")
    finetuned_model, finetuned_tokenizer = e_finetuned.load_model_artifacts()
    average_rogue_score_finetuned, average_sbert_cosine_similarity_scores_finetuned = e_finetuned.evaluate_pipeline(model_type="finetuned", dataset = dataset, model = finetuned_model, tokenizer = finetuned_tokenizer, do_sample = do_sample, max_new_tokens = max_new_tokens, temperature = temperature, top_k = top_k, top_p = top_p, repetition_penalty = repetition_penalty)


