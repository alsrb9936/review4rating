from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections.abc import Sequence

def _load_language_model(model_name, gpu_id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name) 

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding_batch(model_name: str, texts: Sequence[str], batch_size: int = 32, gpu_id:int = 0):
    tokenizer, model, device = _load_language_model(model_name, gpu_id)
    
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches, 
        desc="Getting Embedding", 
        unit="batch"
    ):  
        batch = list(texts[start : start + batch_size])
        encoded_input = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
            )

        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.extend(sentence_embeddings.cpu().tolist())

    return embeddings
