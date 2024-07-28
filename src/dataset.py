# create dataset class
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import AutoTokenizer


class Dataset4Summarization(Dataset):
	def __init__(self, data, tokenizer, max_length=5096, chunk_length =1024, overlap=16):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.chunk_length = chunk_length
		self.overlap = overlap

	def __len__(self):
		return len(self.data)

	def join_single_documents(self, single_documents):
		text = ""
		for documents in single_documents:
			text += documents['raw_text'] + " "
		return text
	
	def chunking(self, text):
		chunks = []
		for i in range(0, self.max_length, self.chunk_length-self.overlap):
			chunks.append(text[i:i+self.chunk_length])
		return chunks

	def __getitem__(self, idx):
		sample = self.data[idx]
		text = self.join_single_documents(sample['single_documents'])
		inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=5096)
		target = self.tokenizer(sample['summary'], return_tensors='pt', padding='max_length', truncation=True, max_length=256)
		list_chunk = self.chunking(inputs['input_ids'].squeeze())
		list_attention_mask = self.chunking(inputs['attention_mask'].squeeze())


		del text
		del sample
		return {
			'list_input_ids': list_chunk,
			'list_att_mask' : list_attention_mask,
			'target': target['input_ids'].squeeze()
		}

if __name__ == '__main__':

    with open("/home/trnmah/UET-project/summarization/drive-download-20240727T015440Z-001/train_data_new.jsonl", 'r') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization"'facebook/bart-large-cnn')
    dataset = Dataset4Summarization(data, tokenizer)

    train_loader = DataLoader(dataset, batch_size=2)