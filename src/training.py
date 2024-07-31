import json
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def process_data(data):
    single_documents = data.get('single_documents', [])
    # print(single_documents)
    summary = data.get('summary', '')
    # print(summary)
    
    result = []
    for doc in single_documents:
        raw_text = doc.get('raw_text', '')
        result.append([raw_text, summary])
        # print(raw_text)
    
    return result

def processing_data(input_file):
    all_results = []
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            result = process_data(data)
            all_results.extend(result)
    
    return all_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_file = "/kaggle/input/summarization-dataset-vlspbusu2022/train_data_new.jsonl"  # Replace with your actual input file name
valid_file = "/kaggle/input/summarization-dataset-vlspbusu2022/train_data_new.jsonl"  # Replace with your actual input file name
train_data = processing_data(input_file)
valid_data = processing_data(valid_file)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")
# model.to(device)
class Dataset4Sum_train(torch.utils.data.Dataset):
	def __init__(self, tokenizer, data, max_input_length=2048, max_output_length=256):
		self.tokenizer = tokenizer
		self.data = data
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, ids):
		data = self.data[ids]
		input_text, target_text = data[0], data[1]
		input_text = "Tóm tắt văn bản sau, đảm bảo rằng văn bản tóm tắt cần mang đủ thông tin quan trong của văn bản ban đầu: " + input_text
		tokenized_input = self.tokenizer(input_text, max_length=self.max_input_length, truncation=True, padding='max_length', return_tensors='pt')
		tokenized_target = self.tokenizer(target_text, max_length=self.max_output_length, truncation=True, padding='max_length', return_tensors='pt')
		return {
			'input_ids': tokenized_input['input_ids'].flatten(),
			'attention_mask': tokenized_input['attention_mask'].flatten(),
			'target_ids': tokenized_target['input_ids'].flatten(),
			'target_attention_mask': tokenized_target['attention_mask'].flatten()
		}
train_data = Dataset4Sum_train(tokenizer, train_data, 2048, 256 )
valid_data = Dataset4Sum_train(tokenizer, valid_data, 2048, 256 )
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, num_workers=4)
# criterion = torch.nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
# train model 
import time 


def train(world_size, model, train_loader, epochs, rank ):
        model.train()
        start= time.time()
        model.to(rank)
        running_loss = 0.0
        ddp_model = DDP(model, device_ids=[rank])

        criterion = torch.nn.CrossEntropyLoss().to(rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98) 

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                input_ids = data['input_ids'].to(rank)
                attention_mask = data['attention_mask'].to(rank)
                target_ids = data['target_ids'].to(rank)
                target_attention_mask = data['target_attention_mask'].to(rank)

    #             if ( i % 20 == 0):
                model.zero_grad(set_to_none=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=target_ids, decoder_attention_mask=target_attention_mask, labels=target_ids, return_dict=True)
                loss = (outputs.loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
    #             if ( i % 20 == 0):
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                # print loss after 100 steps
                if i % 100 == 0:
                    end = time.time()
                    print(f"Epoch: {epoch}, Step: {i}, Loss: {running_loss/i}, Time: {end-start}")
            end = time.time()
            print(f"Epoch: {epoch}, Loss: {running_loss / (epoch * len(train_loader))}, Time: {end-start}")
        cleanup()
        end = time.time()
        print(f"Full Time: {end-start}")

def run(train, world_size, model, train_loader , epochs ):
    mp.spawn(train, args=(world_size, model, train_loader,  epochs), nprocs=world_size, join=True)

run(train, 2, model, train_loader, 2, device)