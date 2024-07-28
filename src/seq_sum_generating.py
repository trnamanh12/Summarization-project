import torch

def train_fn(model, model1, train_loader, optimizer, scheduler, criterion, epochs, device):
	model.train()
	for epoch in range(epochs):
		for iter in train_loader:
			summaries = []
			model.zero_grad(set_to_none=True)
			inputs = iter['list_input_ids']
			att_mask = iter['list_att_mask']
			target = iter['target'].to(device)
			for i in range(len(inputs)-1):
				if i == 0:
					inputs[i] = inputs[i].to(device)
					# att_mask[i] = att_mask[i].to(device)
					summary = model.generate(inputs[i], max_length=96, num_beams=4, early_stopping=True)
					summaries.append(summary)
				else:
					inputs[i] = torch.cat([inputs[i], [summary[i-1]]], dim=1).to(device)
					
					# att_mask[i] = att_mask[i].to(device)
					summary = model.generate(inputs[i], max_length=96, num_beams=4, early_stopping=True)
					summaries.append(summary)

			# summaries = torch.cat(summaries, dim=1).to(device)
			# summaries = model.generate(summaries, max_length=256, num_beams=4)
			# del inputs
			final_summary = summaries[-1]
			if summaries.shape[1] > 256:
				final_summary = final_summary[:, :256]
				att_mask = torch.ones((final_summary.shape[0], 256)).to(device)
			else:
				att_mask = torch.ones((final_summary.shape[0], final_summary.shape[1])).to(device)
				final_summary = torch.nn.functional.pad(final_summary, (0, 256-final_summary.shape[1]), value=0)
				att_mask = torch.nn.functional.pad(att_mask, (0, 256-att_mask.shape[1]), value=0)
			
			summaries = model1(final_summary,attention_mask=att_mask, labels=target)
			loss = summaries.loss

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			loss.backward()
			optimizer.step()
			
			print(loss.item())
		scheduler.step()
		print(f'Epoch {epoch} done')
