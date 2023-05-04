import torch
import torch.nn as nn
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import defaultdict
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import time


pretrained_path = 'cross-encoder/nli-distilroberta-base'
max_length = 128

def make_data():
    # read_query
    query_data = {}
    with open('queries.jsonl') as query_file:
        for line in query_file:
            temp = json.loads(line)
            query_data[temp['_id']] = temp['text']
            
    # read relevance score
    qrels_df = pd.read_csv('test.tsv', sep="\t", header=0)
    
    # read doc content
    doc_data = {}
    with open('corpus.jsonl') as corpus_file:
        for line in corpus_file:
            temp = json.loads(line)
            doc_data[temp['_id']] = temp['title'] + ' ' +  temp['text']
    doc_ids = set(doc_data)
    
    # data augmentation
    query_doc_score = defaultdict(defaultdict)
    for i, row in qrels_df.iterrows():
        query_doc_score[row[0]][row[1]] = row[2]
    for qid in query_doc_score:
        irre_docs = list(doc_ids.difference(set(query_doc_score[qid])))
        # sample 0 relevance score docs
        irre_docs = random.sample(irre_docs, len(query_doc_score[qid]))
        for irre_doc in irre_docs:
            query_doc_score[qid][irre_doc] = 0
    
    # make train data
    train_data = []
    for qid in query_doc_score:
        docids = list(set(query_doc_score[qid]))
        sample_docids = random.sample(docids, min(60, len(docids)))
        train_data.append({'qid': qid,
                           'text': [[query_data[qid], doc_data[docid]] for docid in sample_docids],
                           'label': [query_doc_score[qid][docid] for docid in sample_docids]})
    
    # split 
    random.Random(42).shuffle(train_data)
    length = len(train_data)
    split_point = int(length * 0.7)
    
    # train, dev
    return train_data[:split_point], train_data[split_point:]

class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # encode all documents under a query. Each concatenated with the query
        # so the query will be used repeatedly
        instance = self.data[index]
        encoding = self.tokenizer(instance['text'],padding='max_length',
                                  max_length=max_length,
                                  truncation=True, return_tensors='pt')
        return {'encoding': encoding, 'label': torch.tensor(instance['label'], dtype=torch.int64)}


class Trainer:
    def __init__(self, loaders, model, optimizer, device, n_epochs, 
                 print_freq, loss_func):
        self.data_loaders = loaders
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.print_freq = print_freq
        self.loss_func = loss_func
        
    def run(self):
        for i in range(self.n_epochs):
            print('-------------Epoch {}-----------'.format(i+1))
            print('Doing Training...')
            self.train()
            print('Doing Evaluation...')
            self.eval()
    
    def get_logits_and_loss(self, batch):
        # by not using **batch, we are indicating that we do not use token_type_ids for this project
        logits = self.model(**batch['encoding']).logits
        batch_loss = self.loss_func(logits, batch['label'].view(-1))
        return logits, batch_loss
    
    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        n_batch, temp_loss = 0, 0.
        labels = []
        preds = []
        for idx, batch in enumerate(self.data_loaders['train']):
            for item in batch['encoding']:
                batch['encoding'][item] = batch['encoding'][item].view(-1, max_length)
                batch['encoding'][item] = batch['encoding'][item].to(self.device)
            batch['label'] = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            logits, batch_loss = self.get_logits_and_loss(batch)
            labels.extend(list(batch['label'].view(-1).cpu()))
            preds.extend(list(torch.argmax(logits.cpu(), dim=-1)))
            temp_loss += batch_loss
            n_batch += 1
            batch_loss.backward()
            self.optimizer.step()
            if n_batch % int(self.print_freq) == 0:
                print('Avg Loss for batch {} - {}: {:.3f}'.format(n_batch - int(self.print_freq) + 1,
                                                                  n_batch,
                                                                  temp_loss / int(self.print_freq)))
                temp_loss = .0
                print('accumulative micro F1 ', f1_score(labels, preds, average='micro'))
                print('accumulative macro F1 ', f1_score(labels, preds, average='macro'))

    def eval(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        # when doing evaluations, the only metric we want to print is the loss
        n_batch, temp_loss = 0, 0.
        preds, labels = [], []
        with torch.no_grad():
            for idx, batch in enumerate(self.data_loaders['dev']):
                for item in batch['encoding']:
                    batch['encoding'][item] = batch['encoding'][item].view(-1, max_length)
                    batch['encoding'][item] = batch['encoding'][item].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                logits, batch_loss = self.get_logits_and_loss(batch)
                labels.extend(list(batch['label'].view(-1).cpu()))
                preds.extend(list(torch.argmax(logits.cpu(), dim=-1)))
                temp_loss += batch_loss
                n_batch += 1
        print('total micro F1 ', f1_score(labels, preds, average='micro'))
        print('total macro F1 ', f1_score(labels, preds, average='macro'))
        torch.save(self.model.state_dict(), 'cross_encoder_model.pt')


if __name__ == '__main__':
    train_data, dev_data = make_data()
    train_set = TensorDataset(train_data)
    dev_set = TensorDataset(dev_data)
    # actually batch size not 1 -> all documents under a single query
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=True, pin_memory=True)
    loaders = {'train': train_loader, 'dev': dev_loader}
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    device = 'mps'
    n_epochs = 5
    print_freq = 5
    loss_func = nn.CrossEntropyLoss()

    trainer = Trainer(loaders = loaders, model = model, optimizer = optimizer,
                  device = device, n_epochs = n_epochs, print_freq = print_freq,
                  loss_func = loss_func)
    trainer.run()