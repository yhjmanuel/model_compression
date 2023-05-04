import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaLayer

from train import make_data, pretrained_path, max_length, TensorDataset, f1_score, DataLoader, Dataset
import time


def get_extended_attention_mask(attention_mask, input_shape):
    dtype = torch.float32
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class StudentCrossEncoderModel(nn.Module):
    def __init__(self, compress_ratio, teacher_model):
        # compress_ratio: the magnitude to which you would like to compress the model
        # 2, 3, etc.
        super(StudentCrossEncoderModel, self).__init__()
        self.embedding, self.classifier = None, None
        self.roberta_layers = nn.ModuleList()
        assert(int == type(compress_ratio))
        if compress_ratio <= 1:
            raise ValueError('Student model should have less Roberta layers than teacher model')
        elif teacher_model.config.num_hidden_layers % compress_ratio != 0:
            raise ValueError('wrong compression ratio, make sure n_teacher_model_layers % n_teacher_model_layers = 0')
        self.n_layers = int(teacher_model.config.num_hidden_layers / compress_ratio)
        self._init_student_model(teacher_model)
    
    def _init_student_model(self, teacher_model):
        self.embeddings = teacher_model.roberta.embeddings
        # for example, if we want to compress 6 layers to 3, we copy the weights of 
        # the teacher model's layer 4, 5, 6 for the student model
        start_layer = teacher_model.config.num_hidden_layers - self.n_layers
        for i in range(start_layer, teacher_model.config.num_hidden_layers):
            self.roberta_layers.append(teacher_model.roberta.encoder.layer[i])
            # self.roberta_layers.append(RobertaLayer(teacher_model.config))
        self.classifier = teacher_model.classifier
    
    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        for i in range(self.n_layers):
            extended_attention_mask = get_extended_attention_mask(attention_mask, x.shape)
            x = self.roberta_layers[i](x, attention_mask=extended_attention_mask)[0]
        return self.classifier(x)
    
    def partial_forward(self, x, attention_mask, n_layer):
        extended_attention_mask = get_extended_attention_mask(attention_mask, x.shape)
        return self.roberta_layers[n_layer](x, attention_mask = extended_attention_mask)
    
    def embedding_forward(self, x):
        return self.embeddings(x)
    
    def classifier_forward(self, x):
        return self.classifier(x)


class TeacherCrossEncoderModel(RobertaForSequenceClassification):
    def partition_roberta_layers(self, compress_ratio):
        self.partitions = nn.ModuleList()
        n_partitions = int(self.config.num_hidden_layers / compress_ratio)
        partition_len = int(self.config.num_hidden_layers / n_partitions)
        for i in range(n_partitions):
            partition = nn.ModuleList()
            for j in range(i * partition_len, (i+1) * partition_len):
                partition.append(self.roberta.encoder.layer[j])
            self.partitions.append(partition)
    
    def partial_forward(self, x, attention_mask, n_partition):
        partition = self.partitions[n_partition]
        for i in range(len(partition)):
            extended_attention_mask = get_extended_attention_mask(attention_mask, x.shape)
            x = partition[i](x, attention_mask=extended_attention_mask)[0]
        return x


class BaseReplacementRateScheduler:
    def __init__(self, replace_config):
        self.rate = replace_config.base_rate
        self.update_factor = replace_config.update_factor
        self.mode = replace_config.mode
        self.steps = 0
        
    def update(self):
        self.steps += 1
        if self.mode == 'linear':
            self.rate += self.update_factor
        else:
            # more drastically increasing the rate
            self.rate += self.steps * self.update_factor
    
    def get_rate(self):
        print('current replacement rate: '+ str(min(1.0, self.rate)))
        return min(1.0, self.rate)

    
class KDTrainer:
    def __init__(self, teacher_model, student_model, loaders, optimizer, device, n_epochs, 
                 print_freq, loss_func, replace_scheduler):
        self.t_model = teacher_model
        self.s_model = student_model
        self.data_loaders = loaders
        self._norm_teacher_model()
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.print_freq = print_freq
        self.loss_func = loss_func
        self.rs = replace_scheduler
        # used for progressive module replacing
        self.bnl = Bernoulli(torch.tensor(self.rs.get_rate()))
        
    def _norm_teacher_model(self):
        # freeze param
        # for param in self.t_model.parameters():
        #     param.requires_grad = False
        # delete embedding layer + classifier layer to save memory
        del self.t_model.roberta.embeddings
        del self.t_model.classifier
    
    def run(self):
        self.s_model = self.s_model.to(self.device)
        self.s_model.train()
        self.t_model = self.t_model.to(self.device)
        self.t_model.eval()
        for i in range(self.n_epochs):
            print('-------------Epoch {}-----------'.format(i+1))
            print('Doing Training...')
            self.train()
            print('Doing Evaluation...')
            self.eval()
            
    def get_logits_and_loss(self, batch, mode='train'):
        # by not using **batch, we are indicating that we do not use token_type_ids for this project
        if mode == 'train':
            x = self.s_model.embedding_forward(batch['encoding']['input_ids'])
            for i in range(self.s_model.n_layers):
                # student model
                if self.bnl.sample() == 1:
                    x = self.s_model.partial_forward(x, batch['encoding']['attention_mask'], i)[0]
                else:
                    # teacher model
                    x = self.t_model.partial_forward(x, batch['encoding']['attention_mask'], i)
            logits = self.s_model.classifier_forward(x)
        else:
            logits = self.s_model(input_ids = batch['encoding']['input_ids'],
                                  attention_mask = batch['encoding']['attention_mask'])
        batch_loss = self.loss_func(logits, batch['label'].view(-1))
        return logits, batch_loss
    
    def train(self):
        n_batch, temp_loss = 0, 0.
        labels = []
        preds = []
        for idx, batch in enumerate(self.data_loaders['train']):
            for item in batch['encoding']:
                batch['encoding'][item] = batch['encoding'][item].view(-1, max_length)
                batch['encoding'][item] = batch['encoding'][item].to(self.device)
            batch['label'] = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            logits, batch_loss = self.get_logits_and_loss(batch, mode='train')
            labels.extend(list(batch['label'].view(-1).cpu()))
            preds.extend(list(torch.argmax(logits.cpu(), dim=-1)))
            temp_loss += batch_loss
            n_batch += 1
            batch_loss.backward()
            self.optimizer.step()
            if n_batch % int(self.print_freq) == 0:
                self.rs.update()
                self.bnl = Bernoulli(torch.tensor(self.rs.get_rate()))
                print('Avg Loss for batch {} - {}: {:.3f}'.format(n_batch - int(self.print_freq) + 1,
                                                                  n_batch,
                                                                  temp_loss / int(self.print_freq)))
                temp_loss = .0
                print('accumulative micro F1 ', f1_score(labels, preds, average='micro'))
                print('accumulative macro F1 ', f1_score(labels, preds, average='macro'))
                
    def eval(self):
        # when doing evaluations, the only metric we want to print is the loss
        n_batch, temp_loss = 0, 0.
        preds, labels = [], []
        with torch.no_grad():
            for idx, batch in enumerate(self.data_loaders['dev']):
                for item in batch['encoding']:
                    batch['encoding'][item] = batch['encoding'][item].view(-1, max_length)
                    batch['encoding'][item] = batch['encoding'][item].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                logits, batch_loss = self.get_logits_and_loss(batch, mode='eval')
                labels.extend(list(batch['label'].view(-1).cpu()))
                preds.extend(list(torch.argmax(logits.cpu(), dim=-1)))
                temp_loss += batch_loss
                n_batch += 1
        print('total micro F1 ', f1_score(labels, preds, average='micro'))
        print('total macro F1 ', f1_score(labels, preds, average='macro'))
        torch.save(self.s_model.state_dict(), 'kd_cross_encoder_model_3_layer.pt')


class RSConfig:
    base_rate = 0.2
    mode = 'linear'
    update_factor = 0.02


if __name__ == '__main__':
    # load data and make dataloaders
    train_data, dev_data = make_data()
    train_set = TensorDataset(train_data)
    dev_set = TensorDataset(dev_data)
    # actually batch size not 1 -> all documents under a single query
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=True, pin_memory=True)
    loaders = {'train': train_loader, 'dev': dev_loader}
    
    # make teacher & student models
    compress_ratio = 6
    teacher_model = TeacherCrossEncoderModel
    teacher_model = teacher_model.from_pretrained(pretrained_path)
    ce_model_saved = 'cross_encoder_model.pt'
    teacher_model.load_state_dict(torch.load(ce_model_saved))
    teacher_model.partition_roberta_layers(compress_ratio)
    student_model = StudentCrossEncoderModel(compress_ratio, teacher_model)
    
    # train config
    optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-6)
    device = 'mps'
    n_epochs = 7
    print_freq = 10
    loss_func = nn.CrossEntropyLoss()
    rs = BaseReplacementRateScheduler(RSConfig)

    # start training
    trainer = KDTrainer(teacher_model = teacher_model, student_model = student_model,
                        loaders = loaders, optimizer = optimizer, device = device,
                        n_epochs = n_epochs, print_freq = print_freq,
                        loss_func = loss_func, replace_scheduler = rs)
    trainer.run()