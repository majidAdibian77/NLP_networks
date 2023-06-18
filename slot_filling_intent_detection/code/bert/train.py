from model import NewBert

from tqdm import tqdm  # for our progress bar
from torch.optim import AdamW
from transformers.models.bert.modeling_bert  import BertConfig
import torch 
import matplotlib.pyplot as plt
import os
import pickle

def plot_losses(losses, path, title):
    plt.plot(losses['train_loss'], label='train')
    plt.plot(losses['val_loss'], label='val')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path)
    print(f"The loss change diagram was saved in {path}")

def train_model(model, optim, epochs, train_dataloader, val_dataloader):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        model.train()
        # setup loop with TQDM and dataloader
        loop = tqdm(train_dataloader, leave=True)
        train_epoch_loss = []
        for step, batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device) 
            slots_ids = batch[3].to(device)
            intent_ids = batch[4].to(device)
            # process
            outputs = model(input_ids, attention_mask, token_type_ids, intent_ids, slots_ids)
            loss = outputs[1]
            loss.backward()
            # update parameters
            optim.step()
            train_epoch_loss.append(loss.item())
            avg_epoch_loss = sum(train_epoch_loss)/len(train_epoch_loss)
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch} (train)')
            loop.set_postfix(loss=avg_epoch_loss)
        
        train_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        ## save trained model after any epoches
        torch.save(model.state_dict(), os.path.join('bert', 'model', 'bert.pt'))

        model.eval()
        loop = tqdm(val_dataloader, leave=True)
        val_epoch_loss = []
        for i, batch in enumerate(loop):
            # pull all tensor batches required for training
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device) 
            slots_ids = batch[3].to(device)
            intent_ids = batch[4].to(device)
            # process
            with torch.no_grad(): 
                outputs = model(input_ids, attention_mask, token_type_ids, intent_ids, slots_ids)

            # extract loss
            loss = outputs[1]
            intent_logits, slot_logits = outputs[0]
            val_epoch_loss.append(loss.item())
            avg_epoch_loss = sum(val_epoch_loss)/len(val_epoch_loss)
            loop.set_description('validation')
            loop.set_postfix(loss=avg_epoch_loss)
        val_loss.append(sum(val_epoch_loss)/len(val_epoch_loss))
    return train_loss, val_loss

if __name__ == "__main__":
    print("load train and val dataloader..")
    with open(os.path.join('bert', 'data', 'train_dataloader.pth'), "rb") as file:
        train_dataloader = pickle.load(file)
    with open(os.path.join('bert', 'data', 'val_dataloader.pth'), "rb") as file:
        val_dataloader = pickle.load(file)
    with open(os.path.join('bert', 'data', 'info.p'), 'rb') as file:
        info = pickle.load(file)

    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device: ' + str(device))

    print("create model..")
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = NewBert(config, len(info['intent2id']), len(info['slot2id']), 0.1)
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    epochs = 10
    print("#########################")
    train_loss, val_loss = train_model(model, optim, epochs, train_dataloader, val_dataloader)
    print("#########################")
    plot_losses({'train_loss':train_loss, 'val_loss':val_loss}, os.path.join('bert', 'model', 'loss.jpg'), 'loss per epoch')