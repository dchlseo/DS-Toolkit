"""
This script is designed to demonstrate a complete workflow using a transformer model from Hugging Face for a classification task. 
It involves the following steps: data import, data preparation, model training, validation, and evaluation on test datasets. 
The script is structured to be modular, with functions handling specific parts of the workflow.

Functions:
- get_tokenizer(model_name): Returns a tokenizer for a given transformer model.
- load_model(model_name, num_labels): Loads and returns a specified transformer model.
- get_ids_masks(sentences, tokenizer, max_length): Tokenizes sentences and returns input IDs and attention masks.
- prepare_data(df, tokenizer, max_length, batch_size, is_training_data): Prepares data for training or testing and returns dataloaders.
- model_train(...): Trains and validates the model, with an option to save the model.
- evaluate_model(model, dataloader): Evaluates the model's performance on a given dataset.

The sample data used here is the 'Justice' dataset from the "Aligning AI With Shared Human Values" project.
Article Citation:
@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
Dataset GitHub: https://github.com/hendrycks/ethics/tree/master

"""

# ---------------------------------------------
## IMPORT PACKAGES AND DATA

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

import datetime

# INSERT: YOUR DATA DIR (given that train test data are split)
train_data_dir = './ethics/justice/justice_train.csv'  
test_data_dir = './ethics/justice/justice_test.csv'
test_hard_dir = './ethics/justice/justice_test_hard.csv'
    
train_df = pd.read_csv(train_data_dir)
test_df = pd.read_csv(test_data_dir)
test_hard_df = pd.read_csv(test_hard_dir)
print('data import complete.')


## PREPARING DATA

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_model(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

# tokenization, masks
def get_ids_masks(sentences, tokenizer, max_length):
    encodings = tokenizer(sentences, 
                          max_length=max_length, 
                          padding='max_length',
                          truncation=True,
                          return_tensors='pt') # get pytorch tensor
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    return input_ids, attention_masks


def prepare_data(df, tokenizer, max_length, batch_size, is_training_data=False):
    input_ids, attention_masks = get_ids_masks(list(df['scenario']), tokenizer, max_length)
    labels = torch.tensor(df['label'].values)

    if is_training_data:
        # train test split
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,  
                                                                                            random_state=42, test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, 
                                                               random_state=42, test_size=0.1)

        # create tensor dataset
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

        # create dataloaders
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)

        return train_dataloader, validation_dataloader
    else:
        data = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        return dataloader
    

## MODEL TRAINING (AND SAVING)

def model_train(model, model_name, train_dataloader, validation_dataloader, epochs=3, lr=5e-5, save_model=False, optimizer_name='AdamW'):
    
    # initialize optimizer
    if optimizer_name == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized")    
    
    # set training steps
    total_steps = len(train_dataloader) * epochs

    # init lr scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # # loss function
    # loss_fn = CrossEntropyLoss()

    # training loop
    for epoch in range(epochs):
        # training
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        print('commence model training....')

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()  # gradient init

            # forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            # calculate accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            total_train_accuracy += (preds == b_labels).cpu().numpy().mean()

            # backward pass
            loss.backward()

            # optimizer update
            optimizer.step()
            scheduler.step()

        # get average loss and accuracy over all batches (training)
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.3f}, Train Accuracy: {avg_train_accuracy:.3f}")
        
        # validation
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        print('commence model validation....')

        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():  # no need for gradients
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            total_eval_loss += loss.item()

            # calculate accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            total_eval_accuracy += (preds == b_labels).cpu().numpy().mean()

        # get average loss and accuracy over all batches (validation)
        avg_eval_loss = total_eval_loss / len(validation_dataloader)
        avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_eval_loss:.3f}, Validation Accuracy: {avg_eval_accuracy:.3f}")

    # save model
    if save_model:
        current_time = datetime.datetime.now().strftime("%y%m%d-%H-%M")
        model_filename = f"sdc_{model_name}-{current_time}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f'Model saved as {model_filename}.')
    
    # return metrics
    return avg_train_loss, avg_train_accuracy, avg_eval_loss, avg_eval_accuracy


## MODEL EVALUATION

def evaluate_model(model, dataloader):
    total_accuracy = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask)

            preds = torch.argmax(outputs.logits, dim=1)
            total_accuracy += (preds == b_labels).cpu().numpy().mean()
    
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_accuracy


## DATA PROCESSING
# init
model_name = 'distilbert-base-uncased'
tokenizer = get_tokenizer(model_name)
model = load_model(model_name, 2)
max_length = 128
batch_size = 32

# data prep (train, test, test_hard)
train_dataloader, validation_dataloader = prepare_data(train_df, tokenizer, max_length, batch_size, is_training_data=True)
test_dataloader = prepare_data(test_df, tokenizer, max_length, batch_size)
test_hard_dataloader = prepare_data(test_hard_df, tokenizer, max_length, batch_size)
print('data loaders generated.')


# train model
avg_train_loss, avg_train_accuracy, avg_eval_loss, avg_eval_accuracy = \
    model_train(model, model_name, train_dataloader, 
                validation_dataloader, epochs=3, lr=5e-5, 
                save_model=True, optimizer_name='AdamW')   # save_model=True!

print(f"Training and Validation Results:\n"
      f"  Average Training Loss: {avg_train_loss:.3f}\n"
      f"  Average Training Accuracy: {avg_train_accuracy:.2f}\n"
      f"  Average Validation Loss: {avg_eval_loss:.3f}\n"
      f"  Average Validation Accuracy: {avg_eval_accuracy:.2f}")
print('-----------------------------------')

# test model
# test.csv
test_accuracy = evaluate_model(model, test_dataloader)
print(f"Accuracy on Test Data: {test_accuracy:.2f}")

# test_hard.csv
test_hard_accuracy = evaluate_model(model, test_hard_dataloader)
print(f"Accuracy on Test-Hard Data: {test_hard_accuracy:.2f}")
