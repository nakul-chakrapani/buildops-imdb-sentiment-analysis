''' Main Driver for Model training
    - Downloads the data and splits it into Train, test and validation
    - Creates datasets and dataloaders for Train, test and validation
    - Instantiates the model and tokenizer
    - Run Training, tracks loss every epoch
    - Evaluates on the test dataset and prints Accuracy
'''

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import math

from data_preprocessing import get_imdb_raw_data, get_datasets, get_dataloaders
from model import get_tokenizer, get_pretrained_model
from custom_loss import CustomLoss

def get_input_tensors(input_encoding, device):
    '''
    Function to extract tensors from tokenizer output
    '''
    input_ids = input_encoding['input_ids']
    attention_masks = input_encoding['attention_mask']
    labels = input_encoding['label']
    num_tokens = input_encoding['num_tokens']

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = labels.to(device)
    num_tokens = num_tokens.to(device)

    return input_ids, attention_masks, labels, num_tokens

def train_model(train_dataloader, val_dataloader, num_epochs, learning_rate,
                num_labels, device='cpu'):

    imdb_model = get_pretrained_model(num_labels=1)
    imdb_model.to(device)

    # Right now using only Adam optimizer
    optimizer = torch.optim.AdamW(imdb_model.parameters(), lr=learning_rate)

    if custom_loss_fn:
        print("Using Custom loss function")
    else:
        print("Using default loss from Distilbert")

    for epoch in range(num_epochs):
      imdb_model.train()
      train_loss = 0
      for step_num, input_encoding in enumerate(train_dataloader):
          input_ids, attention_masks, labels, num_tokens = get_input_tensors(input_encoding, device)
          preds = imdb_model(input_ids, attention_mask=attention_masks, labels=labels.float())

          if custom_loss_fn is not None:
              loss = custom_loss_fn(preds.logits, labels, num_tokens)
          else:
              loss = preds.loss

          # Accumulate Training loss over all steps in an epoch
          train_loss += loss.item()

          # Backward propagation
          optimizer.zero_grad()
          loss.backward()

          # Update model weights
          optimizer.step()

          if (step_num+1)%500 == 0:
              print(f"Epoch: {epoch+1}/{num_epochs}, Step: {step_num+1}/{num_total_steps}, Training Loss so far: {train_loss/((step_num+1)*batch_size)}")

      print(f"Epoch: {epoch+1}/{num_epochs},Total Training Loss: {train_loss/(len(train_dataset))}")

      # Calculate Validation at the end of every epoch
      imdb_model.eval()
      val_loss = 0
      with torch.no_grad():
          for step_num, input_encoding in enumerate(val_dataloader):
              input_ids, attention_masks, labels, num_tokens = get_input_tensors(input_encoding, device)

              preds = imdb_model(input_ids, attention_mask=attention_masks, labels=labels.float())

              if custom_loss_fn is not None:
                  val_loss += custom_loss_fn(preds.logits, labels, num_tokens)
              else:
                  val_loss += preds.loss

          print(f"Epoch: {epoch+1}/{num_epochs}, Total Validation Loss: {val_loss.item()/len(val_dataset)}")

    return imdb_model

def evaluate_model(imdb_model, test_dataloader, device='cpu'):
    imdb_model.eval()
    acc = 0
    with torch.no_grad():
        for step_num, input_encoding in enumerate(test_dataloader):
            input_ids, attention_masks, labels, num_tokens = get_input_tensors(input_encoding, device)

            outputs = imdb_model(input_ids, attention_mask=attention_masks, labels=labels)
            preds = torch.argmax(outputs.logits, dim=1)

            # Accumulate correct predictions
            acc += (labels == preds).sum().item()

    print(f"Accuracy: {acc/len(test_dataset)}")

if __name__ == '__main__':
    model_name = 'distilbert-base-uncased'
    train_size, test_size, val_size = 25000, 2500, 100

    # Hyperparameters for model training
    max_length = 512 # Maximum number of tokens considersd in a sentence
    learning_rate = 1e-5
    momentum = 0.9
    num_epochs = 2
    batch_size = 8
    custom_loss = CustomLoss(parameter=0.01)
    num_outputs = 1 # Single logit for binary classification

    # Step 0) Determine device on which model will be get_pretrained_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on {}".format(device))

    # Step 1) Downloads the data and splits it into Train, test and validation
    try:
        train_df, test_df, val_df = get_imdb_raw_data(train_size, test_size, val_size)
    except BaseException as be:
        print(f"Exception while downloading IMDB dataframes {be}")
        sys.exit(0)

    # Step 2) Get Tokenizer
    try:
        tokenizer_obj = get_tokenizer(model_name)
    except BaseException as be:
        print(f"Exception while obtaining tokenizer {be}")
        sys.exit(0)

    # Step 3) Creates datasets and dataloaders for Train, test and validation
    try:
        train_dataset = get_datasets(train_df['text'].to_list(), train_df['label'].to_list(), tokenizer_obj, max_length=max_length)
        test_dataset = get_datasets(test_df['text'].to_list(), test_df['label'].to_list(), tokenizer_obj, max_length=max_length)
        val_dataset = get_datasets(val_df['text'].to_list(), val_df['label'].to_list(), tokenizer_obj, max_length=max_length)

        train_dataloader = get_dataloaders(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = get_dataloaders(test_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = get_dataloaders(val_dataset, batch_size=batch_size, shuffle=True)
    except BaseException as be:
        print(f"Exception while obtaining pytorch dataset/dataloaders {be}")
        sys.exit(0)

    # Step 4) Run model training
    imdb_model = None
    try:
        imdb_model = train_model(train_dataloader, val_dataloader, num_epochs,
                    learning_rate, num_outputs, device='cpu')
    except BaseException as be:
        print(f"Exception while training the model {be}")
        sys.exit(0)

    # Step 4) Run model evaluation
    try:
        if imdb_model:
            evaluate_model(imdb_model, test_dataloader)
    except BaseException as be:
        print(f"Exception while Evaluating the model {be}")
        sys.exit(0)

    # Step 5) Save the model
    if imdb_model:
        torch.save(imdb_model.state_dict(), '../model_artifacts')
