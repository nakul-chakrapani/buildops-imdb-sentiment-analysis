''' All data preprocessing related code
Wrappers to create pytorch datasets, dataloaders for the IMDB Dataset from huggingface
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import os
import traceback

class IMDBDataset(Dataset):
    def __init__(self, sentences_list, labels_list, tokenizer, max_length=512):
        super(IMDBDataset, self).__init__()

        self.sentences = sentences_list
        self.labels = labels_list

        # BERT Tokenizer for tokenizing sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]

        encoding = self.tokenizer(sentence,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=self.max_length,
                                 return_tensors='pt')

        # Calculate the number of tokens in sentence for custom loss
        # Reducing it by 2 because first and last tokens are CLS and SEP
        # This is used for custom loss implementation later in the training
        num_tokens = (encoding['attention_mask'] == 1).sum() - 2
        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "num_tokens": num_tokens,
            "label": torch.tensor(label)
        }

# Python wrapper to get pytorch datasets
def get_datasets(sentences_list, labels_list, tokenizer, max_length=512):
    dataset = IMDBDataset(sentences_list, labels_list, tokenizer, max_length)
    return dataset

# Python wrapper to get pytorch dataloaders
def get_dataloaders(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_sentence_len(row):
    ''' Function to get the length of sentence based on space as delimiter
    Not perfect function but decent for approximation
    '''
    sentence = row['text']
    return len(sentence.split(' '))

def get_imdb_raw_data(train_size, test_size, val_size):
    ''' Function to download raw IMDB dataset
    Returns - train, test and validation pandas dataframes
    '''
    data_folder = '../data'

    if not os.path.exists(data_folder):
        ds = load_dataset("stanfordnlp/imdb")
    else:
        ds = load_from_disk(data_folder)

    try:
        train_df = ds['train'].to_pandas()
        train_df = train_df.sample(n=train_size)
        all_test_df = ds['test'].to_pandas()

        # Create Validation and Test dataframes
        test_df = all_test_df.sample(n=test_size)
        val_df = all_test_df.sample(n=val_size)

        # Add Sentence length (based on space delimiter) to dataframe
        train_df['text_len'] = train_df.apply(get_sentence_len, axis=1)
        test_df['text_len'] = test_df.apply(get_sentence_len, axis=1)
        val_df['text_len'] = val_df.apply(get_sentence_len, axis=1)

        return train_df, test_df, val_df

    except BaseException as be:
        print("Exception while creating dataframes")
        traceback.print_exc()
