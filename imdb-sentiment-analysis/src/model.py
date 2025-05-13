'''
Python util wrappers for IMDB sentiment analysis model training
'''
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def get_tokenizer(model_name):
    '''
    Function to get the DistilBERT Tokenizer
    '''
    print(f'Using model {model_name}')
    tokenizer_obj = DistilBertTokenizer.from_pretrained(model_name)
    return tokenizer_obj

def get_pretrained_model(model_name, num_labels):
    '''
    Function to get DistilBERT pretrained model
    '''
    print(f'Using model {model_name}')
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
