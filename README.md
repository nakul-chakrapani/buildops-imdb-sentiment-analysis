# buildops-imdb-sentiment-analysis
Take home assignment for Buildsops ML interview
Author - Nakul

## Overview 
This project trains a machine learning model to predict the sentiment of IMDB movie reviews from huggingface Datasets
There are 25000 Training samples and 25000 Test samples and both the datasets are equally distributed (positive and negative reviews)

Machine Learning model was built using DistilBERT as the pre trained model and it was finetuned for the IMDB dataset. 
It's a very primitive DistilBERT model (no extra Fully conencted layers on top of base model) performing Binary sequence Classification.


## Installation Instructions
Please install libraries mentioned in requirements.txt

Training was mainly done on Google colab (which comes with all libraries pre installed). Easy GPU access is the main reason
Here's the link for google colab https://colab.research.google.com/gist/nakul-chakrapani/1f4f9c7c7a1b54eea2467cbd1448bbe6/imdb-sentiment-analysis.ipynb

Google colab follows step by step flow of my problem approach
Data Analysis -> Data preprocessing -> Model training -> Evaluation


### Custom loss function
Custom loss function was used and it's present in a separate file in src
Thought process behind Custom loss is very simple. From Data Analysis, I noticed most of the reviews have average of 220+ tokens. 
If the user is writing a review with more content (words), most probably it will have more inforamtion to understand the sentiment compared to short sentences. 
As most of the sentences are longer, I thought penalizing wrong prediction on longer sentences will make model learn better.

Crossentropy loss is calculated as usual (like normal loss) but a regulirization kind of parameter was added based on the length of review. 
As we have set 512 as maximum sentence length, sentences closer to 512 token will have high penalizing factor which will be added to loss

Unfortunately while training, it didn't show much improvement over the default loss :'(

### Model performance
Accuracy was the metric used to measure the performance of model. 


### Potential Improvements

1) Performance of the model is not good at all (Didn't get much time to debug model performance and syncing with colab killed my time)
2) Model architecture can be better adding extra layers on top base Distilbert model
3) For code organization - poetry/cookie cutter could've been used
4) Dockerize the application
5) Use better loggers
6) Simple RNN/LSTM model should've been given a try

