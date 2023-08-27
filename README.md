# Hate-Speech-Classification-using-Deep-Learning
I have presented a transfer learning-based approach to efficiently detect hate and offensive speech from social media content using deep neural networks.
For the classification task, I started with the well-known RoBERTa model as a base model and then utilized a variety of techniques to fine-tune the model using GRU.
The dataset used in this project is Hate Speech Detection by Davidson et al.(2017) and made accessible through Crowd-Flower. Nearly, 25000 tweets that have been annotated by three different persons are included in this collection. Each tweet has been categorized into one of three classes: "Hate Speech", "Offensive" and "Neither". The training data file which is used in the classification task contains almost 10,654 tweets and the testing data file contains almost 7000 tweets. The distribution of the texts among the three classes in the training data file is not balanced. All three of these classes need to be balanced for the classification task, thus we utilized the random over-sampler technique to achieve that. Each of the three classes now includes 6410 tweets, for a total of 19,230 tweets in the training set, by using the Random Over Sampler technique to balance the three classes. The numerals 0, 1, and 2 were used to represent each of the three classes "Hate Speech," "Offensive Language," and "Neither," respectively.
Pre-processing: We have removed numbers, hashtags, URLs, and user handles (@username). On the data, we have also applied stemming. We have removed all punctuation, confusing Unicode, and extraneous delimiting characters. Aside from that, we changed the case of every tweet to lowercase.
Fine-Tuning Techniques: In this project, I am fine-tuning RoBERTa with the addition of Bi-GRU top layers, which are then connected to the Dense layer to obtain the output values.
Model Training: We have specified the following hyperparameters for our model’s training:
–> batch size: 30 
–> epochs: 10 
–> Adam optimizer with a learning rate of (2e-5) 
–> Early stopping callback set on validation recall
–> Checkpoint: Following every epoch, the model will be recorded in the file.
–> Categorical cross-entropy loss function, which is best suited for multi-class classification applications, is the loss function that is employed.
