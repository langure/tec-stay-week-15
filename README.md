# Week:15 Sequence Labeling for Named Entity Recognition Using Conditional Random Fields
Named Entity Recognition (NER) is a common task in Natural Language Processing (NLP) that involves identifying named entities (like persons, organizations, locations, etc.) in text. It's a form of sequence labeling where each token in the text sequence is assigned a label. One popular method for sequence labeling tasks, including NER, is the use of Conditional Random Fields (CRFs).

What are Conditional Random Fields (CRFs)?
CRFs are a type of statistical modeling method often used in pattern recognition and machine learning for structured prediction. Unlike other methods, CRFs take into account the context within the input sequence to make a prediction, making them particularly well-suited for sequence labeling tasks in NLP. CRFs are discriminative models, which means they model the decision boundary between different classes (or labels) rather than modeling the distribution of each class.

How CRFs Work in NER
In the context of NER, CRFs are used to predict the labels for each token in the input sequence, where the labels represent the entity types (like 'Person', 'Organization', 'Location', etc., or 'O' for tokens that are not named entities).

CRFs make their predictions based on the current token, features of nearby tokens, and predictions for previous tokens. This allows them to capture the dependencies between labels in a sequence. For instance, in a sentence, if the current token is labeled as 'B-Person' (beginning of a Person entity), a CRF can leverage this information to influence the prediction for the next token, which is likely to be 'I-Person' (inside a Person entity) or 'O' (not a named entity).

The training of a CRF involves learning weights for different feature functions, which can include various characteristics of the current token and its context, such as the token itself, its prefix or suffix, its capitalization pattern, the words around it, etc. The objective is to learn weights that maximize the likelihood of the correct labels in the training data.

Benefits and Considerations
CRFs offer several benefits for NER:

Contextual Understanding: CRFs consider the entire sequence when making predictions, allowing them to model the context and dependencies between labels in a sequence.

Feature Flexibility: CRFs can incorporate a wide variety of arbitrary, non-independent features, making them highly flexible.

Boundary Detection: CRFs are good at predicting the boundaries of named entities, as they consider the transition features between labels.

However, training CRFs, particularly on large datasets, can be computationally intensive as it involves global normalization over the entire sequence. Additionally, feature engineering can be a manual and time-consuming process.

Despite these challenges, CRFs remain a popular choice for NER due to their ability to model the context and dependencies between labels in a sequence, providing a powerful tool for sequence labeling tasks in NLP.


# Readings
[An introduction to conditional random fields](https://www.nowpublishers.com/article/DownloadSummary/MAL-013)


[Accurate Information Extraction from Research Papers using Conditional Random Fields](https://aclanthology.org/N04-1042.pdf)

[Named entity recognition](https://www.researchgate.net/profile/Muhammad-Yasar-2/post/What_are_the_methods_or_tools_needed_to_study_language_processing/attachment/59d61dde79197b807797b60e/AS%3A273770000912399%401442283262685/download/Natural+Language+Processing+of+Semitic+Languages.pdf#page=241)

# Code example

