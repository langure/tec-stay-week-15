import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import requests
import zipfile
import os

# Function to download and unzip the GMB dataset


def download_gmb_dataset():
    url = 'https://gmb.let.rug.nl/releases/gmb-2.2.0.zip'
    file_path = 'gmb-2.2.0.zip'
    if not os.path.exists(file_path):
        print("Downloading GMB dataset...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Download completed.")

    if not os.path.exists('gmb-2.2.0'):
        print("Unzipping GMB dataset...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Unzip completed.")


# Download and unzip the GMB dataset
download_gmb_dataset()

# Load the GMB dataset for English NER


def load_gmb_data(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.split(' ')
                token = parts[0]
                label = parts[3]
                sentence.append((token, label))
    return sentences


# Load the GMB dataset for English NER
file_path = 'gmb-2.2.0/data.gm'
sentences = load_gmb_data(file_path)

# Feature extraction function for CRF


def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word': word,
        'is_first_word': i == 0,
        'is_last_word': i == len(sent) - 1,
        'is_numeric': word.isdigit(),
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if i == 0 else sent[i - 1][0],
        'next_word': '' if i == len(sent) - 1 else sent[i + 1][0],
        'prev_word_is_numeric': False if i == 0 else sent[i - 1][0].isdigit(),
        'next_word_is_numeric': False if i == len(sent) - 1 else sent[i + 1][0].isdigit(),
        'label': sent[i][1]
    }
    return features


# Convert dataset to features and labels
X = [[word2features(s, i) for i in range(len(s))] for s in sentences]
y = [[label for token, label in s] for s in sentences]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Evaluate the CRF model
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred))
