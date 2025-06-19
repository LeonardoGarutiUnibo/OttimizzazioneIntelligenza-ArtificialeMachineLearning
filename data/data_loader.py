from datasets import load_dataset
from transformers import RobertaTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_tokenize_dataset(model_name: str, max_length: int):
    dataset = load_dataset("emotion")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets, dataset

def plot_class_distribution(dataset):
    labels = dataset['train']['label']
    sns.countplot(x=labels)
    plt.title("Distribuzione delle classi nel training set")
    plt.xlabel("Classe")
    plt.ylabel("Numero di campioni")
    plt.show()