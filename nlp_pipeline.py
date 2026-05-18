"""
NLP Sentiment Analysis Pipeline - GUI-Ready Functions
======================================================
Milestone 1 : Data Collection & Preprocessing
Milestone 2 : Text Preprocessing & Feature Engineering
Milestone 3 : Model Training & Evaluation  (Naive Bayes + SVM)
Milestone 4 : BERT / DistilBERT Fine-tuning
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import re
import time

import emoji
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob


# ══════════════════════════════════════════════════════════════════
# MILESTONE 1 — Data Collection & Preprocessing
# ══════════════════════════════════════════════════════════════════

def download_nltk_resources():
    """
    تحميل الموارد المطلوبة من NLTK

    يجب تشغيلها مرة واحدة قبل أي خطوة أخرى.
    """
    resources = [
        'punkt', 'stopwords',
        'averaged_perceptron_tagger',
        'wordnet', 'omw-1.4',
        'punkt_tab',
        'averaged_perceptron_tagger_eng'
    ]
    for r in resources:
        nltk.download(r, quiet=True)
    print("NLTK resources downloaded.")


def load_dataset(csv_path="training.csv", sample_per_class=25000,
                 random_state=42):
    
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(csv_path, encoding="latin-1", names=columns)

    pos_val = 4 if 4 in df['sentiment'].values else 'positive'
    neg_val = 0 if 0 in df['sentiment'].values else 'negative'

    df_pos = df[df["sentiment"] == pos_val].sample(
        sample_per_class, random_state=random_state)
    df_neg = df[df["sentiment"] == neg_val].sample(
        sample_per_class, random_state=random_state)

    df = pd.concat([df_pos, df_neg])
    df["sentiment"] = df["sentiment"].replace({0: "negative", 4: "positive"})
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("Dataset loaded successfully!")
    print(df["sentiment"].value_counts())
    return df


def inspect_dataframe(df):
    # بنعمل نسخة عشان منبوظش الداتا الأصلية
    temp_df = df.copy()
    
    # أي عمود فيه List بنحوله لـ String عشان الـ duplicated يشتغل صح
    for col in temp_df.columns:
        if temp_df[col].apply(lambda x: isinstance(x, list)).any():
            temp_df[col] = temp_df[col].astype(str)
            
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "nulls": df.isnull().sum().to_dict(),
        "duplicates": int(temp_df.duplicated().sum()), # بنحسب من النسخة الـ temp
    }
    return info


def drop_duplicates(df):
  
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(df)} duplicates. New shape: {df.shape}")
    return df


def plot_sentiment_distribution(df):
  
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df, ax=ax)
    ax.set_title("Sentiment Distribution")
    plt.tight_layout()
    return fig


def plot_text_length_distribution(df, text_col="text"):
  
    lengths = df[text_col].apply(len)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(lengths, bins=50, ax=ax)
    ax.set_title("Review Length Distribution")
    ax.set_xlabel("Length")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# MILESTONE 2 — Text Preprocessing & Feature Engineering
# ══════════════════════════════════════════════════════════════════

# ── Contractions Dictionary ────────────────────────────────────────
CONTRACTIONS = {
    "don't": "do not",
    "i'm": "i am",
    "you're": "you are",
    "can't": "cannot",
    "it's": "it is",
    "won't": "will not",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "wasn't": "was not",
    "didn't": "did not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
}


def clean_text(text):
 
    text = str(text)
    text = re.sub(r'<.*?>', '', text)                 # HTML Tags
    text = re.sub(r'http\S+|www\S+', '', text)        # URLs
    text = re.sub(r'@\S+|#\S+', '', text)             # Mentions & Hashtags
    text = re.sub(r'\d+', '', text)                   # Numbers
    text = emoji.replace_emoji(text, replace='')      # Emojis
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # Punctuation
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()          # Extra Spaces
    return text


def expand_contractions(text, contractions=None):
   
    if contractions is None:
        contractions = CONTRACTIONS
    for key, value in contractions.items():
        text = text.replace(key, value)
    return text


def apply_text_cleaning(df, text_col="text"):
  
    df['clean_text'] = df[text_col].apply(clean_text)
    df['clean_text'] = df['clean_text'].apply(expand_contractions)
    print("Text cleaning applied.")
    return df


def tokenize_words(df, text_col="clean_text"):
   
    df['tokens'] = df[text_col].apply(word_tokenize)
    return df


def tokenize_sentences(df, text_col="clean_text"):
   
    df['sentences'] = df[text_col].apply(sent_tokenize)
    return df


def remove_stopwords(df, token_col="tokens"):
  
    stop_words = set(stopwords.words('english'))
    df['filtered_tokens'] = df[token_col].apply(
        lambda tokens: [w for w in tokens if w not in stop_words]
    )
    return df


def extract_ngrams(df, token_col="filtered_tokens", n=2):
    
    col_name = f"{'uni' if n == 1 else 'bi' if n == 2 else 'tri'}grams"
    df[col_name] = df[token_col].apply(
        lambda x: list(ngrams(x, n))
    )
    return df, col_name


def pos_tagging(df, token_col="filtered_tokens"):
   
    df['pos_tags'] = df[token_col].apply(nltk.pos_tag)
    return df


def extract_opinion_words(df, pos_col="pos_tags"):
    
    def _extract(pos_tags):
        return [word for word, tag in pos_tags
                if tag.startswith('JJ') or tag.startswith('RB')]

    df['opinion_words'] = df[pos_col].apply(_extract)
    return df


def get_textblob_sentiment(text):
  
    polarity = TextBlob(text).sentiment.polarity
    sentiment = "Positive" if polarity >= 0 else "Negative"
    return sentiment, polarity


def apply_stemming(df, token_col="filtered_tokens"):
  
    stemmer = PorterStemmer()
    df['stemmed'] = df[token_col].apply(
        lambda tokens: [stemmer.stem(w) for w in tokens]
    )
    return df


def apply_lemmatization(df, token_col="filtered_tokens"):
   
    lemmatizer = WordNetLemmatizer()
    df['lemmatized'] = df[token_col].apply(
        lambda tokens: [lemmatizer.lemmatize(w) for w in tokens]
    )
    return df


# ══════════════════════════════════════════════════════════════════
# MILESTONE 3 — Model Training & Evaluation
# ══════════════════════════════════════════════════════════════════

def build_tfidf_features(df, text_col="clean_text",
                          label_col="sentiment",
                          test_size=0.2, random_state=42):
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df[text_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, tfidf


def train_naive_bayes(X_train, y_train):
    
    nb_model = MultinomialNB()
    start = time.time()
    nb_model.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)
    print(f"Naive Bayes Training Time: {train_time}s")
    return nb_model, train_time


def train_svm(X_train, y_train, kernel='linear', C=1.0):
   
    svm_model = SVC(kernel=kernel, C=C)
    start = time.time()
    svm_model.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)
    print(f"SVM Training Time: {train_time}s")
    return svm_model, train_time


def evaluate_model(model, X_test, y_test, model_name="Model"):
   
    start = time.time()
    preds = model.predict(X_test)
    pred_time = round(time.time() - start, 4)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Accuracy: {acc:.4f}")
    print(f"Prediction Time: {pred_time}s")
    print("Classification Report:\n", report)
    return preds, acc, pred_time, report


def plot_accuracy_comparison(nb_acc, svm_acc,
                              nb_train_time, svm_train_time):
   
    comparison_df = pd.DataFrame({
        'Model':           ['Naive Bayes', 'SVM'],
        'Accuracy':        [nb_acc, svm_acc],
        'Training Time(s)':[nb_train_time, svm_train_time]
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.barplot(x='Model', y='Accuracy',
                data=comparison_df, hue='Model',
                legend=False, palette='viridis', ax=axes[0])
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylim(0, 1.0)

    sns.barplot(x='Model', y='Training Time(s)',
                data=comparison_df, hue='Model',
                legend=False, palette='magma', ax=axes[1])
    axes[1].set_title("Training Time Comparison")

    plt.tight_layout()
    return fig, comparison_df


def plot_confusion_matrices(y_test, nb_preds, svm_preds):
   
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(confusion_matrix(y_test, nb_preds),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Naive Bayes Confusion Matrix')

    sns.heatmap(confusion_matrix(y_test, svm_preds),
                annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('SVM Confusion Matrix')

    plt.tight_layout()
    return fig


def predict_sentiment(text, tfidf, model):
    """
    التنبؤ بمشاعر نص جديد باستخدام نموذج مدرب

    Parameters
    ----------
    text  : النص المراد تحليله
    tfidf : الـ TfidfVectorizer المدرب
    model : النموذج المدرب (NB أو SVM)

    Returns
    -------
    prediction : "positive" أو "negative"
    """
    vector = tfidf.transform([text])
    prediction = model.predict(vector)
    return prediction[0]


# ══════════════════════════════════════════════════════════════════
# MILESTONE 4 — BERT / DistilBERT Fine-tuning
# ══════════════════════════════════════════════════════════════════

def prepare_bert_datasets(df, text_col="clean_text",
                           label_col="sentiment",
                           test_size=0.2, random_state=42,
                           max_length=128):
   
    from datasets import Dataset
    from transformers import AutoTokenizer

    # تحويل التسميات لأرقام
    data = df[[text_col, label_col]].copy()
    data[label_col] = data[label_col].replace(
        {'positive': 1, 'negative': 0}
    ).astype(int)

    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    train_df = train_df.rename(
        columns={text_col: 'text', label_col: 'label'}
    ).reset_index(drop=True)
    test_df = test_df.rename(
        columns={text_col: 'text', label_col: 'label'}
    ).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset  = Dataset.from_pandas(test_df)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_test  = test_dataset.map(tokenize_fn, batched=True)

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test  = tokenized_test.remove_columns(["text"])

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    print("BERT datasets ready!")
    return tokenized_train, tokenized_test, tokenizer


def load_bert_model(model_name="distilbert-base-uncased", num_labels=2):
   
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    print(f"Model '{model_name}' loaded.")
    return model


def train_bert_model(model, tokenizer,
                     tokenized_train, tokenized_test,
                     output_dir="./results",
                     epochs=5, batch_size=16,
                     learning_rate=2e-5,
                     early_stopping_patience=2,
                     use_fp16=True):
  
    import torch
    from transformers import (EarlyStoppingCallback, Trainer,
                               TrainingArguments)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=use_fp16 and torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience
        )]
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"DistilBERT Final Validation Accuracy:"
          f" {results['eval_accuracy']:.4f}")
    return trainer, results


def save_bert_model(model, tokenizer, save_dir="./my_best_model"):
  
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to '{save_dir}'")


def evaluate_bert_model(trainer):
   
    results = trainer.evaluate()
    print(f"BERT Validation Accuracy: {results['eval_accuracy']:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════
# Full Pipeline (helper للـ GUI)
# ══════════════════════════════════════════════════════════════════

def run_classic_pipeline(csv_path="training.csv",
                          sample_per_class=25000):
  
    result = {}

    download_nltk_resources()

    # ── M1: Load & Inspect ───────────────────
    df = load_dataset(csv_path, sample_per_class)
    df = drop_duplicates(df)

    # ── M2: Text Processing ──────────────────
    df = apply_text_cleaning(df)
    df = tokenize_words(df)
    df = remove_stopwords(df)
    df, _ = extract_ngrams(df, n=2)
    df = pos_tagging(df)
    df = extract_opinion_words(df)
    df = apply_stemming(df)
    df = apply_lemmatization(df)

    result["df"] = df

    # ── M3: Models ───────────────────────────
    X_train, X_test, y_train, y_test, tfidf = build_tfidf_features(df)

    nb_model,  nb_train_time  = train_naive_bayes(X_train, y_train)
    svm_model, svm_train_time = train_svm(X_train, y_train)

    nb_preds, nb_acc,  nb_pred_time,  _ = evaluate_model(
        nb_model,  X_test, y_test, "Naive Bayes")
    svm_preds, svm_acc, svm_pred_time, _ = evaluate_model(
        svm_model, X_test, y_test, "SVM")

    result.update({
        "tfidf":          tfidf,
        "nb_model":       nb_model,
        "svm_model":      svm_model,
        "nb_acc":         nb_acc,
        "svm_acc":        svm_acc,
        "nb_train_time":  nb_train_time,
        "svm_train_time": svm_train_time,
        "nb_preds":       nb_preds,
        "svm_preds":      svm_preds,
        "y_test":         y_test,
    })

    print("\n✅ Classic Pipeline Completed!")
    print(f"  Naive Bayes Accuracy : {nb_acc:.4f}")
    print(f"  SVM Accuracy         : {svm_acc:.4f}")
    return result
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def load_nlp_system(model_path="./my_best_model"):
    """
    تحميل موديل DistilBERT والـ Tokenizer المحفوظين
    """
    # 1. تحميل الـ Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. تحميل الموديل
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 3. إنشاء الـ Pipeline للتحليل الفوري
    nlp_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    return nlp_classifier

def analyze_sentiment(classifier, raw_text):
    """
    توقع المشاعر للنص المترجم
    """
    # الموديل بيعمل التوقع فوراً
    result = classifier(raw_text)[0]
    
    return {
        "text": raw_text,
        "label": result['label'],
        "score": result['score']
    }