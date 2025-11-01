# Sentiment and Sarcasm Detection using RoBERTa

## Overview
This project performs sentiment analysis and sarcasm detection using transformer-based language models. It combines a custom fine-tuned RoBERTa model for sentiment classification with a pretrained CardiffNLP RoBERTa model for sarcasm detection. The goal is to understand both the emotional tone and sarcastic intent of user reviews or social media text.

---

## Project Structure
Sentiment_Sarcasm_Project/
│
├── app.py # Streamlit application entry point
├── sentiment_model/ # Fine-tuned sentiment model files
├── sarcasm_model/ # Pretrained sarcasm detection model
├── dataset/ # Training dataset
├── train_sentiment.py # Fine-tuning script for sentiment model
├── utils.py # Helper functions for preprocessing and prediction
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Dataset Description
The dataset used for training the sentiment model is derived from Amazon product reviews. Each record includes metadata and text fields. The main columns are:

| Column | Description |
|---------|-------------|
| rating | Numeric star rating (1–5), used as the sentiment label |
| title | Short title of the review |
| text | Main review content |
| images | URLs or paths to product images (not used in this model) |
| asin | Amazon product ID |
| parent_asin | Parent product group ID |
| user_id | Reviewer identifier |
| timestamp | Review posting date and time |
| helpful_vote | Number of helpful votes received |
| verified_purchase | Indicates if the reviewer purchased the product |

During preprocessing, the `title` and `text` columns are combined for input, and the `rating` column is used as the sentiment label.

---

## Models Used

### Sentiment Model
- Base Model: `roberta-base`
- Fine-tuned on: Amazon Reviews dataset
- Task: 5-class sentiment classification (very negative to very positive)
- Loss Function: CrossEntropyLoss

### Sarcasm Model
- Model: `cardiffnlp/twitter-roberta-base-irony`
- Fine-tuned on: SemEval-2018 Task 3 (Irony Detection in English Tweets)
- Task: Binary classification (ironic or non-ironic)

---

