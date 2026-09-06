# Email Spam Detection with Machine Learning

**Program:** Oasis Infobyte SIP — AICTE OIBSIP
**Track:** Data Science
**Task:** Task 4 — Email Spam Detection with Machine Learning

## Objective

Build a Natural Language Processing (NLP) binary classifier that distinguishes spam
messages from legitimate (ham) messages, using TF-IDF feature extraction and two
classical machine learning models.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) —
5,572 SMS messages labeled `ham` or `spam` (5,169 after removing duplicates). It's the
standard public dataset for this exact task and generalizes directly to email spam
filtering, since the underlying problem — classifying short unstructured text as
wanted/unwanted — is the same.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, metrics)
- NLTK (stopword removal)
- matplotlib, seaborn, WordCloud
- Jupyter Notebook

## Pipeline

1. **Load & inspect** — check shape, drop duplicates, confirm class distribution
   (~87% ham / ~13% spam — imbalanced, as real message traffic typically is).
2. **Text preprocessing** — lowercase, strip punctuation/digits, tokenize, remove
   English stopwords.
3. **Feature extraction** — TF-IDF vectorization (unigrams + bigrams, 3,000 features).
4. **Train/test split** — 80/20, stratified to preserve the class ratio.
5. **Model training** — Multinomial Naive Bayes (text-classification baseline) and
   Logistic Regression (`class_weight='balanced'`), trained side by side.
6. **Evaluation** — accuracy, precision, recall, F1, confusion matrix for each model.
7. **Discussion** — why recall matters more than raw accuracy for spam detection.
8. **Bonus** — WordCloud visualization of the vocabulary distinctive to each class.

## Results

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Multinomial Naive Bayes | 97.29% | 0.99 | 0.79 | 0.88 |
| Logistic Regression | 97.20% | 0.89 | 0.89 | 0.89 |

Both models score similarly on raw accuracy, but they fail differently: Naive Bayes is
more conservative (very few false alarms, but misses ~21% of actual spam), while
Logistic Regression catches noticeably more spam (11% missed) at the cost of a few more
false positives. See the notebook's Section 8 for the full discussion of why **recall**
is the metric to optimize for in a spam filter — a missed spam message (false negative)
is a materially worse outcome than an occasional legitimate message flagged for review
(false positive).

## Key Visuals

- `screenshots/01_class_distribution.png` — ham vs spam class balance
- `screenshots/02_message_length_distribution.png` — spam messages skew longer
- `screenshots/03_confusion_matrices.png` — side-by-side model error breakdown
- `screenshots/04_wordclouds.png` — vocabulary that distinguishes spam from ham

## How to Run

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook Email_Spam_Detection.ipynb
```

## Files

```
OIBSIP_Python_Task4_EmailSpamDetection/
├── Email_Spam_Detection.ipynb   # Full notebook, executed with outputs
├── spam.csv                     # Dataset (SMS Spam Collection)
├── README.md                    # This file
└── screenshots/                 # Exported chart images for quick viewing
```

## Author

*Submitted as part of the AICTE OIBSIP Python Programming track, Task 4.*
