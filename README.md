# OIBSIP — Oasis Infobyte Summer Internship Program

**Track:** Data Science / Python Programming
**Program:** AICTE OIBSIP

This repository contains my completed tasks for the Oasis Infobyte Summer Internship
Program. Each task lives in its own folder with a fully executed Jupyter Notebook, the
dataset used, a task-specific README, and exported chart screenshots.

## Completed Tasks

| # | Task | Summary | Key Result |
|---|---|---|---|
| 1 | [Email Spam Detection](./Python-Task4-EmailSpamDetection) | NLP binary classifier (TF-IDF + Naive Bayes / Logistic Regression) distinguishing spam from legitimate SMS/email text. | Logistic Regression caught 88.6% of spam (recall) vs. Naive Bayes' 79.4% — discussed why recall matters more than raw accuracy for spam filtering. |
| 2 | [Car Price Prediction](./Python-Task3-CarPricePrediction) | Regression model predicting used-car resale price from age, mileage, fuel type, and original showroom price. | Linear Regression (R² = 0.75) outperformed Random Forest (R² = 0.51) — a small dataset with one dominant linear driver didn't reward the extra model complexity. |
| 3 | [Sales Prediction](./Python-Task5-SalesPrediction) | Regression model predicting product sales from TV/Radio/Newspaper advertising spend. | Random Forest (R² = 0.98) ranked TV as the top overall driver, but Linear Regression coefficients showed Radio had the better *per-dollar* return — the two models answer different questions. |

## Tech Stack

Python 3 · pandas · numpy · scikit-learn · matplotlib · seaborn · NLTK · WordCloud ·
Jupyter Notebook

## Repository Structure

```
OIBSIP/
├── Python-Task3-CarPricePrediction/
│   ├── Car_Price_Prediction.ipynb
│   ├── car_data.csv
│   ├── README.md
│   └── screenshots/
├── Python-Task4-EmailSpamDetection/
│   ├── Email_Spam_Detection.ipynb
│   ├── spam.csv
│   ├── README.md
│   └── screenshots/
├── Python-Task5-SalesPrediction/
│   ├── Sales_Prediction.ipynb
│   ├── advertising.csv
│   ├── README.md
│   └── screenshots/
└── README.md   (this file)
```

Each task folder is self-contained — open its notebook directly, or check its own
`README.md` for that task's objective, dataset, pipeline, and results in detail.

## How to Run Any Task

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter
jupyter notebook
```
Then open the `.ipynb` file inside the relevant task folder and run all cells. (Or upload
the notebook + its dataset to Google Colab and use Run All — no local setup needed.)

## About the Program

Oasis Infobyte's SIP pairs interns with self-sourced project tasks across tracks like
Data Science, Web Development, Android, and Cybersecurity. More info: [oasisinfobyte.com](http://www.oasisinfobyte.com)

## Author

**NotSaran** — Data Science Track, AICTE OIBSIP
