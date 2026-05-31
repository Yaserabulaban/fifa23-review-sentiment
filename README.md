# Sentiment Analysis and Opinion Mining of FIFA 23 Steam Reviews

This repository contains a Social Media Computing assignment notebook for sentiment analysis and opinion mining on FIFA 23 Steam reviews. The notebook builds an end-to-end NLP workflow covering data cleaning, exploratory data analysis, text preprocessing, hypothesis testing, traditional machine learning, transformer-based sentiment comparison, opinion mining, aspect-based sentiment analysis, and saved outputs for report writing.

## Dataset Description

The dataset file is `fifa23_steam_reviews.csv`. It contains 25,737 Steam review records with 16 original columns:

- `id`
- `language`
- `review`
- `created`
- `voted_up`
- `votes_up`
- `comment_count`
- `steam_purchase`
- `recieved_for_free`
- `written_during_early_access`
- `author_num_games_owned`
- `author_num_reviews`
- `author_playtime_forever`
- `author_playtime_last_two_weeks`
- `author_playtime_at_review`
- `author_last_played`

The main supervised sentiment label is `voted_up`:

- `True` = Positive sentiment / recommended review
- `False` = Negative sentiment / not recommended review

The notebook removes missing or empty review text, removes exact duplicate review texts, converts date columns, creates sentiment labels, creates review-length features, and converts playtime minutes into hours for interpretation.

## How to Run the Notebook in Colab

1. Open `tutorial_1_smc.ipynb` in Google Colab.
2. Upload `requirements.txt` and `fifa23_steam_reviews.csv` to the same Colab session or keep them in the repository root when opening from GitHub.
3. Run the first setup cell. It installs dependencies from `requirements.txt`. If `requirements.txt` is missing, it installs the core notebook packages directly.
4. Select `Runtime > Restart and run all`.
5. After execution, review generated tables in `outputs/` and figures in `outputs/figures/`.

## Required Files

- `tutorial_1_smc.ipynb`: Main analysis notebook.
- `requirements.txt`: Python dependencies required to run the notebook.
- `fifa23_steam_reviews.csv`: Raw FIFA 23 Steam reviews dataset.
- `outputs/`: Generated CSV result files.
- `outputs/figures/`: Generated PNG charts for the report.

## Outputs Folder

The `outputs/` folder contains report-ready CSV files, including:

- `final_cleaned_fifa23_reviews.csv`: Cleaned dataset with raw and processed review text.
- `model_results.csv`: Traditional ML model metrics.
- `combined_model_results.csv`: Traditional ML and transformer metrics in one table.
- `hypothesis_results.csv`: Statistical hypothesis test results and interpretations.
- `aspect_sentiment_results.csv`: Aspect-based sentiment counts and percentages.
- `aspect_mentions.csv`: Review-level aspect mentions with matched keywords and VADER polarity.
- `sample_error_analysis.csv`: False positive and false negative examples.
- `transformer_sample_predictions.csv`: Transformer sentiment predictions for the sampled reviews.
- `transformer_confidence_summary.csv`: Transformer confidence bands split by whether predictions match `voted_up`.
- `transformer_error_examples.csv`: High-confidence transformer mismatches for discussion.
- `cross_validation_results.csv`: Cross-validation results for selected traditional models.
- `aspect_example_reviews.csv`: Example reviews for detected aspects.
- `positive_opinion_words.csv` and `negative_opinion_words.csv`: Frequent opinion adjectives.
- `positive_opinion_phrases.csv` and `negative_opinion_phrases.csv`: Simple adjective-noun opinion phrases.
- `strong_positive_examples.csv` and `strong_negative_examples.csv`: Reviews with strong VADER polarity.

The `outputs/figures/` folder contains PNG visualizations such as sentiment distribution, monthly review trends, review-length distribution, metadata box/violin plots, correlation heatmap, word clouds, TF-IDF term charts, confusion matrices, model comparison charts, transformer confidence analysis, opinion-word and opinion-phrase charts, aspect sentiment counts, and aspect polarity heatmaps.

## Main Model Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8566 | 0.8403 | 0.8534 | 0.8468 |
| Linear SVM | 0.8414 | 0.8193 | 0.8448 | 0.8318 |
| Random Forest | 0.8155 | 0.7802 | 0.8389 | 0.8085 |
| Multinomial Naive Bayes | 0.7793 | 0.8621 | 0.6246 | 0.7244 |
| Transformer DistilBERT SST-2 | 0.7565 | 0.8184 | 0.6114 | 0.6999 |

Logistic Regression achieved the strongest overall traditional ML performance by F1-score in the saved results. The transformer section is included as a practical pretrained deep-learning benchmark, with additional confidence and mismatch examples to explain why a general sentiment model may disagree with Steam recommendation labels.

## Future Work

Future improvements could include fine-tuning a transformer model directly on the FIFA 23 Steam review labels, expanding the aspect dictionary with more football-specific and EA-specific terms, adding sarcasm and mixed-opinion detection, comparing results across multiple FIFA/EA Sports FC releases, and using manual annotation on a review sample to evaluate how well `voted_up` represents written sentiment.
