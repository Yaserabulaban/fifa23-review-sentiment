# Sentiment Analysis and Opinion Mining of FIFA 23 Steam Reviews

This repository contains a Social Media Computing assignment notebook for sentiment analysis and opinion mining on FIFA 23 Steam reviews. The notebook builds an end-to-end NLP workflow covering data cleaning, exploratory data analysis, text preprocessing, hypothesis testing, traditional machine learning, a trained BiLSTM deep learning model, transformer-based sentiment comparison, VADER-derived multi-class sentiment modelling, opinion mining, aspect-based sentiment analysis, and saved outputs for report writing.

## Dataset Description

The dataset file is `dataset/fifa23_steam_reviews.csv`. It contains 25,737 Steam review records with 16 original columns:

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

Because `voted_up` is binary, the supervised task is also binary. Steam does not provide a reliable neutral ground-truth label, so neutral sentiment is not used as a supervised class. VADER is still reported descriptively with Positive, Neutral, and Negative labels, then converted to binary for direct comparison with `voted_up`.

The notebook also includes a separate multi-class experiment using VADER-derived weak labels. This additional experiment predicts written-sentiment categories (`Negative`, `Neutral`, `Positive`) created from VADER compound score thresholds. These labels are not manual human annotations and should not be described as true ground truth; they are used only to compare model behaviour when a neutral class is included.

## How to Run the Notebook in Colab

1. Open `notebook/tutorial_1_smc.ipynb` in Google Colab.
2. Keep `requirements.txt` in the repository root and keep the dataset at `dataset/fifa23_steam_reviews.csv`. If running outside the repository, upload the CSV manually when prompted.
3. Run the first setup cell. It installs dependencies from `requirements.txt`. If `requirements.txt` is missing, it installs the core notebook packages directly.
4. Select `Runtime > Restart and run all`.
5. After execution, review generated tables in `outputs/` and figures in `outputs/figures/`.

## Required Files

- `notebook/tutorial_1_smc.ipynb`: Main analysis notebook.
- `requirements.txt`: Python dependencies required to run the notebook.
- `dataset/fifa23_steam_reviews.csv`: Raw FIFA 23 Steam reviews dataset.
- `outputs/`: Generated CSV result files.
- `outputs/figures/`: Generated PNG charts for the report.
- `report/`: Word report draft/final report files.

## Outputs Folder

The `outputs/` folder contains report-ready CSV files, including:

- `final_cleaned_fifa23_reviews.csv`: Cleaned dataset with raw and processed review text.
- `model_results.csv`: Traditional ML model metrics.
- `combined_model_results.csv`: Traditional ML and transformer metrics in one table.
- `combined_model_results_with_deep_learning.csv`: Traditional ML, trained BiLSTM deep learning, and transformer metrics in one table.
- `deep_learning_bilstm_results.csv`: Metrics for the trained BiLSTM neural network.
- `deep_learning_bilstm_training_history.csv`: Training and validation loss/accuracy history for the BiLSTM model.
- `deep_learning_bilstm_predictions.csv`: BiLSTM test-set predictions and positive-class probabilities.
- `hypothesis_results.csv`: Statistical hypothesis test results and interpretations.
- `vader_3class_distribution.csv`: Descriptive VADER Positive/Neutral/Negative distribution before binary conversion.
- `vader_binary_distribution.csv`: VADER distribution after conversion to Positive/Negative for comparison with `voted_up`.
- `multiclass_vader_label_distribution.csv`: VADER-derived weak-label distribution used for the additional multi-class experiment.
- `multiclass_model_results.csv`: Accuracy, macro metrics, and weighted metrics for multi-class ML and BiLSTM models.
- `multiclass_model_results_detailed.csv`: Per-class precision, recall, F1-score, and support for each multi-class model.
- `multiclass_bilstm_training_history.csv`: Multi-class BiLSTM training and validation history.
- `multiclass_bilstm_predictions.csv`: Multi-class BiLSTM predictions and class probabilities.
- `multiclass_error_examples.csv`: Example errors from the best multi-class model.
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

The `outputs/figures/` folder contains PNG visualizations such as sentiment distribution, monthly review trends, review-length distribution, metadata box/violin plots, correlation heatmap, word clouds, TF-IDF term charts, confusion matrices, model comparison charts, BiLSTM training history, VADER 3-class and binary distributions, multi-class model comparison charts, transformer confidence analysis, opinion-word and opinion-phrase charts, aspect sentiment counts, and aspect polarity heatmaps.

## Main Model Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8566 | 0.8403 | 0.8534 | 0.8468 |
| Linear SVM | 0.8414 | 0.8193 | 0.8448 | 0.8318 |
| BiLSTM Neural Network | 0.8404 | 0.8479 | 0.7997 | 0.8231 |
| Random Forest | 0.8155 | 0.7802 | 0.8389 | 0.8085 |
| Multinomial Naive Bayes | 0.7793 | 0.8621 | 0.6246 | 0.7244 |
| Transformer DistilBERT SST-2 | 0.7565 | 0.8184 | 0.6114 | 0.6999 |

Logistic Regression achieved the strongest overall performance by F1-score in the saved results. The trained BiLSTM neural network provides a true deep learning model trained on the FIFA 23 review dataset and achieved an F1-score of 0.8231. It underperformed Logistic Regression but outperformed Random Forest, Multinomial Naive Bayes, and the pretrained DistilBERT SST-2 benchmark. The transformer section is retained as a practical pretrained benchmark, with additional confidence and mismatch examples to explain why a general sentiment model may disagree with Steam recommendation labels.

## Additional Sentiment Outputs

VADER's descriptive 3-class distribution is:

| VADER Label | Review Count | Percentage |
|---|---:|---:|
| Positive | 7,167 | 35.75% |
| Neutral | 6,224 | 31.04% |
| Negative | 6,659 | 33.21% |

For H4, VADER is converted to binary using the documented rule `compound >= 0` = Positive and `compound < 0` = Negative, because the Steam `voted_up` label has no neutral class.

After binary conversion, the VADER distribution is:

| Binary VADER Label | Review Count | Percentage |
|---|---:|---:|
| Positive | 13,295 | 66.31% |
| Negative | 6,755 | 33.69% |

## Multi-Class Sentiment Experiment

The additional multi-class experiment uses VADER compound score thresholds to create weak labels:

- `Negative`: compound score `<= -0.05`
- `Neutral`: compound score between `-0.05` and `0.05`
- `Positive`: compound score `>= 0.05`

The resulting weak-label distribution is:

| Derived Label | Review Count | Percentage |
|---|---:|---:|
| Negative | 6,659 | 33.21% |
| Neutral | 6,224 | 31.04% |
| Positive | 7,167 | 35.75% |

The multi-class models were evaluated with macro F1-score as the main metric because it treats each class equally:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Multiclass Logistic Regression | 0.8511 | 0.8511 | 0.8538 | 0.8511 | 0.8503 |
| Multiclass Linear SVM | 0.8474 | 0.8473 | 0.8496 | 0.8473 | 0.8463 |
| Multiclass BiLSTM Neural Network | 0.8429 | 0.8470 | 0.8439 | 0.8450 | 0.8436 |
| Multiclass Random Forest | 0.7890 | 0.7928 | 0.7938 | 0.7883 | 0.7872 |
| Multiclass Multinomial Naive Bayes | 0.6845 | 0.7252 | 0.6745 | 0.6630 | 0.6670 |

Logistic Regression achieved the best multi-class macro F1-score. The multi-class BiLSTM performed competitively but did not outperform the strongest TF-IDF models. In per-class results, the neutral category was handled well by Logistic Regression, Linear SVM, and BiLSTM, while Multinomial Naive Bayes had much weaker neutral recall.

## Future Work

Future improvements could include fine-tuning a transformer model directly on the FIFA 23 Steam review labels, experimenting with stronger deep learning architectures such as GRU/CNN hybrids, expanding the aspect dictionary with more football-specific and EA-specific terms, adding sarcasm and mixed-opinion detection, comparing results across multiple FIFA/EA Sports FC releases, and using manual annotation on a review sample to evaluate how well `voted_up` represents written sentiment.
