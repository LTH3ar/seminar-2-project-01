download data from: https://github.com/nlbse2024/issue-report-classification

This repo is a competition toolkit for classifying GitHub issue reports into three categories: bug, feature, or question. It provides a dataset of 3,000 labeled issues from 5 open-source projects (React, TensorFlow, VSCode, Bitcoin, OpenCV), split 50/50 into train/test sets. (1500 for train, 1500 for test/val)

Notebook breakdown:
1.Extracts the dataset from GitHub API using PyGithub. Collects 200 issues per class per repo (600 per repo, 3000 total), normalizes labels, splits into train/test, saves to CSV.
     
2.SetFit baseline (few-shot). Uses all-mpnet-base-v2 sentence-transformer. Concatenates title+body as text. Trains 1 SetFit classifier per repo (750 examples each, 1 epoch, 20 contrastive iterations). Achieves 0.8270 cross-repo F1.

3.RoBERTa baseline (fine-tuning). Uses roberta-base. Applies text cleaning (remove code blocks, links, digits, special chars). Fine-tunes for 10 epochs per repo with validation. Achieves lower F1 than SetFit on some repos.

4.fastText baseline. Converts data to fastText format (__label__ prefix). Trains train_supervised for 100 epochs per repo. Lightweight but lowest performance (~0.7184 cross-repo F1).
