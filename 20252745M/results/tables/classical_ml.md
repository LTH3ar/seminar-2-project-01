| model | evaluation | repository | sample_count | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | fold_macro_f1_mean | fold_macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Linear SVM | stratified_group_k_fold | bitcoin/bitcoin | 300 | 0.6400 | 0.6374 | 0.6400 | 0.6384 | 0.6384 | 0.6375 | 0.0211 |
| TF-IDF + Linear SVM | stratified_group_k_fold | facebook/react | 300 | 0.7600 | 0.7695 | 0.7600 | 0.7636 | 0.7636 | 0.7632 | 0.0398 |
| TF-IDF + Linear SVM | stratified_group_k_fold | microsoft/vscode | 300 | 0.6967 | 0.6997 | 0.6967 | 0.6975 | 0.6975 | 0.6955 | 0.0466 |
| TF-IDF + Linear SVM | stratified_group_k_fold | opencv/opencv | 300 | 0.7567 | 0.7592 | 0.7567 | 0.7572 | 0.7572 | 0.7564 | 0.0506 |
| TF-IDF + Linear SVM | stratified_group_k_fold | tensorflow/tensorflow | 300 | 0.7133 | 0.7195 | 0.7133 | 0.7156 | 0.7156 | 0.7112 | 0.0278 |
| TF-IDF + Linear SVM | stratified_group_k_fold | overall | 1500 | 0.7133 | 0.7171 | 0.7133 | 0.7145 | 0.7145 |  |  |
| TF-IDF + Logistic Regression | stratified_group_k_fold | bitcoin/bitcoin | 300 | 0.6500 | 0.6500 | 0.6500 | 0.6491 | 0.6491 | 0.6478 | 0.0207 |
| TF-IDF + Logistic Regression | stratified_group_k_fold | facebook/react | 300 | 0.7867 | 0.7967 | 0.7867 | 0.7900 | 0.7900 | 0.7896 | 0.0546 |
| TF-IDF + Logistic Regression | stratified_group_k_fold | microsoft/vscode | 300 | 0.7000 | 0.7049 | 0.7000 | 0.7016 | 0.7016 | 0.7010 | 0.0497 |
| TF-IDF + Logistic Regression | stratified_group_k_fold | opencv/opencv | 300 | 0.7433 | 0.7470 | 0.7433 | 0.7428 | 0.7428 | 0.7416 | 0.0524 |
| TF-IDF + Logistic Regression | stratified_group_k_fold | tensorflow/tensorflow | 300 | 0.7267 | 0.7366 | 0.7267 | 0.7296 | 0.7296 | 0.7248 | 0.0272 |
| TF-IDF + Logistic Regression | stratified_group_k_fold | overall | 1500 | 0.7213 | 0.7271 | 0.7213 | 0.7226 | 0.7226 |  |  |
| TF-IDF + Complement Naive Bayes | stratified_group_k_fold | bitcoin/bitcoin | 300 | 0.6633 | 0.6631 | 0.6633 | 0.6552 | 0.6552 | 0.6535 | 0.0156 |
| TF-IDF + Complement Naive Bayes | stratified_group_k_fold | facebook/react | 300 | 0.7600 | 0.7612 | 0.7600 | 0.7543 | 0.7543 | 0.7523 | 0.0772 |
| TF-IDF + Complement Naive Bayes | stratified_group_k_fold | microsoft/vscode | 300 | 0.6600 | 0.6672 | 0.6600 | 0.6588 | 0.6588 | 0.6578 | 0.0467 |
| TF-IDF + Complement Naive Bayes | stratified_group_k_fold | opencv/opencv | 300 | 0.6667 | 0.6633 | 0.6667 | 0.6580 | 0.6580 | 0.6591 | 0.0585 |
| TF-IDF + Complement Naive Bayes | stratified_group_k_fold | tensorflow/tensorflow | 300 | 0.7000 | 0.7090 | 0.7000 | 0.7014 | 0.7014 | 0.6981 | 0.0281 |
| TF-IDF + Complement Naive Bayes | stratified_group_k_fold | overall | 1500 | 0.6900 | 0.6928 | 0.6900 | 0.6855 | 0.6855 |  |  |
| TF-IDF + Random Forest | stratified_group_k_fold | bitcoin/bitcoin | 300 | 0.6233 | 0.6267 | 0.6233 | 0.6243 | 0.6243 | 0.6198 | 0.0557 |
| TF-IDF + Random Forest | stratified_group_k_fold | facebook/react | 300 | 0.7067 | 0.7195 | 0.7067 | 0.7103 | 0.7103 | 0.7076 | 0.0601 |
| TF-IDF + Random Forest | stratified_group_k_fold | microsoft/vscode | 300 | 0.6967 | 0.7022 | 0.6967 | 0.6977 | 0.6977 | 0.6982 | 0.0617 |
| TF-IDF + Random Forest | stratified_group_k_fold | opencv/opencv | 300 | 0.6833 | 0.6860 | 0.6833 | 0.6822 | 0.6822 | 0.6797 | 0.0455 |
| TF-IDF + Random Forest | stratified_group_k_fold | tensorflow/tensorflow | 300 | 0.8333 | 0.8514 | 0.8333 | 0.8331 | 0.8331 | 0.8327 | 0.0459 |
| TF-IDF + Random Forest | stratified_group_k_fold | overall | 1500 | 0.7087 | 0.7172 | 0.7087 | 0.7095 | 0.7095 |  |  |
| TF-IDF + Logistic Regression | official_holdout | bitcoin/bitcoin | 300 | 0.6733 | 0.6822 | 0.6733 | 0.6754 | 0.6754 |  |  |
| TF-IDF + Logistic Regression | official_holdout | facebook/react | 300 | 0.8200 | 0.8232 | 0.8200 | 0.8202 | 0.8202 |  |  |
| TF-IDF + Logistic Regression | official_holdout | microsoft/vscode | 300 | 0.6700 | 0.6812 | 0.6700 | 0.6700 | 0.6700 |  |  |
| TF-IDF + Logistic Regression | official_holdout | opencv/opencv | 300 | 0.7600 | 0.7618 | 0.7600 | 0.7570 | 0.7570 |  |  |
| TF-IDF + Logistic Regression | official_holdout | tensorflow/tensorflow | 300 | 0.8267 | 0.8351 | 0.8267 | 0.8285 | 0.8285 |  |  |
| TF-IDF + Logistic Regression | official_holdout | overall | 1500 | 0.7500 | 0.7567 | 0.7500 | 0.7502 | 0.7502 |  |  |
