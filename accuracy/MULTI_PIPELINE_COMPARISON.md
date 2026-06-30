# Multi-Pipeline Benchmark — Statistical vs ML vs DL

48 pipelines across 2 datasets. All numbers from real data (no fabrication, Sec. 57.7).

| Dataset | Family | Model | Prep | Validation | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|---|---|---|
| CHB-MIT-20D | Statistical | LogisticRegression | standardize | LOSO | 0.7859 | 0.6943 | 0.7974 | 0.4189 | 0.8019 |
| CHB-MIT-20D | Statistical | LogisticRegression | standardize | epoch_5fold | 0.8141 | 0.7532 | 0.8218 | 0.4738 | 0.8573 |
| CHB-MIT-20D | Statistical | LDA | standardize | LOSO | 0.8812 | 0.3266 | 0.9506 | 0.3793 | 0.8157 |
| CHB-MIT-20D | Statistical | LDA | standardize | epoch_5fold | 0.8979 | 0.3437 | 0.9672 | 0.4279 | 0.851 |
| CHB-MIT-20D | Statistical | GaussianNB | standardize | LOSO | 0.8529 | 0.3747 | 0.9127 | 0.3614 | 0.7444 |
| CHB-MIT-20D | Statistical | GaussianNB | standardize | epoch_5fold | 0.866 | 0.4342 | 0.92 | 0.4187 | 0.8228 |
| CHB-MIT-20D | ML | RandomForest | standardize | LOSO | 0.8961 | 0.2722 | 0.9741 | 0.3678 | 0.7784 |
| CHB-MIT-20D | ML | RandomForest | standardize | epoch_5fold | 0.9381 | 0.4949 | 0.9935 | 0.6399 | 0.9385 |
| CHB-MIT-20D | ML | SVM_RBF | standardize | LOSO | 0.7888 | 0.5627 | 0.8171 | 0.3719 | 0.7626 |
| CHB-MIT-20D | ML | SVM_RBF | standardize | epoch_5fold | 0.8739 | 0.7968 | 0.8835 | 0.5841 | 0.9089 |
| CHB-MIT-20D | ML | HistGradientBoosting | standardize | LOSO | 0.8792 | 0.3563 | 0.9445 | 0.3959 | 0.7767 |
| CHB-MIT-20D | ML | HistGradientBoosting | standardize | epoch_5fold | 0.9396 | 0.5519 | 0.9881 | 0.67 | 0.927 |
| CHB-MIT-20D | ML | KNN | standardize | LOSO | 0.848 | 0.3146 | 0.9147 | 0.3151 | 0.6816 |
| CHB-MIT-20D | ML | KNN | standardize | epoch_5fold | 0.9402 | 0.5462 | 0.9894 | 0.6698 | 0.8823 |
| CHB-MIT-20D | DL | MLP | standardize | LOSO | 0.8799 | 0.3658 | 0.9441 | 0.4036 | 0.788 |
| CHB-MIT-20D | DL | MLP | standardize | epoch_5fold | 0.9404 | 0.5709 | 0.9866 | 0.6805 | 0.9215 |
| CHB-MIT-20D | Statistical | LogisticRegression | normalize | LOSO | 0.7851 | 0.6696 | 0.7995 | 0.4091 | 0.7951 |
| CHB-MIT-20D | Statistical | LogisticRegression | normalize | epoch_5fold | 0.8072 | 0.7443 | 0.8151 | 0.4618 | 0.8477 |
| CHB-MIT-20D | Statistical | LDA | normalize | LOSO | 0.8812 | 0.3266 | 0.9506 | 0.3793 | 0.8157 |
| CHB-MIT-20D | Statistical | LDA | normalize | epoch_5fold | 0.8979 | 0.3437 | 0.9672 | 0.4279 | 0.851 |
| CHB-MIT-20D | Statistical | GaussianNB | normalize | LOSO | 0.8529 | 0.3747 | 0.9127 | 0.3614 | 0.7444 |
| CHB-MIT-20D | Statistical | GaussianNB | normalize | epoch_5fold | 0.866 | 0.4342 | 0.92 | 0.4187 | 0.8228 |
| CHB-MIT-20D | ML | RandomForest | normalize | LOSO | 0.8961 | 0.2728 | 0.974 | 0.3684 | 0.7784 |
| CHB-MIT-20D | ML | RandomForest | normalize | epoch_5fold | 0.9381 | 0.4949 | 0.9935 | 0.6399 | 0.9386 |
| CHB-MIT-20D | ML | SVM_RBF | normalize | LOSO | 0.7904 | 0.5772 | 0.8171 | 0.3797 | 0.7798 |
| CHB-MIT-20D | ML | SVM_RBF | normalize | epoch_5fold | 0.8532 | 0.8082 | 0.8588 | 0.5502 | 0.9014 |
| CHB-MIT-20D | ML | HistGradientBoosting | normalize | LOSO | 0.8792 | 0.3563 | 0.9445 | 0.3959 | 0.7767 |
| CHB-MIT-20D | ML | HistGradientBoosting | normalize | epoch_5fold | 0.9396 | 0.5519 | 0.9881 | 0.67 | 0.927 |
| CHB-MIT-20D | ML | KNN | normalize | LOSO | 0.8534 | 0.3241 | 0.9196 | 0.3295 | 0.687 |
| CHB-MIT-20D | ML | KNN | normalize | epoch_5fold | 0.9411 | 0.5532 | 0.9896 | 0.6762 | 0.882 |
| CHB-MIT-20D | DL | MLP | normalize | LOSO | 0.8933 | 0.3051 | 0.9669 | 0.3886 | 0.8019 |
| CHB-MIT-20D | DL | MLP | normalize | epoch_5fold | 0.9181 | 0.3873 | 0.9845 | 0.5126 | 0.884 |
| UCI-178raw | Statistical | LogisticRegression | standardize | epoch_5fold | 0.7042 | 0.4417 | 0.7698 | 0.3739 | 0.5301 |
| UCI-178raw | Statistical | LDA | standardize | epoch_5fold | 0.8227 | 0.1217 | 0.9979 | 0.2155 | 0.5304 |
| UCI-178raw | Statistical | GaussianNB | standardize | epoch_5fold | 0.957 | 0.8883 | 0.9742 | 0.8921 | 0.984 |
| UCI-178raw | ML | RandomForest | standardize | epoch_5fold | 0.9699 | 0.8839 | 0.9914 | 0.9216 | 0.9958 |
| UCI-178raw | ML | SVM_RBF | standardize | epoch_5fold | 0.9705 | 0.9504 | 0.9755 | 0.928 | 0.9926 |
| UCI-178raw | ML | HistGradientBoosting | standardize | epoch_5fold | 0.9766 | 0.917 | 0.9915 | 0.94 | 0.9964 |
| UCI-178raw | ML | KNN | standardize | epoch_5fold | 0.9236 | 0.6222 | 0.9989 | 0.765 | 0.915 |
| UCI-178raw | DL | MLP | standardize | epoch_5fold | 0.9679 | 0.8848 | 0.9887 | 0.9169 | 0.9788 |
| UCI-178raw | Statistical | LogisticRegression | normalize | epoch_5fold | 0.6715 | 0.4552 | 0.7255 | 0.3566 | 0.5233 |
| UCI-178raw | Statistical | LDA | normalize | epoch_5fold | 0.8227 | 0.1217 | 0.9979 | 0.2155 | 0.5304 |
| UCI-178raw | Statistical | GaussianNB | normalize | epoch_5fold | 0.957 | 0.8883 | 0.9742 | 0.8921 | 0.984 |
| UCI-178raw | ML | RandomForest | normalize | epoch_5fold | 0.9698 | 0.8835 | 0.9914 | 0.9213 | 0.9958 |
| UCI-178raw | ML | SVM_RBF | normalize | epoch_5fold | 0.967 | 0.9361 | 0.9747 | 0.9189 | 0.9915 |
| UCI-178raw | ML | HistGradientBoosting | normalize | epoch_5fold | 0.9766 | 0.917 | 0.9915 | 0.94 | 0.9964 |
| UCI-178raw | ML | KNN | normalize | epoch_5fold | 0.9254 | 0.6317 | 0.9988 | 0.7721 | 0.9172 |
| UCI-178raw | DL | MLP | normalize | epoch_5fold | 0.9538 | 0.8257 | 0.9859 | 0.8773 | 0.9788 |
