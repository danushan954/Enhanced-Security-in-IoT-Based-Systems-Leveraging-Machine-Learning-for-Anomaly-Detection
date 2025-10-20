# Enhanced-Security-in-IoT-Based-Systems-Leveraging-Machine-Learning-for-Anomaly-Detection
Developed a machine learning pipeline to detect and mitigate cyber-attacks in healthcare IoT systems. Implemented supervised models (Random Forest, AdaBoost, Logistic Regression, Perceptron, Deep Neural Networks on CICIOT2023 dataset.
1. Project Overview

This repository documents a robust, end-to-end machine learning pipeline for the classification of IoT network traffic, utilizing the comprehensive CICIoT2023 dataset. The project's central objective is to demonstrate a scalable and effective methodology for processing, balancing, and modeling large-scale security data to accurately distinguish between benign and malicious network activities.
The primary challenge addressed is the sheer scale and complexity of the source data: an initial dataset of 42 million rows, occupying 12 GB in its raw format, and characterized by a severe class imbalance common in real-world network logs This project presents a pragmatic approach to overcoming these hurdles within a resource-constrained environment like Google Colab. The technical solution emphasizes memory-efficient data engineering, employing Dask for out-of-memory parallel processing and the Parquet file format for optimized storage. The pipeline proceeds through systematic class balancing, using techniques like downsampling and the Synthetic Minority Over-sampling Technique (SMOTE), to generate three distinct datasets for binary, multi-class (8-class), and fine-grained (34-class) classification tasks. Finally, it conducts a comparative analysis of five machine learning and deep learning models to evaluate their performance across these varying levels of complexity.
This work serves not only as an exercise in building high-performance classification models but also as a practical case study in data engineering under significant hardware limitations. The consistent focus on memory management—from the choice of Dask over Pandas to the strategic flushing of RAM during data generation—reveals that the core engineering challenge was resource management rather than pure algorithmic complexity. The resulting pipeline provides a blueprint for tackling large-scale data science problems on accessible, prosumer-level cloud platforms.

2. Key Features

Scalable Data Processing: Efficiently handles a 42-million-row dataset within the memory constraints of Google Colab. This is achieved by leveraging Dask for parallelized, out-of-memory computation and the Parquet file format, which provides highly compressed, columnar storage for rapid data access.
Multi-Granularity Class Balancing: Addresses severe class imbalance by creating three distinct, class-balanced datasets. The pipeline employs a hybrid strategy, combining strategic downsampling for majority classes with the Synthetic Minority Over-sampling Technique (SMOTE) for minority classes, resulting in binary, 8-class, and 34-class versions of the data.
Effective Dimensionality Reduction: Reduces model complexity, mitigates overfitting, and accelerates training times by systematically removing redundant features. The methodology identifies and eliminates features with a Pearson Correlation Coefficient (PCC) greater than 0.9, with the effectiveness of the reduction visually validated using correlation heatmaps.
Comprehensive Model Evaluation: Conducts a rigorous, comparative analysis of five distinct machine learning and deep learning models: Random Forest, Logistic Regression, AdaBoost, Perceptron, and a Deep Neural Network (DNN). This evaluation identifies the optimal classifiers for tasks of varying complexity, from simple binary classification to fine-grained, 34-class identification.

3. Technical Stack

Data Processing & Manipulation: Dask, pandas, NumPy 
Machine Learning & Modeling: Scikit-learn, Imbalanced-learn (for SMOTE) 
Deep Learning: TensorFlow, Keras 
Environment & Storage: Google Colab Pro, Google Drive 
Utilities: tqdm (for progress bars), glob (for file path matching) 

4. Repository Structure
.
├── data/
│   ├── raw/               # Placeholder for the original CICIoT2023 dataset
│   ├── processed/         # Output of the preprocessing pipeline
│   └── balanced/          # Location for the 2, 8, and 34-class balanced datasets
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_Dataset_Balancing.ipynb
│   ├── 3_Feature_Reduction.ipynb
│   └── 4_Model_Training_and_Evaluation.ipynb
│   └── 5_Visualization.ipynb
├── results/
│   └── Model performance reports
└── README.md             



5. The Machine Learning Pipeline: A Step-by-Step Guide

This section provides a detailed walkthrough of the entire project workflow, from initial environment setup to final model evaluation.

5.2. Stage 1: Data Preprocessing

Objective: To clean and prepare the raw 12 GB CICIoT2023 dataset for machine learning, with a primary focus on efficiency and scalability.
The initial challenge stems from the dataset's size, which makes conventional tools like Pandas loading a CSV file directly into memory infeasible. Such an approach would lead to memory overload and session crashes within a typical Google Colab environment.1 The architectural choices made at this stage are not minor optimizations but foundational enablers for the entire project. The selection of Dask and Parquet directly solves the primary memory constraint, and this solution propagates through the pipeline, allowing all subsequent stages to operate on the full, uncompromised dataset.
The solution involves a strategic shift to a more scalable toolchain. By converting the original 12 GB CSV to the Parquet format, the on-disk size was reduced to 3.8 GB due to Parquet's efficient columnar storage and compression. This format, combined with Dask's lazy evaluation and parallel processing capabilities, allows for out-of-memory operations on the complete dataset without resorting to downsampling or data loss.1

Workflow Steps:
Load Data: The dataset is loaded efficiently using dask.dataframe.read_parquet(), which reads the data in chunks rather than all at once.1
Deduplication: The pipeline identifies and removes 34 duplicate records to prevent data redundancy from biasing the models.1
Handle Missing Values: A check is performed, confirming that the dataset contains no missing or null values that would require imputation or removal.1
Variance Thresholding: Six features exhibiting near-zero variance are removed. These features are essentially constant across all samples and thus provide no discriminatory information for machine learning models.1
Feature Scaling: dask_ml.preprocessing.StandardScaler is applied to standardize all numerical features. This process transforms the data to have a mean of zero and a unit variance, which is a critical prerequisite for the optimal performance of scale-sensitive models like Logistic Regression, Perceptron, and Deep Neural Networks.1

5.3. Stage 2: Dataset Balancing

Objective: To mitigate the extreme class imbalance inherent in the dataset and create three distinct, balanced datasets for evaluating model performance on binary, multi-class, and fine-grained classification tasks.
The creation of three separate datasets is a deliberate experimental design choice. It facilitates a nuanced analysis of how each model's performance and generalization capabilities are affected as the complexity of the classification problem—and its decision boundaries—increases.
Methodology:
2-Class Version (Binary): All 33 malicious labels are consolidated into a single "Attack" class, which is then balanced against the "Benign" class. A pure downsampling technique is used to create a perfectly balanced dataset containing 8,450 samples per class, for a total of 16,900 rows.1
8-Class Version (Multi-Class): The original 34 labels are mapped into 8 broader attack categories (e.g., DDoS, DoS, Reconnaissance). A hybrid balancing approach is employed: majority classes are downsampled, while minority classes are oversampled using SMOTE to reach a target of 33,800 samples per class. To manage memory, each category is processed in batches, with intermediate SMOTE data temporarily saved to Google Drive to flush RAM between operations.1
34-Class Version (Fine-Grained): Each of the 34 unique labels is balanced to a target of 84,500 samples using a combination of undersampling and SMOTE. During this process, a practical limitation of SMOTE was observed: 13 of the most under-represented classes could not reach the full target due to an insufficient number of seed samples. These classes were capped at approximately 82,280 samples each. This outcome is a significant finding, as it highlights the real-world constraints of synthetic data generation; SMOTE can only generate new samples within the convex hull of existing ones, limiting its effectiveness when the initial minority class is extremely sparse. The final dataset contains approximately 2.84 million rows.1

5.4. Stage 3: Feature Reduction

Objective: To simplify the datasets by removing redundant features, which helps to reduce overfitting, improve model training speed, and enhance interpretability.1
Methodology:
Calculate Correlation: For each of the three balanced datasets, the Pearson Correlation Coefficient (PCC) is computed between every pair of numerical features. The PCC measures the linear relationship between two variables, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).1
Identify Redundancy: Any pair of features with a PCC value greater than 0.9 is flagged as highly correlated and therefore redundant. This indicates that the two features carry very similar information.1
Drop Features: One feature from each highly correlated pair is systematically removed. The process is designed to preserve the most critical features and ensures that the target 'label' column is never dropped.1
Validation: The success of this process is visually validated using heatmaps. A "before" heatmap of the original feature set shows prominent red blocks, indicating high multicollinearity that can degrade model performance. An "after" heatmap generated from the reduced dataset shows the elimination of these blocks, confirming that the feature redundancy has been successfully addressed.1
Results:
2-Class Dataset: 10 features dropped. Final shape: 16,900 rows × 31 columns.1
8-Class Dataset: 7 features dropped. Final shape: 246,022 rows × 34 columns.1
34-Class Dataset: 8 features dropped. Final shape: 2,844,140 rows × 33 columns.1

5.5. Stage 4: Model Training and Evaluation

Objective: To train and evaluate five different machine learning models on the three feature-reduced, balanced datasets to conduct a comprehensive comparative performance analysis.
Preprocessing for Modeling:
Data Loading: The final, cleaned Parquet files are loaded into memory for model training using pandas.read_parquet().1
Label Encoding: Scikit-learn's LabelEncoder is used to convert the string-based class labels into integer representations, which is a requirement for most machine learning libraries.1
Train-Test Split: The data is split into an 80% training set and a 20% testing set using train_test_split. Two key parameters are used: stratify=y_enc ensures that the balanced class distribution is preserved in both the training and testing sets, preventing evaluation bias, and random_state=42 ensures the split is reproducible.1
Model Configurations:
Random Forest: An ensemble model comprising 100 decision trees (n_estimators=100). It is well-suited for high-dimensional, non-linear data and is configured with n_jobs=-1 to utilize all available CPU cores for parallel processing.1
Logistic Regression: A linear classifier configured with multi_class='multinomial' to enable proper softmax-based classification for more than two classes. The 'saga' solver is chosen for its efficiency on large datasets.1
AdaBoost: An ensemble model that builds 100 sequential weak learners (decision stumps). It is included to evaluate a boosting-based approach.1
Perceptron: A simple, single-layer neural network that serves as a baseline linear classifier.1
Deep Neural Network (DNN): A 3-layer sequential model built with Keras. It uses ReLU activation functions for hidden layers and Dropout(0.3) to prevent overfitting. The output layer uses a softmax activation for multi-class probability distribution. The model is compiled with the Adam optimizer and categorical_crossentropy loss function.1
Evaluation Metrics:
All models are evaluated using a classification_report and a confusion_matrix. The primary metric for comparison is the Macro Average F1-Score. This metric calculates the F1-score for each class independently and then takes the unweighted average. It is particularly important for this project because it treats all classes as equally important, providing a robust measure of a model's ability to generalize across every category, regardless of any residual differences in class representation.1

6. Performance and Results

The final results provide a clear picture of how each model's architecture interacts with the complexity of the classification task.

6.1. Consolidated Performance Metrics

The following table synthesizes the performance of all five models across the three datasets, using both overall accuracy and the more telling Macro F1-Score to assess performance.1
Table: Consolidated Model Performance (Accuracy & Macro F1-Score)
Model
2-Class Accuracy
2-Class Macro F1
8-Class Accuracy
8-Class Macro F1
34-Class Accuracy
34-Class Macro F1
Random Forest
0.99
0.99
0.91
0.90
0.97
0.97
Deep Neural Network
0.99
0.99
0.73
0.72
0.73
0.73
Logistic Regression
0.98
0.98
0.66
0.62
0.51
0.47
AdaBoost
0.99
0.99
0.72
0.66
0.40
0.33
Perceptron
0.97
0.97
0.58
0.54
0.46
0.41


6.2. Analysis and Key Takeaways

The consolidated results reveal a compelling narrative about the suitability of different model architectures for problems of varying complexity. The progression from a simple binary task to a highly granular 34-class problem effectively stress-tests each model's capabilities.
Overall Best Performer: Random Forest is the unequivocal top performer across the board. It achieves near-perfect scores on both the 2-class and 34-class tasks and maintains a very strong 0.90 Macro F1-Score on the 8-class problem. Its inherent ability to capture complex, non-linear relationships and interactions between features makes it exceptionally well-suited for this type of high-dimensional classification task.1
Most Robust Model: The Deep Neural Network (DNN) demonstrates the most robust and consistent performance as complexity increases. While it does not reach the peak scores of Random Forest, it maintains a respectable Macro F1-Score of approximately 0.73 on both the 8-class and 34-class problems. This consistency highlights its superior ability to generalize across many classes compared to the linear and simpler ensemble models.1
The Linear Model Breakdown: The performance of Logistic Regression and Perceptron degrades sharply as the number of classes increases. Their Macro F1-scores plummet to below 0.50 on the 34-class problem. This provides clear empirical evidence that the decision boundaries separating the fine-grained attack types are highly non-linear. These linear models, by their nature, are incapable of learning such complex boundaries and consequently fail to generalize.1
AdaBoost's Limitations: While strong on the binary task, AdaBoost also struggles significantly with the 34-class problem, with its Macro F1-score dropping to 0.33. This suggests that its approach of sequentially boosting simple decision stumps is insufficient to capture the intricate patterns required for fine-grained classification in this domain.1
Ultimately, the results powerfully illustrate a fundamental principle in machine learning: for the simple, likely linearly separable binary problem, nearly any model performs well. However, as the intrinsic complexity of the problem grows, the architectural superiority of models capable of learning non-linear and hierarchical representations—namely Random Forest and the DNN—becomes overwhelmingly apparent. The experimental design of this project successfully demonstrates this principle in a practical, real-world context.
