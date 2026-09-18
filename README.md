# Machine Learning Projects

## Project Overview
This repository brings together a set of applied machine learning studies spanning environmental analysis, food delivery optimization, and retail forecasting. Each subfolder is self-contained with its dataset, notebook workflow, and project-specific README.

The collection is organized around four themes:

**1. Deforestation analysis**
- `Deforestation_Issue_Analysis_SVM`: Predicts tree cover loss percentage from socioeconomic and environmental variables. The workflow covers one-hot encoding of country, train-test splitting, scaling, SVR training with linear, RBF, and polynomial kernels, GridSearchCV tuning, and permutation-based feature interpretation. A separate `result.ipynb` summarizes findings on agriculture pressure, governance, corruption, and illegal lumbering, with policy recommendations.

**2. Food delivery time modeling**
- `Food_deliverytime_time_regression`: Estimates delivery duration from order, distance, weather, traffic, and cost features. It applies deduplication, label encoding, haversine distance calculation from customer and restaurant coordinates, scaling, and linear and logistic modeling with confusion matrix and ROC evaluation.
- `Food_timing_classification`: Frames delivery performance as a binary fast versus delayed problem. It compares K-Nearest Neighbors with cross-validated selection of K, Naive Bayes, and Decision Tree classifiers, supported by classification reports, confusion matrices, and ROC curves.
- `Food_timing_clustring_ANN`: Combines unsupervised exploration with supervised prediction. It engineers distance and rush-hour indicators, applies KMeans, DBSCAN, and agglomerative clustering with PCA visualization, then trains a TensorFlow Keras multilayer perceptron to predict delivery status.
- `Food_timing_CNN`: Extends the same delivery dataset with a dense neural network built in Keras, tuned through Scikeras and GridSearchCV, and benchmarked against logistic regression. Evaluation includes accuracy, precision, recall, F1 score, confusion matrix, and ROC AUC.

**3. Global pollution and population studies**
- `Global_population_Apriori`: Uses Apriori association rule mining on global pollution records. After standardization and encoding, it extracts frequent itemsets and rules measured by support, confidence, and lift to surface relationships among pollution indicators. Interpreted output is retained in `Result.ipynb`.
- `Global_Population_Classification`: Assigns countries to low, medium, and high pollution severity categories. It evaluates Naive Bayes, K-Nearest Neighbors, and Decision Tree models with stratified splitting, scaling, cross-validation, and detailed confusion matrix comparison.
- `Global_Population_Clustering_ANN`: Examines pollution structure through clustering and predicts energy recovery potential with an artificial neural network. The Keras regressor is tuned with GridSearchCV and contrasted with a linear regression baseline using MAE, MSE, and R2. Supporting material includes `requirements.txt`, `task.pdf`, `report.pdf`, and `graphs.pdf`.
- `Global_Population_Regression`: Models the connection between air, water, and soil pollution measures and recoverable energy outcomes. The notebook addresses preprocessing, stratified splitting, scaling, model fitting, and classification-style evaluation to inform reduction and energy conversion strategies.

**4. Exploratory analysis and retail forecasting**
- `Exploratory_Data_Analysis`: Provides a complete exploratory workflow on the Titanic dataset. It reviews shape, types, descriptive statistics, missing values, duplicates, categorical distributions, outlier handling, label and one-hot encoding, Pearson correlation against the target, and a summary of the preprocessing pipeline.
- `Linear_Regression`: Forecasts weekly Walmart sales from store, temperature, fuel price, CPI, unemployment, and calendar attributes. It adds year, month, week, and quarter features, treats outliers, applies standardization, fits ordinary least squares regression, and reports MAE, MSE, RMSE, R2, and adjusted R2 with store-level visualizations.

## Technologies Used
Core data handling and visualization across all projects rely on Python with pandas, NumPy, Matplotlib, and Seaborn, executed through Jupyter notebooks.

Classical machine learning is implemented with scikit-learn, including StandardScaler, LabelEncoder, train-test splitting, GridSearchCV, cross-validation, linear and logistic regression, SVR, KNeighborsClassifier, Gaussian Naive Bayes, DecisionTreeClassifier, KMeans, DBSCAN, agglomerative clustering, PCA, and metrics such as accuracy, precision, recall, F1, ROC AUC, MAE, MSE, RMSE, R2, and adjusted R2.

Pattern mining uses mlxtend for Apriori frequent itemsets and association rules, with SciPy supporting statistical correlation in the exploratory study.

Neural network projects add TensorFlow and Keras with Scikeras wrappers, covering Sequential MLP and dense architectures, binary cross-entropy and regression losses, Adam optimization, hyperparameter search, and training history visualization.

## Execution / Usage
General requirements are Python 3 with Jupyter Notebook or JupyterLab. The clustering ANN project lists its dependencies explicitly and can be installed with:

```
pip install -r Global_Population_Clustering_ANN/requirements.txt
```

For other projects, the commonly needed packages are `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `mlxtend`, `tensorflow`, `keras`, and `scikeras`.

To run any study:

1. Open the corresponding project folder.
2. Confirm the expected CSV file is present in the same folder, for example `deforestation_dataset.csv`, `Food_Delivery_Time_Prediction.csv`, `Global_Pollution_Analysis.csv`, or `Walmart_Sales.csv`. The exploratory analysis project loads the Titanic dataset from Seaborn and requires internet access unless a local copy is supplied.
3. Launch the notebook, such as `main.ipynb`, `eda_and_preprocessing.ipynb`, or `Linear_Regression.ipynb`, and execute the cells sequentially from top to bottom.
4. Consult the additional notebooks where available: `result.ipynb` in the deforestation SVM project, `Result.ipynb` in the Apriori project, and the PDF reports in the clustering ANN project.

## Learning Outcomes
Working through this collection provided practical experience across the full machine learning workflow, from data inspection and cleaning to model selection and result interpretation. It strengthened understanding of preprocessing choices such as encoding, scaling, outlier treatment, and feature engineering, and clarified when to apply regression, classification, clustering, association mining, or neural networks.

The projects also reinforced model evaluation discipline, including train-test splitting, cross-validation, hyperparameter tuning, and the use of metrics such as RMSE, R2, precision, recall, F1, ROC AUC, and lift. Comparing linear baselines with tree-based models, SVM variants, and Keras networks helped develop judgment about complexity, overfitting, and interpretability.

Most of the work was carried out through self-directed learning, primarily using YouTube tutorials and AI assistance for concepts, code structure, and debugging. Whenever progress stalled, additional support was drawn from official documentation, reference articles, instructional videos, and AI explanations to resolve errors and refine the approach.
