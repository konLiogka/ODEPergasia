
# ODEPergasia
Assignment for the course "Extraction and Information Mining"

Written in python, using the libraries scikit learn for clustering and machine learning algorithms, matplotlib for showing graphs and visualising data and apyori for apriori algorithm.

# Task A - Imputation

The task involves addressing missing values in the provided dataset, "marketing.csv," marked as '?' due to incomplete responses from respondents. To handle this issue, the KNNImputer function from the sklearn library is utilized. This method effectively fills missing values by searching for neighboring samples with similar characteristics, so there aren't outliers. It performs well for the current dataset. It would need more computational power for large datasets as it computes every distance between every sample.

Experimentation with different imputation values reveals that a value of 35 yields optimal results. Before proceeding with clustering, data analysis is conducted using seaborn library to visualize relationships between some features I've chosen.

# Task B - Clustering:

In order to obtain clear results in clustering algorithms, it's crucial to choose an appropriate number of clusters (K) to avoid overfitting and misinterpretation. For the KMEANS algorithm, the elbow method is employed to determine the optimal K value. This method calculates the sum of squared errors for different K values and visualizes them to identify the point where the line starts to resemble a linear trend or an elbow shape. For the dataset in question, K is determined to be between 13-15.

After determining the appropriate K value, parallel coordinates plots are generated for 7 features to observe how many clusters should be formed based on where distortion starts to diminish.

Applying the KMEANS algorithm, parallel coordinates plots for features are created, using the same number of plots as clusters for better visualization.

Hierarchical clustering divides samples into clusters based on similarity. Visualization using dendrograms suggests 12 clusters if a horizontal line is drawn at the 100 level.

For DBSCAN, the challenge lies in determining suitable values for min points and epsilon. A distance-based method like k-distances helps identify these values. In the provided example, min points is found to be 35 and epsilon is 3.

# Task C - Classification:

For proper classification of classes 8 and 9, minority oversampling using SMOTE is applied to increase the number of samples in these classes for the training sets. Training and testing sets are split 80%-20%.

Naive Bayes algorithm, despite its simplicity in applying the Bayes rule assuming independence among features, doesn't yield satisfactory results, especially in terms of precision and recall.

![NaiveByes](https://github.com/konLiogka/ODEPergasia/assets/78957746/89dc127e-93f8-463c-9830-1b06147db85a)

KNN, a widely used machine learning algorithm, can perform well depending on the value of K. Here, I set K to 35. The accuracy reaches 39%, with improved recall and precision for classes 8 and 9. However, for large datasets, training might be slow due to the algorithm needing to compute distances between samples.

![KNN](https://github.com/konLiogka/ODEPergasia/assets/78957746/2aebe2a4-a14c-40d9-a01f-4b3cccd289f2)

Decision tree classifiers categorize data based on various decisions made at each node, offering straightforward interpretation. While the accuracy remains unchanged, there is an improvement in recall and precision, indicating fewer false negatives and false positives.

![dt](https://github.com/konLiogka/ODEPergasia/assets/78957746/349388bb-2c86-4399-9061-d03a2e2ba209)


Random forest performed the best (depth 15), which is essentially multiple decision trees. The output is decided by the output most trees agree with. SLightly better accuracy but overall better recall and precision.
![rf](https://github.com/konLiogka/ODEPergasia/assets/78957746/558d75f0-8738-4243-9edf-eb502cef2df1)

# Task D - Apriori:

The Apriori algorithm is utilized to discover correlations among the dataset by extracting rules. Lift, measuring the frequency of occurrence of associations relative to the random occurrence frequency, is employed here.

All features are used, with manual mapping of numeric values to the responses. Setting the lift to 2.7 reveals various rules.

Initially, individuals with low income are observed to live with their families. Conversely, married individuals tend to reside in the area for over 10 years and own their homes, while those not in a relationship tend to rent apartments. Additionally, married couples have both single and dual income. Individuals with low income are either unemployed or students. English is the dominant language. Two-member families have their own homes. Individuals who own homes tend to be of Caucasian ethnicity. Those studying or in school live with their families.

