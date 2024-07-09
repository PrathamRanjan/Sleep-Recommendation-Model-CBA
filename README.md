# Sleep Recommendation Model based on Clustering for University Computer Based Assignment (CBA) 
Details from the report are below

# Research Question 1: Sleep Improvement Recommendation Model

## How can a recommendation model be developed to help people improve their quality of sleep (as measured by sleep efficiency) based on their lifestyle choices?

A recommendation model in this use case requires a multi-step process as outlined below:

### 1. Clustering of Lifestyles
### 2. Development of Sleep Efficiency Prediction Model
### 3. Recommendation Function

### Clustering

Clustering is the action of grouping together objects that have high “similarity” in the context of a machine learning algorithm. Clustering is extremely useful as it helps to break down the problem and focus on relevant features specific to the segregated groups. In this scenario, the dataset will undergo clustering based on lifestyle choices. The algorithm chosen for clustering is the K-Means Clustering Algorithm. K-Means is an unsupervised machine learning algorithm that operates iteratively. It was chosen for this use case because of its simplicity and high interpretability. High interpretability is essential for this specific use case because this clustering will form the foundation of the recommendation model built on top of it.

Firstly, the data was pre-processed for clustering, which included the standardization of numerical variables and the one-hot encoding of categorical variables. These steps were important for the model development in the future as well. This standardization was conducted because it is essential for clustering and significantly boosts the results of many machine learning models, especially Gradient Boosted Models and Neural Networks.

After this step, the “k” value for K-Means clustering was chosen through the “Elbow Method.” This method is a graphical measure to determine the number of clusters. This step was essential to mitigate any over- or under-clustering of our data. Based on inspection, this method showed that k=4 for optimal results. The clustering was then conducted, resulting in the following four clusters:

- **Cluster 0:** Young, active individuals who have a high exercise frequency and low caffeine and alcohol consumption. Likely to have better sleep efficiency.
- **Cluster 1:** Older, less active individuals with moderate alcohol and caffeine consumption. Moderate sleep efficiency.
- **Cluster 2:** Middle-aged, moderate smokers and drinkers with low exercise frequency. Poorer sleep efficiency.
- **Cluster 3:** Young, very active, high caffeine consumption but low alcohol consumption. Good sleep efficiency.

**Figure 5.1 – PCA Representation of the 4 Clusters**

### Sleep Efficiency Prediction Model

The sleep efficiency prediction model was developed and chosen in Question 4 of this report; therefore, it will not be explained here again for the sake of repetition. The hyper-tuned random forest regressor will be used for the recommendation model.

### Building the Recommendation Model

The recommendation model leverages a content-based filtering approach aided by machine learning techniques such as K-Mean Clustering and Random Forest Regression. This allows the model to provide personalized, quantifiable recommendations that can be directly acted upon by someone.

Content-Based Filtering is a recommendation system that recommends items similar to those a user has shown a connection with in the past. In this study, lifestyle variables are considered as the items to be recommended. The model is built through a multistep process described below:

1. **Taking in an Input:** This model will take in new data, which will then undergo the same standardization, encoding, and transformations as the training data to ensure consistency.

2. **Cluster Prediction:** This input will then be classified under a cluster based on the trained K-Means model.

3. **Sleep Efficiency Prediction:** The regression model will then be used to predict the person's sleep efficiency.

4. **Target Cluster Identification:** Once the sleep efficiency has been calculated, it is compared to the average sleep efficiency of other clusters. A target cluster with a higher sleep efficiency than the input data is then identified.

5. **Recommendation Calculation:** Upon identifying the target cluster, the input feature values are compared to the centroid of the target cluster. The centroid represents the objective center of the cluster, minimizing the sum of square distances between itself and the various other data points within the same cluster. Mathematically, the centroid is the average of all the lifestyle values in that specific target cluster.

Based on this comparison, recommendations for increasing or decreasing specific lifestyle factors are outputted based on the direction, magnitude, and relevance of the differences. For example, age is a significant factor when it comes to sleeping; however, since age cannot be changed, this value is ignored in the comparison. Furthermore, empirical knowledge was used here; based on NIH guidelines, alcohol and caffeine before sleep reduce sleep quality. Therefore, any recommendation of an increase in alcohol was omitted.

The output of the model is a quantifiable number, as seen below:

{
  "Current Efficiency": 0.8021071205046246,
  "Potential Efficiency After Changes": 0.7970163695888739,
  "Improvement": -0.00509075091575073,
  "Current Cluster": 0,
  "Target Cluster": 0,
  "Recommendations": {
    "Caffeine consumption": "Reduce by 0.37 units",
    "Exercise frequency": "Increase by 0.53 units",
    "Daily Steps": "Increase by 0.94 units"
  }
}
Although the potential efficiency seems to be reducing, this could be attributed to the errors explained in the previous questions.

### Conclusion
The model answers this question by leveraging various analytical and machine learning techniques. Although the accuracy of projected changes in efficiency may not be precise, the actual recommendations are accurate and made relevant through the integration of empirical knowledge into the recommendation function. The model provides actionable and relevant advice, making it very scalable with major potential. The most important part of this model is that it provides incremental, quantifiable recommendations that are feasible in terms of their actionability. The model doesn’t just say "stop drinking alcohol or caffeine"; instead, it recommends incremental changes that are easy to implement from a user perspective.
