# Movie Recommendation System with Spark ML
This project aims at generating movie recommendations with high accuracy and low costs. It is capable of handling millions of ratings.

# Requirement
This project requires PySpark package. As of 24 Feb 2022, it is only compatible with specific versions of Java and Python. Please check the official document of Apache Spark for details.

# Dataset
This project uses the MovieLens Dataset. It contains more than 25 millions ratings from 162541 users. Please prepare dataset with similar structure or obtain the dataset legally from MovieLens.

# Alogirthm
This model makes use of Alternating Least Square (ALS), which is a matrix factorization algorithm.

# How to Use

## Modify Configurations
Spark allows user to configure some of the properties. Please modify the configurations according to the hardware, especially `.config("spark.driver.memory", )` and `.config("spark.executor.cores", )`.

## Hyperparameters of ALS
Hyperparameters of the ALS such as `regParam` regularization parameter can be modified to optimize the model and prevent overfitting.

## Number of Recommendations Generated for each User
The following line of code decides the number of recommendations generated for each user:

`usersRecs = model.recommendForAllUsers(n)`

## Obtain the Recommendations for a Specific User
The recommendations for a specific user can be obtained by:

`userNRecs = usersRecs.filter(usersRecs['userId'] == n).collect()`

After all the processes, run the project with Python. It might take a couple of minutes.

`python SparkALS_25m.py`
