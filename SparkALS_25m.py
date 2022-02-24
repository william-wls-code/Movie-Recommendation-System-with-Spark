from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

import csv


def getMovieNames():
    ''' Extract movie names from csv file. '''
    movieID_to_Name = {}

    with open("./ml-25m/movies.csv", newline='', encoding='ISO-8859-1') as csvfile:
        movies = csv.reader(csvfile)
        # Skip header line
        next(movies)
        for row in movies:
            movieID = int(row[0])
            movieName = row[1]
            movieID_to_Name[movieID] = movieName
    return movieID_to_Name


# Modify the following configurations based on hardware:
# .config("spark.driver.memory", )
# .config("spark.executor.cores", )
if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALS Movie Recommendation System")\
        .config("spark.driver.memory", "8g")\
        .config("spark.executor.cores", '8')\
        .getOrCreate()

    # Load the data into Spark and store in a RDD
    # Raw data
    lines = spark.read.option("header", "true").csv("./ml-25m/ratings.csv").rdd

    # The lambda function transforms each line into a Row object
    ratingsRDD = lines.map(lambda r: Row(
        userId=int(r[0]),
        movieId=int(r[1]),
        rating=float(r[2]),
        timestamp=int(r[3])))

    # Convert into DataFrame, a dataset that Spark machine learning library identifies
    ratings = spark.createDataFrame(ratingsRDD)

    # Divide the data set into training set and test set
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Train the ALS (Alternating Least Squares) model
    alsModel = ALS(maxIter=10, regParam=0.02,
                   userCol="userId", itemCol="movieId",
                   ratingCol="rating", coldStartStrategy="drop")
    model = alsModel.fit(training)

    # Generate predictions
    # Compute the rmse (Root-mean-square Error) by comparing the predicted ratings and actual ratings
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    print()
    print("Root-mean-square Error = " + str(rmse))

    # Generate top-N recommendations
    usersRecs = model.recommendForAllUsers(10)

    # Pick a random userID and print the recommendations
    user83Recs = usersRecs.filter(usersRecs['userId'] == 83).collect()
    spark.stop()
    movieID_to_name = getMovieNames()

    for row in user83Recs:
        for rec in row.recommendations:
            if rec.movieId in movieID_to_name:
                print(movieID_to_name[rec.movieId])
