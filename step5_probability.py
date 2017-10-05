
from random import shuffle
import math
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import re
from pyspark import SparkContext, SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext()
sqlContext = SQLContext(sc)

sampled_dataset = sqlContext.read.parquet("last/sampled_dataset.parquet")
section_probability = sqlContext.read.parquet("last/section_probability.parquet")


""" SELECT THE TESTING SET """
testing_set = sampled_dataset.filter("testing=TRUE").cache()
testing_set.registerTempTable("testing_set")

""" AGGREGATE THE TESTING SET """
testing_rdd = testing_set.map(lambda r: (r.category, [Row(id=r.aid, sections=r.sections)])).reduceByKey(
    lambda a, b: a + b)
testing = sqlContext.createDataFrame(testing_rdd.map(lambda r: Row(category=r[0], testing_set=r[1])))

testing.registerTempTable("testing")




############################
# Merged
recommendation_rdd = section_probability.map(
    lambda r: (r.category, [(r.section, r.probability)])) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda c: Row(category=c[0], sections=sorted(c[1], key=lambda s: -s[1])[0:100]))
recommendation = sqlContext.createDataFrame(recommendation_rdd).persist()
recommendation.registerTempTable("recommendation")

dataset = sqlContext.sql("""
    SELECT r.category, r.sections recommendations, t.testing_set testing_set
    FROM testing t
    JOIN recommendation r
    ON r.category = t.category
""")
