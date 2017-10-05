
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


dataset = sqlContext.read.parquet("sept_evaluation/dataset.parquet")

sampled_categories = sqlContext.read.parquet("sept_evaluation/sampled_categories.parquet")

testing_set = sampled_categories.filter("testing=FALSE")
testing_set.registerTempTable("testing_set")


category_count = sqlContext.sql("""
    select category, count(DISTINCT (aid)) cnt
    from testing_set
    GROUP by category
""").cache()
category_count.registerTempTable("category_count")





def get_precision_recall(row):
    result = []
    for k in range(1, 15):
        recommendations = set([r[0] for r in row.recommendations[:k]])
        if len(recommendations) > 0:
            precision = 0.0
            recall = 0.0
            for t in row.testing_set:
                #article = t.id
                sections = t.sections
                positive = 0
                for s in sections:
                    positive += 1 if s in recommendations else 0
                precision += positive / float(len(recommendations))
                recall += positive / float(len(sections))
            precision = precision / len(row.testing_set)
            recall = recall / len(row.testing_set)
            result.append(Row(category=row.category, k=k, precision=precision, recall=recall))
    return result



precision_recall = sqlContext.createDataFrame(dataset.flatMap(get_precision_recall))
precision_recall.registerTempTable("precision_recall")



avg_precision_recall = sqlContext.sql("""
    SELECT 50*CAST(c.cnt/50 AS INTEGER) lower, 50*(1+CAST(c.cnt/50 AS INTEGER)) upper, k, AVG (precision) precision, AVG (recall) recall, COUNT(*) total
    FROM precision_recall p
    JOIN category_count c
    ON p.category = c.category
    and k <= 10
    GROUP BY CAST(c.cnt/50 AS INTEGER) , k
    ORDER BY CAST(c.cnt/50 AS INTEGER), k
""")


avg_precision_recall.write.json("step50.json")


