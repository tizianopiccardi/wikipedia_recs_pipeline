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
    SELECT k, AVG (precision) precision, AVG (recall) recall
    FROM precision_recall
    GROUP BY k
""")

values = avg_precision_recall.map(lambda r: json.dumps({"k":r.k, "precision": r.precision, "recall":r.recall})).collect()


for v in values:
    print(v)

