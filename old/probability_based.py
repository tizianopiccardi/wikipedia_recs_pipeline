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

sampled_categories = sqlContext.read.parquet("sept_evaluation/sampled_categories.parquet")

testing_set = sampled_categories.filter("testing=FALSE")

expanded_sections = sqlContext.createDataFrame(
    testing_set.flatMap(lambda r: [Row(category=r.category, aid=r.aid, section=s) for s in r.sections]))
expanded_sections.registerTempTable("expanded_sections")


# format (category, section, count)
section_category_count = sqlContext.sql("""
    SELECT category, section, count(DISTINCT(aid)) occurrences
    FROM expanded_sections
    GROUP BY category, section
""")
section_category_count.registerTempTable("section_category_count")

training_set_count = sqlContext.sql("""
    SELECT category, COUNT (DISTINCT(aid)) articles_count
    FROM expanded_sections
    GROUP BY category
    having articles_count > 1
""")
training_set_count.registerTempTable("training_set_count")

section_probability = sqlContext.sql("""
    SELECT ts.category, sbc.section, sbc.occurrences/ts.articles_count as probability, sbc.occurrences occurrences
    FROM section_category_count sbc
    JOIN training_set_count ts
    ON sbc.category = ts.category
""")

section_probability.write.mode('overwrite').parquet("sept_evaluation/section_probability.parquet")


############################################

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

sampled_categories = sqlContext.read.parquet("sept_evaluation/sampled_categories.parquet")
section_probability = sqlContext.read.parquet("sept_evaluation/section_probability.parquet")




testing_set = sampled_categories.filter("testing=TRUE").cache()
testing_set.registerTempTable("testing_set")



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

dataset.write.parquet("sept_evaluation/dataset.parquet")



