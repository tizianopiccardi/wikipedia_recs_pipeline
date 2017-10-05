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

relevant_categories = sqlContext.read.parquet("relevant_categories.parquet")
relevant_categories.registerTempTable("relevant_categories")

def sampler(category_articles):
    category = category_articles[0]
    articles = category_articles[1]
    shuffle(articles)
    testing_article = articles.pop(0)
    return [Row(category=category, aid=a[0], sections=a[1], testing=False) for a in articles]\
            +[Row(category=category, aid=testing_article[0], sections=testing_article[1], testing=True)]


group_by_category = relevant_categories.map(lambda r: (r.category, [(r.aid, r.sections)])).reduceByKey(lambda a,b:a+b)
sampled_categories = sqlContext.createDataFrame(group_by_category.flatMap(sampler))

testing_set = sampled_categories.filter("testing=FALSE")

expanded_sections = sqlContext.createDataFrame(
    testing_set.flatMap(lambda r: [Row(category=r.category, aid=r.aid, section=s) for s in r.sections]))
expanded_sections.registerTempTable("expanded_sections")



category_indexes = sqlContext.createDataFrame(expanded_sections.select("category").distinct()
                                              .rdd.zipWithIndex().map(lambda r: Row(category=r[0].category, index=r[1])))

sections_indexes = sqlContext.createDataFrame(expanded_sections.select("section").distinct()
                                              .rdd.zipWithIndex().map(lambda r: Row(section=r[0].section, index=r[1])))


#
# category_indexes.write.parquet("sept_evaluation/category_indexes.parquet")
# sections_indexes.write.parquet("sept_evaluation/sections_indexes.parquet")
# sampled_categories.write.parquet("sept_evaluation/sampled_categories.parquet")
#
# category_indexes = sqlContext.read.parquet("sept_evaluation/category_indexes.parquet")
# sections_indexes = sqlContext.read.parquet("sept_evaluation/sections_indexes.parquet")
# sampled_categories = sqlContext.read.parquet("sept_evaluation/sampled_categories.parquet")


# format (category, section, count)
section_category_count = sqlContext.sql("""
    SELECT category, section, count(DISTINCT(aid)) occurrences
    FROM expanded_sections
    GROUP BY category, section
""")


section_category_count.registerTempTable("section_category_count")
category_indexes.registerTempTable("category_indexes")
sections_indexes.registerTempTable("sections_indexes")

# sampled_categories = sqlContext.createDataFrame(sc.parallelize(category_indexes.rdd.takeSample(False, 5000)))
# sampled_categories.registerTempTable("sampled_categories")

matrix_entries = sqlContext.sql("""
    SELECT scc.*, ci.index category_index, si.index section_index
    FROM section_category_count scc
    JOIN category_indexes ci
    JOIN sections_indexes si
    ON scc.section = si.section
    AND scc.category = ci.category
""").cache()
matrix_entries.registerTempTable("matrix_entries")

#
# categories = matrix_entries.map(lambda r: (r.category_index, [(r.section_index, r.occurrences)])).reduceByKey(lambda a,b:a+b)
# categories.map(lambda r: json.dumps(r[1])).saveAsTextFile("category_index.json")

training_ratings = matrix_entries.map(lambda l: Rating(l.category_index, l.section_index, l.occurrences))


model = ALS.trainImplicit(training_ratings, 75, 20, lambda_=0.001, alpha=120.0)
recommendations = model.recommendProductsForUsers(15)
top15 = sqlContext.createDataFrame(recommendations.flatMap(
                 lambda r: [Row(category_index=e.user, section_index=e.product, rating=e.rating) for e in r[1]]))
top15.registerTempTable("top15")

recs = sqlContext.sql("""
            SELECT ci.category, si.section, t.rating
            FROM top15 t
            JOIN category_indexes ci
            JOIN sections_indexes si
            ON t.category_index = ci.index
            AND t.section_index = si.index
            ORDER BY category
        """)

