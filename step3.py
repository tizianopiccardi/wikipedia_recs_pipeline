"""
SPLIT THE DATASET IN TRAINING/TESTING
"""
from random import shuffle
from pyspark.sql import *
from pyspark import SparkContext, SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)
sqlContext.setConf("spark.sql.parquet.compression.codec", "snappy")

relevant_categories = sqlContext.read.parquet("relevant_categories.parquet")

bycategory_rdd = relevant_categories.map(lambda r: (r.category, [(r.aid, r.sections)])).reduceByKey(
    lambda a, b: a + b).cache()


def sampler(row):
    articles_list = list(row[1])
    shuffle(articles_list)
    testing_test = articles_list.pop(0)
    result = [Row(category=row[0], article=Row(aid=a[0], sections=a[1]), training=True) for a in articles_list]
    result.append(
        Row(category=row[0], article=Row(aid=testing_test[0], sections=testing_test[1]), training=False))
    return result


sampled_dataset = sqlContext.createDataFrame(bycategory_rdd.flatMap(sampler))

sampled_dataset.write.parquet("last/sampled_dataset.parquet")

################
""" SAVE LOG - UNIQUE """
testing_articles = sampled_dataset.filter("training=FALSE").map(lambda r: r.article.aid).distinct()
testing_articles.map(lambda r: str(r)).saveAsTextFile("testing_set.txt")

""" SAVE LOG - BY CATEGORY """
testing_category_article = sampled_dataset.filter("training=FALSE").map(lambda r: (r.category, r.article.aid))
testing_category_article.map(lambda r: r[0]+"\t"+str(r[1])).saveAsTextFile("testing_set_by_category.txt")
