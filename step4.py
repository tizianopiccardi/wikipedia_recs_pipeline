from random import shuffle
import math
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import re
from pyspark import SparkContext, SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)
sqlContext.setConf("spark.sql.parquet.compression.codec", "snappy")

sampled_dataset = sqlContext.read.parquet("last/sampled_dataset.parquet")

""" GET ARTICLES FLAGGED AS TRAINING SET """
training_set = sampled_dataset.filter("training=TRUE").select(["category", "article"])

expanded_sections = sqlContext.createDataFrame(
    training_set.flatMap(
        lambda r: [Row(category=r.category, aid=r.article.aid, section=s) for s in r.article.sections]))
expanded_sections.registerTempTable("expanded_sections")

"""
COUNT THE NUMBER OF PAIRS <CATEGORY, SECTION>
HOW MANY TIMES SECTION S APPEARS IN CATEGORY C
"""
section_category_count = sqlContext.sql("""
    SELECT category, section, count(DISTINCT(aid)) occurrences
    FROM expanded_sections
    GROUP BY category, section
""")
section_category_count.registerTempTable("section_category_count")

"""
COUNT THE NUMBER OF ARTICLES PER CATEGORY
"""
training_set_count = sqlContext.sql("""
    SELECT category, COUNT (DISTINCT(aid)) articles_count
    FROM expanded_sections
    GROUP BY category
""")
training_set_count.registerTempTable("training_set_count")

""" RETURN CATEGORY, SECTION, PROBABILITY, OCCURRENCES """
section_probability = sqlContext.sql("""
    SELECT ts.category, sbc.section, sbc.occurrences/ts.articles_count as probability, sbc.occurrences occurrences
    FROM section_category_count sbc
    JOIN training_set_count ts
    ON sbc.category = ts.category
""")

section_probability.write.parquet("last/section_probability.parquet")

""" GENERATE INDEX FOR ALS MATRIX """
category_indexes = sqlContext.createDataFrame(section_probability.select("category").distinct()
                                              .rdd.zipWithIndex().map(
                                                lambda r: Row(category=r[0].category, index=r[1])))

sections_indexes = sqlContext.createDataFrame(section_probability.select("section").distinct()
                                              .rdd.zipWithIndex().map(lambda r: Row(section=r[0].section, index=r[1])))

category_indexes.write.parquet("last/category_indexes.parquet")
sections_indexes.write.parquet("last/sections_indexes.parquet")
