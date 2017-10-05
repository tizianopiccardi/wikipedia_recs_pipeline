"""
CREATE THE TUPLES IN THE FORMAT:
    <CATEGORY, ARTICLE_ID, [SECTIONS]>
"""

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

""" KEEP ONLY SURVIVED ARTICLES (NOT EMPTY AFTER PREVIOUS STEPS) """
article_filtered_sections = sqlContext.read.parquet("last/article_filtered_sections.parquet")

""" SELECT HERE THE MIN SIZE OF THE CATEGORY """
category_articles = sqlContext.read.json("gini_filtered_articles.json").filter("size(articles)>1")
""" ^ LIST OF CATEGORIES AND IDs OF THE ARTICLES - INDIVIDUAL ROW (ARTICLE IS EXPLICT IN EVERY PARENT CATEGORY)
+--------------------+--------------------+
|            articles|            category|
+--------------------+--------------------+
|          [36301114]|Olympic_gymnasts_...|
|[6423889, 1319745...|Witchcraft_(band)...|
|[471396, 196662, ...|Australian_aerosp...|
...
+--------------------+--------------------+
"""

""" EXPANDED VERSION """
expanded_category_articles = sqlContext.createDataFrame(
    category_articles.flatMap(lambda r: [Row(category=r.category, aid=i) for i in r.articles]))
""" ^ 
+--------+--------------------+
|     aid|            category|
+--------+--------------------+
|36301114|Olympic_gymnasts_...|
| 6423889|Witchcraft_(band)...|
|13197456|Witchcraft_(band)...|
...
+--------+--------------------+
"""

article_filtered_sections.registerTempTable("article_filtered_sections")
expanded_category_articles.registerTempTable("expanded_category_articles")


""" JOIN WITH SECTIONS """
category_articles_sections = sqlContext.sql("""
    SELECT eca.category, afs.aid, afs.sections
    FROM article_filtered_sections afs
    JOIN expanded_category_articles eca
    ON eca.aid = afs.aid
""")
category_articles_sections.registerTempTable("category_articles_sections")


category_articles_sections.write.parquet("last/category_articles_sections.parquet")
""" ^^^
+--------------------+----+--------------------+
|            category| aid|            sections|
+--------------------+----+--------------------+
|      Linear_algebra|2431|[Common examples,...|
| Functional_analysis|2431|[Common examples,...|
|             Systems|2431|[Common examples,...|
...
+--------------------+----+--------------------+
"""