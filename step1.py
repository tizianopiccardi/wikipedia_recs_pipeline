
from pyspark.sql import *
from pyspark.sql.functions import *
import re
from pyspark import SparkContext, SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)
sqlContext.setConf("spark.sql.parquet.compression.codec", "snappy")

wikipedia = sqlContext.read.parquet("hdfs:///user/piccardi/enwiki_1sept17.parquet")

"""
REMOVE FREQUENT SECTIONS AND FLAG THE STUBS
"""
sections_regex = re.compile(r'(^|[^=])==([^=\n\r]+)==([^=]|$)')
stub_regex = re.compile(r'\{\{[^}]+-stub\}\}')

blacklist = """References
External links
See also
Notes
Further reading
Bibliography
Sources
Footnotes
Notes and references
References and notes
External sources
Links
References and sources
External Links""".split("\n")


def extract_info(row):
    sections = sections_regex.findall(row.text)
    sections = set([s[1].strip() for s in sections])  # some sections are repeated (errors)
    is_stub = len(stub_regex.findall(row.text)) > 0
    if not is_stub:
        return [Row(aid=row.id, section=s.strip()) for s in sections if s.strip() not in blacklist]
    return []


id_sections = wikipedia \
    .filter("ns = '0'").filter("redirect is null") \
    .filter("text is not null") \
    .filter("length(text) > 0") \
    .flatMap(extract_info)

expanded_sections = sqlContext.createDataFrame(id_sections)
expanded_sections.registerTempTable("expanded_sections")

""" REMOVE UNIQUE SECTIONS - LONG TAIL OF 1.3M ITEMS """

non_unique_sections_rdd = expanded_sections.map(lambda r: (r.section, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .filter(lambda r: r[1] > 1 and r[0]) \
    .map(lambda r: Row(section=r[0]))

non_unique_sections = sqlContext.createDataFrame(non_unique_sections_rdd)
non_unique_sections.registerTempTable("non_unique_sections")

""" JOIN TO KEEP ONLY RELEVANT SECTION - NO UNIQUE, NO FREQUENT BLACKLISTED """

filtered_sections = sqlContext.sql("""
    SELECT es.aid, es.section
    FROM expanded_sections es
    JOIN non_unique_sections nus
    ON es.section = nus.section
""")

""" KEEP ONLY SURVIVED ARTICLES (NOT EMPTY AFTER PREVIOUS STEPS) """

article_sections_rdd = filtered_sections.map(lambda r: (r.aid, [r.section])) \
    .reduceByKey(lambda a, b: a + b).map(lambda t: Row(aid=t[0], sections=t[1])).filter(lambda r: len(r.sections) > 0)

article_sections = sqlContext.createDataFrame(article_sections_rdd)

article_sections.write.parquet("last/article_filtered_sections.parquet")