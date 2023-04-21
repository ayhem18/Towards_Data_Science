# Import SparkSession
from pyspark.sql import SparkSession 

# Create SparkSession 
spark = SparkSession \
        .builder \
        .appName("my spark app") \
        .master("local[*]") \
        .getOrCreate()

print("\n\nQUESTION 1")
print("#" * 100)


from pyspark.sql.types import *
schema = StructType([
  StructField('video_id', StringType(), True),
  StructField('trending_date', DateType(), True),
  StructField('title', StringType(), True),
  StructField('channel_title', StringType(), True),
  StructField('category_id', IntegerType(), True),
  StructField('publish_time', TimestampType() , True),
  StructField('tags', StringType(), True),
  StructField('views', LongType(), True),
  StructField('likes', LongType(), True),
  StructField('dislikes', LongType(), True),
  StructField('comment_count', LongType(), True),
  StructField('thumbnail_link', StringType(), True),
  StructField('comments_disabled', BooleanType(), True),
  StructField('ratings_disabled', BooleanType(), True),
  StructField('video_error_or_removed', BooleanType(), True),
  StructField('description', StringType(), True),
])

df = spark.read.csv("GBvideos.csv", header=True, schema=schema)
from pyspark.sql.functions import split
from pyspark.sql.functions import col
df = df.withColumn('tags', split(col('tags'), r'\|'))
df.printSchema()


print("\n\nQUESTION 2")
print("#" * 100)

from pyspark.sql.functions import *

df.select('video_id', 'publish_time', (df.likes - df.dislikes).alias('delta')). \
                            where(df.publish_time < '2010-01-01').orderBy('delta', ascending=False). \
                            show(10)


print("\n\nQUESTION 3")
print("#" * 100)


df.groupby('channel_title').agg(avg('likes').alias('average_likes')).orderBy('average_likes', ascending=False).show(5)




print("\n\nQUESTION 4")
print("#" * 100)

df.select('channel_title', 'views').groupby('channel_title').sum().filter('sum(views) > 1000000').show(15)


print("\n\nQUESTION 5")
print("#" * 100)


df.agg(corr('likes', 'views').alias('correlation')).show()

print("\n\nQUESTION 6")
print("#" * 100)

df.select(explode('tags').alias('individual_tag')). \
groupBy('individual_tag').count().where("individual_tag != '[none]'"). \
orderBy('count', ascending=False).show(10)



print("\n\nQUESTION 7")
print("#" * 100)


df.groupBy("channel_title").max("comment_count").orderBy(desc("max(comment_count)")).show(10)

print("\n\nQUESTION 8")
print("#" * 100)

df.select(explode('tags').alias('individual_tag')). \
groupBy('individual_tag').count().where("individual_tag != '[none]'"). \
orderBy('count', ascending=False).withColumnRenamed('count', 'frequency').show(10)

print("\n\nQUESTION 9")
print("#" * 100)

import re
def match(description):
    if description is not None:
        return [list(re.findall(r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)", description)),
                list(re.findall("\B\@\w+", description))]
    else :
        return [[], []]

regex_udf = udf(match, ArrayType(ArrayType(StringType())))


print("\n\nQUESTION 10")
print("#" * 100)

df.select("title", "description", regex_udf(col("description"))).where('title is not null and description is not null'). \
withColumn("links", col("match(description)").getItem(0)). \
withColumn("mentions", col("match(description)").getItem(1)). \
orderBy([size('links') + size('mentions'), ('title')], ascending=[False, True]).select('title', 'mentions', 'links').show() # order was modified for output purposes


