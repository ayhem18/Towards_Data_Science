import pandas as pd
import pyspark
import numpy as np
from pyspark.sql import SparkSession
from os.path import join


# Import SparkSession
from pyspark.sql import SparkSession 

# TODO: change the configuration parameters to run on HDFS

# Create SparkSession 
spark = SparkSession \
        .builder \
        .appName("my spark app") \
        .master("local[*]") \
        .getOrCreate()


from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, DateType, TimestampType, LongType


# construct a schema for the table like PySpark object 
schema = StructType([
    StructField("video_id", StringType(), ),
    StructField("trending_date", StringType(), False),
    StructField("title", StringType(), False),
    StructField("channel_title", StringType(), False),
    StructField("category_id", IntegerType(), False),
    StructField("publish_time", TimestampType(), False),
    StructField("tags", StringType(), False),
    StructField("views", LongType(), False),
    StructField("likes", IntegerType(), False),
    StructField("dislikes", IntegerType(), False),
    StructField("comment_count", IntegerType(), False),
    StructField("thumbnail_link", StringType(), False),
    StructField("comments_disabled", BooleanType(), False),
    StructField("ratings_disabled", BooleanType(), False),
    StructField("video_error_or_removed", BooleanType(), False),
    StructField("description", StringType(), False),    
])

# read the data from the csv file
df = spark.read.format("csv").load("GBvideos.csv", header = True, schema = schema)
df.createOrReplaceTempView("videos")



print("\nQUESTION 1")
print("#" * 100)

spark.sql(""" 
        SELECT video_id,  SUM(likes) - SUM(dislikes) as dis_likes_difference from videos
        GROUP BY video_id
        ORDER BY dis_likes_difference DESC;          
          """).show(10)


# question 2
print("\nQUESTION 2")
print("#" * 100)



spark.sql(""" 
SELECT channel_title, AVG(likes) as avg_likes
FROM videos
GROUP BY channel_title
ORDER BY avg_likes DESC;
          """).show(10)



# question 3

print("\nQUESTION 3")
print("#" * 100)


spark.sql("""
SELECT channel_title, SUM(views) as total_views_count
FROM videos
group by channel_title
having SUM(views) > POWER(10, 6)
ORDER BY  total_views_count DESC;          
          """).show(10)



# question 4


print("\nQUESTION 4")
print("#" * 100)

spark.sql(""" 
        SELECT channel_title, AVG((size(split(tags, '[|]', -1)))) as avg_num_tags 
        FROM videos                
        GROUP BY channel_title
        ORDER BY avg_num_tags DESC;          
          """).show(10)


# question 5

print("\nQUESTION 5")
print("#" * 100)


spark.sql("""
        SELECT individual_tag, COUNT(*) as frequency FROM 
        (
                SELECT channel_title, explode(split(tags, '[|]', -1)) as individual_tag 
                FROM videos
        ) as t
        WHERE individual_tag != '[none]'
        GROUP BY individual_tag
        ORDER BY frequency DESC
        ;
""").show(10)


# question 6

print("\nQUESTION 6")
print("#" * 100)


spark.sql(""" 

SELECT channel_title, MAX(comment_count) as max_comments_count
FROM videos
GROUP BY channel_title
ORDER BY max_comments_count DESC

          """).show(10)
