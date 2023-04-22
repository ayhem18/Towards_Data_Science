import pyspark 
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from os.path import join

# Import SparkSession
from pyspark.sql import SparkSession 

# Create SparkSession 
spark = SparkSession \
        .builder \
        .appName("my spark app") \
        .master("local[*]").getOrCreate()

sc = spark.sparkContext


print("\n\nQUESTION 1")
print("#" * 100)



json_object = sc.textFile("GB_category_id.json")
json_reduced = json_object.reduce(lambda x, y: "\n".join([x,y]))
import re
a = re.findall(r'"id".*:.*"(\w+)".*', json_reduced)
b = re.findall(r'"title".*:.*"(\w+)"', json_reduced)
categories = dict(zip(map(int, a),b))

print(categories)


print("\n\nQUESTION 2")
print("#" * 100)

df = spark.read.csv("GBvideos.csv", header=True)
res = df.rdd.map(lambda x: x['title']).take(10)
print(res)

print("\n\nQUESTION 3")
print("#" * 100)


res = df.rdd.map(lambda x: x['category_id']).take(20) # limiting the length of the output for visualization purposes
print(res)



print("\n\nQUESTION 4")
print("#" * 100)

res = df.rdd.map(lambda x: x['views']).take(20) # limiting the length of the output for visualization purposes
print(res)

print("\n\nQUESTION 5")
print("#" * 100)


df = spark.read.csv("GBvideos.csv", header=True)
res = df.rdd.map(lambda x: (x['title'], x['views'])).filter(lambda x: x[1] > 10 ** 6).take(5)
print(res)




print("\n\nQUESTION 6")
print("#" * 100)


json_object = sc.textFile("GB_category_id.json")
reduced = json_object.reduce(lambda x, y: "\n".join([x,y]))

ids = re.findall(r'"id".*:.*"(\w+)".*', reduced)
titles = re.findall(r'"title".*:.*"(\w+)"', reduced)
categories = dict(zip(map(int, ids),titles))

rdd2 = df.rdd.map(lambda x: (x['category_id'], x['views']))
rdd3 = rdd2.map(lambda x: categories[int(x[0])])
res = rdd3.take(5)
print(res)


print("\n\nQUESTION 7")
print("#" * 100)

res = df.rdd.map(lambda x: x['channel_title']).take(100)
print(res)

print("\n\nQUESTION 8")
print("#" * 100)


rdd0 = df.rdd.map(lambda x: (x['title'], x['views'], x['channel_title'],))
by_channel = rdd0.groupBy(lambda x: x[2])

sum_views = by_channel.map(lambda x: len(x[1])).reduce(lambda x, y: x + y)
count_videos = by_channel.map(lambda x: 1).reduce(lambda x, y: x + y)
avg_views_video = sum_views / count_videos

final_views = by_channel.filter(lambda x: len(x[1]) > avg_views_video).map(lambda x: x[0])
channels = final_views.collect()
final_res = rdd0.filter(lambda x: x[2] in channels).map(lambda x: x[0]).top(5)

print(final_res)


