# fog/spark_consumer.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder \
    .appName("SOMELEC_Fog") \
    .getOrCreate()

# قراءة البيانات المباشرة من Kafka
stream_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "village_1_data,village_2_data") \
    .load()

# تحويل البيانات
parsed_df = stream_df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json("json", schema).alias("data")) \
    .select("data.*")

# كشف الشذوذات
assembler = VectorAssembler(
    inputCols=["voltage", "current", "power"],
    outputCol="features"
)

anomalies = model.transform(assembler.transform(parsed_df))

# عرض النتائج
query = anomalies \
    .writeStream \
    .format("console") \
    .start()

query.awaitTermination()