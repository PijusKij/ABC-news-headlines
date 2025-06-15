import findspark
findspark.init()
findspark.find()

import pyspark

from pyspark.sql import DataFrame, SparkSession
from typing import List

spark= SparkSession \
       .builder \
       .appName("Our First Spark Example") \
       .getOrCreate()

from pyspark.sql.functions import col, lag, unix_timestamp, to_timestamp, when, sum as _sum, udf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = "abcnews-date-text.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
# df.show(5)

###################################################
# Gettign sentiment score for every article headline
###################################################
def get_sentiment_vader(text):
    """
    Returns sentiment using VADER analyzer
    VADER is specifically tuned for social media text and works well with news headlines
    """
    if text is None or text.strip() == "":
        return "neutral"
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_sentiment_score_vader(text):
    """
    Returns VADER compound sentiment score
    Range: -1 (most negative) to 1 (most positive)
    """
    if text is None or text.strip() == "":
        return 0.0
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return float(scores['compound'])

sentiment_vader_udf = udf(get_sentiment_vader, StringType())
sentiment_score_vader_udf = udf(get_sentiment_score_vader, FloatType())
df_with_vader_sentiment = df.withColumn("sentiment_vader", sentiment_vader_udf(col("headline_text"))) \
                           .withColumn("sentiment_score_vader", sentiment_score_vader_udf(col("headline_text")))

#######################################################
# Formatting data column to have all needed time columns
########################################################
df_analyzed = df_with_vader_sentiment.withColumn(
    "parsed_date", 
    to_date(col("publish_date").cast("string"), "yyyyMMdd")
)

# Extract temporal features
df_temporal = df_analyzed.withColumn("year", year("parsed_date")) \
                        .withColumn("month", month("parsed_date")) \
                        .withColumn("day_of_month", dayofmonth("parsed_date")) \
                        .withColumn("day_of_week", dayofweek("parsed_date")) \
                        .withColumn("day_name", date_format("parsed_date", "EEEE")) \
                        .withColumn("month_name", date_format("parsed_date", "MMMM")) \
                        .withColumn("week_of_year", weekofyear("parsed_date"))

df_temporal.cache()

###################################################
# Sentiment distribution bay day of month (1-31)
###################################################

print("\n=== 1. SENTIMENT PATTERNS BY DAY OF MONTH ===")

# Aggregate sentiment by day of month
day_of_month_sentiment = df_temporal.groupBy("day_of_month", "sentiment_vader") \
                                   .count() \
                                   .orderBy("day_of_month")

print("Sentiment distribution by day of month:")
day_of_month_sentiment.show(31)

# Average sentiment score by day of month
avg_sentiment_by_day = df_temporal.groupBy("day_of_month") \
                                 .agg(
                                     avg("sentiment_score_vader").alias("avg_sentiment"),
                                     count("*").alias("article_count"),
                                     sum(when(col("sentiment_vader") == "positive", 1).otherwise(0)).alias("positive_count"),
                                     sum(when(col("sentiment_vader") == "negative", 1).otherwise(0)).alias("negative_count"),
                                     sum(when(col("sentiment_vader") == "neutral", 1).otherwise(0)).alias("neutral_count")
                                 ) \
                                 .withColumn("positive_pct", round(col("positive_count") * 100.0 / col("article_count"), 2)) \
                                 .withColumn("negative_pct", round(col("negative_count") * 100.0 / col("article_count"), 2)) \
                                 .withColumn("neutral_pct", round(col("neutral_count") * 100.0 / col("article_count"), 2)) \
                                 .orderBy("day_of_month")

print("\nDetailed sentiment analysis by day of month:")
avg_sentiment_by_day.show(31)

###########################################
# Senriment distrivution by day of week (Sun=1, Sat=7)
#############################################

print("\n=== 2. SENTIMENT PATTERNS BY DAY OF WEEK ===")

# Aggregate sentiment by day of week
day_of_week_sentiment = df_temporal.groupBy("day_of_week", "day_name", "sentiment_vader") \
                                  .count() \
                                  .orderBy("day_of_week")

print("Sentiment distribution by day of week:")
day_of_week_sentiment.show()

# Detailed analysis by day of week
weekday_analysis = df_temporal.groupBy("day_of_week", "day_name") \
                             .agg(
                                 avg("sentiment_score_vader").alias("avg_sentiment"),
                                 count("*").alias("article_count"),
                                 sum(when(col("sentiment_vader") == "positive", 1).otherwise(0)).alias("positive_count"),
                                 sum(when(col("sentiment_vader") == "negative", 1).otherwise(0)).alias("negative_count"),
                                 sum(when(col("sentiment_vader") == "neutral", 1).otherwise(0)).alias("neutral_count")
                             ) \
                             .withColumn("positive_pct", round(col("positive_count") * 100.0 / col("article_count"), 2)) \
                             .withColumn("negative_pct", round(col("negative_count") * 100.0 / col("article_count"), 2)) \
                             .withColumn("neutral_pct", round(col("neutral_count") * 100.0 / col("article_count"), 2)) \
                             .orderBy("day_of_week")

print("\nDetailed sentiment analysis by day of week:")
weekday_analysis.show()

##################################################
# Weekend vs. Weekday and Beggining vs. Middle vs. End of the month sentiment 
##################################################

print("\n=== 4. ADVANCED PATTERN ANALYSIS ===")

# Weekend vs Weekday analysis
weekend_weekday = df_temporal.withColumn(
    "day_type", 
    when(col("day_of_week").isin([1, 7]), "Weekend").otherwise("Weekday")
).groupBy("day_type") \
 .agg(
     avg("sentiment_score_vader").alias("avg_sentiment"),
     count("*").alias("article_count"),
     sum(when(col("sentiment_vader") == "positive", 1).otherwise(0)).alias("positive_count"),
     sum(when(col("sentiment_vader") == "negative", 1).otherwise(0)).alias("negative_count")
 ) \
 .withColumn("positive_pct", round(col("positive_count") * 100.0 / col("article_count"), 2)) \
 .withColumn("negative_pct", round(col("negative_count") * 100.0 / col("article_count"), 2))

print("Weekend vs Weekday sentiment analysis:")
weekend_weekday.show()

# Beginning, middle, end of month analysis
month_period_analysis = df_temporal.withColumn(
    "month_period",
    when(col("day_of_month") <= 10, "Beginning")
    .when(col("day_of_month") <= 20, "Middle")
    .otherwise("End")
).groupBy("month_period") \
 .agg(
     avg("sentiment_score_vader").alias("avg_sentiment"),
     count("*").alias("article_count"),
     sum(when(col("sentiment_vader") == "positive", 1).otherwise(0)).alias("positive_count"),
     sum(when(col("sentiment_vader") == "negative", 1).otherwise(0)).alias("negative_count")
 ) \
 .withColumn("positive_pct", round(col("positive_count") * 100.0 / col("article_count"), 2)) \
 .withColumn("negative_pct", round(col("negative_count") * 100.0 / col("article_count"), 2))

print("Beginning/Middle/End of month sentiment analysis:")
month_period_analysis.show()

##################################################################
# Ranking best and worst days by sentiment and volume
##################################################################

print("\n=== 5. STATISTICAL INSIGHTS ===")

# Find days with highest/lowest sentiment
best_days = df_temporal.groupBy("day_name") \
                      .agg(avg("sentiment_score_vader").alias("avg_sentiment")) \
                      .orderBy(desc("avg_sentiment"))

print("Days of week ranked by sentiment (best to worst):")
best_days.show()

# Find days of month with highest/lowest sentiment
best_days_of_month = df_temporal.groupBy("day_of_month") \
                                .agg(avg("sentiment_score_vader").alias("avg_sentiment")) \
                                .orderBy(desc("avg_sentiment"))

print("Days of month ranked by sentiment (top 10):")
best_days_of_month.show(10)

print("Days of month ranked by sentiment (bottom 10):")
best_days_of_month.orderBy("avg_sentiment").show(10)

# Volume analysis
print("\n=== VOLUME ANALYSIS ===")

# Days with most/least articles
volume_by_day = df_temporal.groupBy("day_name") \
                          .count() \
                          .orderBy(desc("count"))

print("Article volume by day of week:")
volume_by_day.show()

volume_by_day_of_month = df_temporal.groupBy("day_of_month") \
                                   .count() \
                                   .orderBy(desc("count"))

print("Top 10 days of month by article volume:")
volume_by_day_of_month.show(10)



