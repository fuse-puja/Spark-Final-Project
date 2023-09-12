# %%
from pyspark.sql.window import Window
import pyspark.sql.functions as f
from textblob import TextBlob
from pyspark.sql.types import StringType
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder.appName('FinalTask')\
        .config('spark.driver.extraClassPath','/usr/lib/jvm/java-11-openjdk-amd64/lib/postgresql-42.6.0.jar')\
        .getOrCreate()
# spark.sparkContext.setLogLevel("DEBUG")


# ### reading from postgres

# %%
reviews_df = spark.read.format("jdbc").options(url='jdbc:postgresql://localhost:5432/Cleaned_Data',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'reviews', 
                                user='postgres',
                                password='postgres').load()

orders_df = spark.read.format("jdbc").options(url='jdbc:postgresql://localhost:5432/Cleaned_Data',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'orders', 
                                user='postgres',
                                password='postgres').load()

order_item_df = spark.read.format("jdbc").options(url='jdbc:postgresql://localhost:5432/Cleaned_Data',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'order_item', 
                                user='postgres',
                                password='postgres').load()


product_df = spark.read.format("jdbc").options(url='jdbc:postgresql://localhost:5432/Cleaned_Data',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'product', 
                                user='postgres',
                                password='postgres').load()


# %%
delivered_orders_df = orders_df.filter(orders_df["Order_status"] == "delivered")

# %%
# Extract the hour of the day from the timestamp
delivered_orders_df = delivered_orders_df.withColumn("Order_purchase_hour", f.hour("Order_purchase_timestamp"))
delivered_orders_df.show()

# %%
delivered_orders_df = delivered_orders_df.withColumn(
    "delivery_time_days",
    f.round((f.unix_timestamp("order_delivered_customer_date") - f.unix_timestamp("order_purchase_timestamp")) / (24 * 3600),3)
)
delivered_orders_df = delivered_orders_df.withColumn(
    "delivery_deviation_in_days",
    f.round((f.unix_timestamp("order_estimated_delivery_date") - f.unix_timestamp("order_delivered_customer_date")) / (24 * 3600),3)
)

delivered_orders_df.show()

#Average delivery time 
average_delivery_time = delivered_orders_df.selectExpr("avg(delivery_time_days) as avg_delivery_time").first()["avg_delivery_time"]
print("Average Delivery Time (in days):", average_delivery_time)

# %%
# Create a column for the time slot 
delivered_orders_df = delivered_orders_df.withColumn("Order_purchase_time_slot",
    f.when((delivered_orders_df["Order_purchase_hour"] >= 0) & (delivered_orders_df["Order_purchase_hour"] <= 6), "Dawn")
    .when((delivered_orders_df["Order_purchase_hour"] >= 7) & (delivered_orders_df["Order_purchase_hour"] <= 12), "Morning")
    .when((delivered_orders_df["Order_purchase_hour"] >= 13) & (delivered_orders_df["Order_purchase_hour"] <= 18), "Afternoon")
    .otherwise("Night")
)
delivered_orders_df.show()

# %%
delivery_status = delivered_orders_df.select("order_id", "order_purchase_timestamp", "order_delivered_customer_date","delivery_deviation_in_days","delivery_time_days","Order_purchase_time_slot")
delivery_status.show()

# %% [markdown]
# ### Question 2

# %% [markdown]
# 

# %%
count_no_comment = reviews_df.filter(f.col("review_comment_message") == "no comment").count()

print("Count of 'No comment':", count_no_comment)

# %% [markdown]
# ### Perform sentiment analysis on the comment

# %%
def analyze_sentiment(text):
    if text.lower() != "no comment":
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        return polarity
    return None  # Return None for "no comment" comments

# %%
sentiment_udf = f.udf(analyze_sentiment, StringType())

# %%
reviews_df = reviews_df.withColumn("sentiment_score", sentiment_udf(reviews_df["review_comment_message"]))

reviews_df=reviews_df.withColumn("sentiment_score", reviews_df["sentiment_score"].cast("float"))


# Filter out rows with sentiment scores (exclude "no comment" comments)
filtered_reviews_df = reviews_df.filter(reviews_df["sentiment_score"].isNotNull())


# Show the filtered DataFrame with sentiment scores
filtered_reviews_df.show()


# %%
correlation = filtered_reviews_df.select(f.corr("sentiment_score", "review_score")).first()[0]
# Create a DataFrame to store the correlation result
correlation_df = spark.createDataFrame([(correlation,)], ["Correlation"])

# Show the correlation result
correlation_df.show()
print("Correlation between sentiment_score and review_score:", correlation)


results_df = filtered_reviews_df.join(order_item_df, "order_id", "inner")
joined_df= results_df.join(product_df, "product_id", "inner")
joined_df = joined_df.select("order_id", "review_id","review_score", "review_comment_message","product_category_name","sentiment_score")

joined_df.show()

# %%


joined_df = joined_df.withColumn(
    "sentiment",
    f.when(joined_df["sentiment_score"] > 0, "positive")
    .when(joined_df["sentiment_score"] == 0, "neutral")
    .otherwise("negative")
)

joined_df.show()



delivery_status.write.format('jdbc').options(url='jdbc:postgresql://localhost:5432/Output',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'delivery_status_table', 
                                user='postgres', 
                                password='postgres').mode('overwrite').save()

joined_df.write.format('jdbc').options(url='jdbc:postgresql://localhost:5432/Output',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'review_sentiment_analysis', 
                                user='postgres', 
                                password='postgres').mode('overwrite').save()

correlation_df.write.format('jdbc').options(url='jdbc:postgresql://localhost:5432/Output',
                                driver = 'org.postgresql.Driver',
                                dbtable = 'correlation_of_reviewscore_and_sentiment', 
                                user='postgres', 
                                password='postgres').mode('overwrite').save()

