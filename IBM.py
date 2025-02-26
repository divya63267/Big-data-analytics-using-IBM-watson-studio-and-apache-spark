# Big Data Analytics Using IBM Watson Studio and Apache Spark
# -------------------------------------------------------------

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, when, count, isnan
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import numpy as np

# Initialize Spark Session - this is how we connect to Spark in Watson Studio
spark = SparkSession.builder \
    .appName("IBM Watson Studio - Big Data Analytics") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

# For Watson Studio, we'd typically use the Object Storage to load data
# This example shows how you would connect to IBM Cloud Object Storage

def load_data_from_cos():
    """
    Function to load data from IBM Cloud Object Storage
    In Watson Studio, you'd typically use credentials from your project
    """
    # Normally, you would use credentials from your Watson Studio project
    # These settings would come from your IBM Cloud Object Storage service credentials
    cos_credentials = {
        "endpoint": "YOUR_COS_ENDPOINT",
        "access_key": "YOUR_ACCESS_KEY",
        "secret_key": "YOUR_SECRET_KEY"
    }
    
    # Configure Spark to read from Cloud Object Storage
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.cos.impl", 
                                                     "com.ibm.stocator.fs.ObjectStoreFileSystem")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.stocator.scheme.list", "cos")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.cos.service.endpoint", 
                                                     cos_credentials["endpoint"])
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.cos.service.access.key", 
                                                     cos_credentials["access_key"])
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.cos.service.secret.key", 
                                                     cos_credentials["secret_key"])
    
    # Read data from Cloud Object Storage
    # In a real application, this would be your COS bucket and file path
    df = spark.read.csv("cos://your-bucket-name.service/path/to/your/data.csv", 
                       header=True, inferSchema=True)
    
    return df

# For demonstration purposes, we'll create a synthetic dataset
# In a real Watson Studio project, you'd use real data from COS or other sources
def create_synthetic_data():
    """Create synthetic dataset for demonstration"""
    # Define schema
    schema = StructType([
        StructField("customer_id", IntegerType(), False),
        StructField("age", IntegerType(), True),
        StructField("income", DoubleType(), True),
        StructField("education", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("purchases", IntegerType(), True),
        StructField("credit_score", IntegerType(), True),
        StructField("churn_risk", StringType(), True)
    ])
    
    # Create synthetic data
    data = []
    np.random.seed(42)
    education_levels = ["High School", "Bachelor", "Master", "PhD"]
    occupations = ["Engineer", "Doctor", "Teacher", "Sales", "Manager", "Other"]
    churn_levels = ["Low", "Medium", "High"]
    
    for i in range(100000):  # Creating 100K records for big data simulation
        age = np.random.randint(18, 80)
        income = np.random.normal(60000, 20000)
        education = education_levels[np.random.randint(0, len(education_levels))]
        occupation = occupations[np.random.randint(0, len(occupations))]
        purchases = np.random.randint(0, 50)
        credit_score = np.random.randint(300, 850)
        
        # Create relationships in the data to make it realistic
        if age > 50 and income > 80000 and credit_score > 700:
            churn_risk = "Low"
        elif age < 30 and purchases < 10:
            churn_risk = "High"
        else:
            churn_risk = churn_levels[np.random.randint(0, len(churn_levels))]
            
        data.append((i, age, income, education, occupation, purchases, credit_score, churn_risk))
    
    # Create Spark DataFrame
    df = spark.createDataFrame(data, schema)
    
    # Add some missing values to make it more realistic
    df = df.withColumn("age", when(col("customer_id") % 10 == 0, None).otherwise(col("age")))
    df = df.withColumn("income", when(col("customer_id") % 15 == 0, None).otherwise(col("income")))
    
    return df

# Load or create data
# In practice, you would use load_data_from_cos() with your actual credentials
df = create_synthetic_data()

# Show basic information about the dataset
print("Dataset Overview:")
print(f"Number of records: {df.count()}")
print(f"Number of features: {len(df.columns)}")
df.printSchema()
df.show(5)

# Data Exploration and Analysis
print("\nData Summary Statistics:")
df.describe().show()

# Check for missing values
print("\nMissing Value Analysis:")
for column in df.columns:
    missing_count = df.filter(col(column).isNull() | isnan(col(column))).count()
    percentage = round((missing_count / df.count()) * 100, 2)
    print(f"Column: {column}, Missing Values: {missing_count}, Percentage: {percentage}%")

# Distribution of categorical variables
print("\nDistribution of Education:")
df.groupBy("education").count().orderBy("count", ascending=False).show()

print("\nDistribution of Occupation:")
df.groupBy("occupation").count().orderBy("count", ascending=False).show()

print("\nDistribution of Churn Risk:")
df.groupBy("churn_risk").count().orderBy("count", ascending=False).show()

# Data Preprocessing
# 1. Handle missing values
df = df.na.fill({
    "age": df.select("age").agg({"age": "avg"}).collect()[0][0],
    "income": df.select("income").agg({"income": "avg"}).collect()[0][0]
})

# 2. Convert categorical variables to numerical
indexer_education = StringIndexer(inputCol="education", outputCol="education_index")
indexer_occupation = StringIndexer(inputCol="occupation", outputCol="occupation_index")
indexer_churn = StringIndexer(inputCol="churn_risk", outputCol="churn_index")

# Apply indexers to transform the data
df = indexer_education.fit(df).transform(df)
df = indexer_occupation.fit(df).transform(df)
df = indexer_churn.fit(df).transform(df)

# 3. Create feature vector
feature_cols = ["age", "income", "education_index", "occupation_index", "purchases", "credit_score"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_transformed = assembler.transform(df)

# Split data into training and testing sets
train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)

# Build and train a machine learning model
# Using Random Forest Classifier to predict churn risk
rf_classifier = RandomForestClassifier(
    labelCol="churn_index",
    featuresCol="features",
    numTrees=100,
    maxDepth=10,
    seed=42
)

# Train the model
rf_model = rf_classifier.fit(train_data)

# Make predictions
predictions = rf_model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="churn_index",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Feature importance analysis
feature_importance = rf_model.featureImportances
features_list = feature_cols

# Convert to DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': features_list,
    'Importance': feature_importance.toArray()
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for idx, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Advanced Analysis: Customer Segmentation
# We'll use age and income to segment customers
# In a real application, you might use more advanced clustering techniques

def analyze_segments(df):
    """Perform customer segmentation analysis"""
    # Convert to Pandas for easier plotting and analysis
    pd_df = df.select("age", "income", "churn_risk", "purchases").toPandas()
    
    # Define segments
    pd_df['segment'] = 'Unknown'
    pd_df.loc[(pd_df['age'] < 30) & (pd_df['income'] < 50000), 'segment'] = 'Young Budget'
    pd_df.loc[(pd_df['age'] < 30) & (pd_df['income'] >= 50000), 'segment'] = 'Young Affluent'
    pd_df.loc[(pd_df['age'] >= 30) & (pd_df['age'] < 50) & (pd_df['income'] < 70000), 'segment'] = 'Mid-age Budget'
    pd_df.loc[(pd_df['age'] >= 30) & (pd_df['age'] < 50) & (pd_df['income'] >= 70000), 'segment'] = 'Mid-age Affluent'
    pd_df.loc[(pd_df['age'] >= 50) & (pd_df['income'] < 60000), 'segment'] = 'Senior Budget'
    pd_df.loc[(pd_df['age'] >= 50) & (pd_df['income'] >= 60000), 'segment'] = 'Senior Affluent'
    
    # Analyze churn risk by segment
    segment_analysis = pd_df.groupby(['segment', 'churn_risk']).size().unstack().fillna(0)
    
    # Calculate percentages
    for col in segment_analysis.columns:
        segment_analysis[f'{col}_pct'] = segment_analysis[col] / segment_analysis.sum(axis=1) * 100
    
    # Average purchases by segment
    avg_purchases = pd_df.groupby('segment')['purchases'].mean().sort_values(ascending=False)
    
    return pd_df, segment_analysis, avg_purchases

pd_df, segment_analysis, avg_purchases = analyze_segments(df)

print("\nCustomer Segments Analysis:")
print("\nDistribution of Customers by Segment:")
segment_counts = pd_df['segment'].value_counts()
for segment, count in segment_counts.items():
    print(f"{segment}: {count} customers ({count/len(pd_df)*100:.2f}%)")

print("\nChurn Risk Distribution by Segment (%):")
risk_columns = [col for col in segment_analysis.columns if '_pct' in col]
print(segment_analysis[risk_columns].round(2))

print("\nAverage Purchases by Segment:")
for segment, avg in avg_purchases.items():
    print(f"{segment}: {avg:.2f} items")

# Demonstrate how to save results back to Cloud Object Storage
# In a real application, you would save results to your actual COS bucket

def save_results_to_cos(predictions_df, model, feature_importance_df):
    """
    Save results back to IBM Cloud Object Storage
    In a real application, this would use your actual COS credentials
    """
    # Save predictions as CSV
    predictions_df.select("customer_id", "churn_risk", "prediction").write.csv(
        "cos://your-bucket-name.service/results/predictions.csv",
        header=True, mode="overwrite"
    )
    
    # Save model
    model.save("cos://your-bucket-name.service/models/churn_prediction_model")
    
    # For demonstration, we'll just print a message
    print("\nResults would be saved to IBM Cloud Object Storage in a real application.")

# For demonstration, we'll just print rather than actually saving
print("\nWould save results to IBM Cloud Object Storage in a real application.")

# Create a dashboard-like representation of key findings
print("\n" + "="*50)
print("BIG DATA ANALYTICS DASHBOARD")
print("="*50)
print(f"Total Customers Analyzed: {df.count():,}")
print(f"Model Prediction Accuracy: {accuracy*100:.2f}%")
print("\nTop 3 Factors Affecting Customer Churn:")
for idx, row in importance_df.head(3).iterrows():
    print(f"- {row['Feature']}: {row['Importance']*100:.2f}%")

print("\nCustomer Segments with Highest Churn Risk:")
high_risk_segments = segment_analysis.sort_values('High_pct', ascending=False).index.tolist()
for segment in high_risk_segments[:2]:
    high_pct = segment_analysis.loc[segment, 'High_pct']
    print(f"- {segment}: {high_pct:.2f}% high risk")

print("\nRecommended Actions:")
print("1. Target retention programs for " + high_risk_segments[0] + " segment")
print("2. Improve product features related to " + importance_df.iloc[0]['Feature'])
print("3. Develop loyalty programs for customers with high " + importance_df.iloc[1]['Feature'])
print("="*50)

# Clean up Spark session when done
spark.stop()