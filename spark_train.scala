import org.apache.spark.sql.SparkSession
//spark-shell --driver-memory 2G --executor-memory 10G --executor-cores 8

val spark = SparkSession.builder()
	.appName("Ctr Prediction Model Training")
	.getOrCreate()

import spark.implicits._

// for testing dataset, randomly assign value between 1-5
// for testing, randomly sample 20 messages, rank and order? 
val ctrPredictTrain = spark.read.format("com.databricks.spark.csv")
	.option("delimiter", ",")
	.option("header", "true")
	.load("file:///<dir>/avazu-ctr-prediction/train")

// ctrPredictTrain.printSchema()

ctrPredictTrain.createOrReplaceTempView("CtrPredTrain")

val ctrPrediction = spark.sql("""
	SELECT * FROM CtrPredTrain""")

// // check if there are duplicated ad ID
// val countUniqueId = ctrPrediction.selectExpr("*").groupBy("id").count().sort(desc("count"))
// countUniqueId.take(5)


/***
*	Spark Pipeline for transforming the data
* 	Include one hot encoding for categorical variables
*	Combine all features into one feature vector column
*/

// transform categorial features into indexed feature
val stringFeatures = Array("site_domain","site_category","app_domain","app_category","device_model")
val catCol = Array("C1","banner_pos","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21")
val stringCatFeatures = stringFeatures ++ catCol
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
val catFeaturesIndexor = stringCatFeatures.map(
	cname => new StringIndexer()
		.setInputCol(cname)
		.setOutputCol(s"${cname}_index")
)

import org.apache.spark.ml.Pipeline
val indexPipeline = new Pipeline().setStages(catFeaturesIndexor)
val model = indexPipeline.fit(ctrPrediction)
val indexedDf = model.transform(ctrPrediction)
// indexedDf.show()

// One Hot Encoding for categorical features
val indexedCols = indexedDf.columns.filter(x => x contains "index")
val indexedFeatureEncoder = indexedCols.map(
	indexed_cname => new OneHotEncoder()
		.setInputCol(indexed_cname)
		.setOutputCol(s"${indexed_cname}_vec")
)

val encodedPipeline = indexPipeline.setStages(indexedFeatureEncoder)
val encodeModel = encodedPipeline.fit(indexedDf)
val encodedDf = encodeModel.transform(indexedDf)
// encodedDf.show()

// ad id is not a feature, neither is the output "click"
// also only keep encoded feature and ignore original cat features and their index
val nonFeatureCol = Array("id", "click", "site_id", "app_id", "device_id", "device_ip") // not using them
// val featureCol = ctrPrediction.columns.diff(nonFeatureCol) 
val featureCol = encodedDf.columns.filter(x => x contains "_vec")

// Combine all feature columns into a big column
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler().setInputCols(featureCol).setOutputCol("features")
val encodedTrainingSet = assembler.transform(encodedDf)

// Convert click to double
val finalTrainSet = encodedTrainingSet.selectExpr("*", "double(click) as click_output")


/***
*	Split data and ready for training and testing
*/

// split the data for training 
val Array(training, test) = finalTrainSet.randomSplit(Array(0.7, 0.3))

training.cache()
test.cache()

println(training.count())
println(test.count())

// Train the model
import org.apache.spark.ml.classification.LogisticRegression
val logisticRegModel = new LogisticRegression().setLabelCol("click_output").setFeaturesCol("features")
val lrFitted = logisticRegModel.fit(training)

// Test the Model
val holdout = lrFitted.transform(test)
val holdoutResult = holdout.selectExpr("id", "prediction", "click_output")
holdoutResult.cache()
val ranked = holdoutResult.filter(holdoutResult("prediction").between(0.1, 0.9))

// Save the Pipeline
lrFitted.write.save("./lrModelPipeline")

// load the Model. Notice that you need to use LogisticRegressionModel instead of LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
val lrFitted = LogisticRegressionModel.load("./Spark/lrModelPipeline")
