//1
import org.apache.spark.sql.SparkSession
//2
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//3
val spark = SparkSession.builder().getOrCreate()
//4
import org.apache.spark.ml.clustering.KMeans
//5
val df = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesalecustomersdata.csv")
//6
val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
//7
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
//8
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
//9
val training_data = assembler.transform(feature_data).select("features")
//10
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)
//11
val WSSE = model.computeCost(training_data)
println(s"Within set sum of Squared Errors = $WSSE")
//12
model.clusterCenters.foreach(println)
