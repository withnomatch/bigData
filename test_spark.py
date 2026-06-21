from pyspark import SparkContext

sc = SparkContext("local[*]", "TestApp")
data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)
print("Sum:", rdd.sum())
sc.stop()
