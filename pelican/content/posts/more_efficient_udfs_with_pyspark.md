---
title: More Efficient UD(A)Fs with PySpark
date: 2019-10-11 12:30
modified: 2019-03-08 12:30
category: post
tags: spark, python, big data
authors: Florian Wilhelm
status: draft
---

Some time has passed since my blog post on [Efficient UD(A)Fs with PySpark]({filename}/posts/efficient_udfs_with_pyspark.md) which demonstrated how to define *User-Defined Aggregation Function* (UDAF) with [PySpark][] 2.1 that allow you to use [Pandas][]. Meanwhile, things got a lot easier with the release of Spark 2.3 which provides the [pandas_udf][] decorator. This decorator gives you the same functionality as our custom `pandas_udaf` in the former post but performs much faster if [Apache Arrow][] is activated. *Nice, so life is good now? No more workarounds!? Well, almost...*

If you are just using simple datatypes in your Spark dataframes everything will work and even blazingly fast if you got Arrow activated but don't you dare dealing with complex datatypes like maps (dictionaries), arrays (lists) and structs. In that case, all you will get is a `TypeError: Unsupported type in conversion to Arrow` which is already tracked under issue [SPARK-21187]. Even a simple `toPandas()` does not work which might get you to deactivate Arrow support altogether but this would also keep you from using `pandas_udf` which is really nice... 

To save you from this dilemma, this blog post will demonstrate how to work around the current limitations of Arrow without too much hassle. I tested this on Spark 2.3 and it should also work on Spark 2.4. But before we start, let's first take a look into which features `pandas_udf` provides and why we should make use of it.


# Features of `pandas_udf`

Just to give you a little overview about the functionality, take a look at the table below.

| function type | Operation   | Input → Output        | Pandas equivalent   |
|---------------|-------------|-----------------------|---------------------|
| `SCALAR`      | Mapping     | Series → Series       | `df.transform(...)` |
| `GROUPED_MAP` | Group & Map | DataFrame → DataFrame | `df.apply(...)`     |
| `GROUPED_AGG` | Reduce      | Series → Scalar       | `df.aggregate(...)` |

Besides the return type of your UDF, the `pandas_udf` needs you to specify a function type which describes the general behavior of your UDF. If you just want to map a scalar onto a scalar or equivalently a vector onto a vector with the same length, you would pass `PandasUDFType.SCALAR`. This would also determine that your UDF retrieves a Pandas series as input and needs to return a series of the same length. It basically does the same as the `transform` method of a Pandas dataframe. A `GROUPED_MAP` UDF is the most flexible one since it gets a Pandas dataframe and is allowed to return a modified or new dataframe with an arbitrary shape. From Spark 2.4 on you also have the reduce operation `GROUPED_AGG` which takes a Pandas Series as input and needs to return a scalar. Read more details in the [official Spark documentation][].

# Basic idea

<figure>
<p align="center">
<img class="noZoom" src="/images/pandas_udf_complex.png" alt="Converting complex data types to JSON before applying the UDF">
</p>
</figure>


```python
df.filter(df.is_sold == True)
```


[PySpark]: https://spark.apache.org/docs/latest/api/python/index.html
[Pandas]: http://pandas.pydata.org/
[pandas_udf]: http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf
[SPARK-21187]: https://jira.apache.org/jira/browse/SPARK-21187
[Python decorator]: https://wiki.python.org/moin/PythonDecorators#What_is_a_Decorator
[Pandas transform]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.transform.html
[Pandas aggregate]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.aggregate.html
[Pandas apply]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
[pyspark23_udaf.py]: {filename}/src/pyspark23_udaf.py
[official Spark documentation]: https://spark.apache.org/docs/2.4.0/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf
