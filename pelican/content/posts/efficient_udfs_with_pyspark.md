---
title: Efficient UD(A)Fs with PySpark
date: 2017-06-01 12:30
modified: 2017-06-01 19:30
category: article
tags: spark, python 
authors: Florian Wilhelm
status: draft
---

Spark nowadays surely is one of the most prevalent technologies in the field of data science. Luckily, even though it's built on a Java Stack it comes with Python bindings also known as [PySpark][] whose API was heavily influenced by Pandas. With respect to the functionality modern PySpark has about the same functionality as Pandas when it comes to typical ETL and data wrangling, e.g. groupby, aggregations and so on. A general rule of thumb is to use PySpark instead of Pandas whenever the data set has more than 10,000,000 rows which depending on the number of columns and data types translates to about 5-10 GB of memory usage. Typically, at that point you should consider leaving Pandas behind and PySpark is an option but of course there are others like for instance [Dask][]. 
  
If you are new to Spark one important thing to note is that Spark has two remarkable features besides its programmatic data wrangling capabilities. One is that Spark comes with SQL as an alternative way of defining queries and the other is [Spark.ml][] for machine learning. Both topics are beyond the scope of this post but should be considered if you are considering PySpark as an alternative to Pandas and Scikit-Learn for larger data sets. 
 
 But enough praise for PySpark, there are still some ugly sides as well as rough edges to it and we want to address some of them here, of course, in a constructive way. You might have heard the rumours that PySpark is so much slower compared to Spark with Scala and as it is often the case with rumours, there is a tiny bit of truth to it. But before we start, a deeper understanding of how PySpark does its magic is needed.
 
 PySpark is a wrapper around the actual Spark core written in Scala. That means when you call a method of a PySpark data frame, somewhere your Python call gets translated into the corresponding Scala call. This is in general extremely fast and the overhead can be neglected as long as you don't call the function millions of times. The more expensive part in terms of execution time though is the translation of complex object arguments that are passed to a function. That means if a Scala object like a list, dictionary or a row needs to be translated into a Python object and vice versa. This translation internally means that the object from one programming language is serialized into a memory location, then copied over to the memory space of the other programming language where it is deserialized again. Technically, it's even a bit more complicated since Scala and Python run in so called Virtual Machines (VMs) but you get the point. So practically we should keep these translations as low as possible. We can summarize this short low-level excursion with two important insights:
 
  1) Function calls are cheap but avoid excessive number of calls,
  2) Translation of complex objects is a tolerable overhead as long as it happens rarely.
  
With modern PySpark versions higher than 2.1, whenever you do data wrangling, like calling ``filter``, ``select``, ``groupby`` and so on, the overhead of the function call is neglectable as we have just learned. The arguments to those functions are mostly simple objects like strings or numbers defining the column name or indices and the costs of their translation to Scale is therefore also neglectable. Okay, so now in which cases do things go south then and when do we hit a major performance bumps? Let's say we have a data frame ``df`` of one billion rows with a boolean ``is_sold`` column and we want to filter for rows with sold products. One possibility would be to say
     
```python
df.filter(lambda x: x.is_sold == True)
```

which works but will be extremely slow because we just violated our rules 1) and 2) at the same time! Since df has one billion rows we need to evaluate our anonymous lambda function one billion times so Scala actually calls the Python lambda an excessive number of times clearly violating 1). But that's not nearly all, it even gets worse. We pass every row from Scala to Python and do the translation of that complex object also a billion times therefore violating rule 2)! The easy solution is do the operation without involving Python at all with:
   
```python
df.filter(df.is_sold == True)
```   

In this case our filter condition will be translation one time to Scale where it is then evaluated a billion times really fast, without any callback to Python!
To give a short summery, as long as we stick to the rules 1) and 2) a PySpark program will be approximately as fast as Spark program based on Scala.

Before we move on, two sides notes should also be kept in mind. The first is that what we just learnt not only applies to PySpark but also to Pandas, Numpy and Python in general since all these actually wrap a lot of C and sometimes even Fortran code. The second remark is that the general problem of object translation at least in the realm of data analytics is currently addressed by the creator of Pandas [Wes McKinney][]. His [Apache Arrow][] project tries to standardize the way complex objects are stored in memory so that everyone using Arrow won't need to do the cumbersome object translation by serialization and deserialization anymore. Hopefully with version 2.3 as shown in the issues [SPARK-13534][] and [SPARK-21190][] Spark will make use of Arrow and translation of complex objects like rows and data frames will have next to no overhead. Still, even in that case we should avoid making a large number of translations.
 
So far we have only talked about avoiding certain operations to keep the performance up. But what if we actually want to implement a User-Defined Function (UDF) or User-Defined Aggregation Function (UDAF)? The [databricks documentation][] explains how to define a UDF in PySpark in a few and easy steps but clearly violating our rules which leads to bad performance in practice: 
 
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType


def squared(s):
    return s * s


squared_udf = udf(squared, LongType())
df = sqlContext.table("test")
display(df.select("id", squared_udf("id").alias("id_squared")))
```
 
Since ``squared`` works on a single row, i.e. entry, of the ``id`` column the ``squared`` function will be called as many times as there are rows in the table ``test``. Since the ``id`` is an integer or long type and therefore a primitive type the translation overhead of a single number is almost neglectable but again we do it as many times as there are rows. Besides UDFs, there seems to be no "official" way of defining an arbitrary UDAF function. Depending on your use-case, this might even a reason to completely discard PySpark as a tool for it.
 
The obvious question is how can we tackle the problem of using UD(A)Fs without much of a performance hit? Looking at our little rule set, we see the pattern that if we do something with an overhead we should at least try to do it not so often. This leads directly leads us to the idea that a UD(A)F should do the object translation only a few times by working not on single rows but rather on whole partitions. This functionality is provided by the [RDD][] method ``mapPartitions``. As a short reminder, an Resilient Distributed Dataset (RDD) is the low-level data structures of Spark and a Spark [DataFrame][] is built on top of it. As we are mostly dealing with DataFrames in PySpark, we can get access to the underlying RDD with the help of the ``rdd`` attribute and convert it back with ``toDF()``. Putting all ingredients together we can apply an arbitrary Python function to a DataFrame ``df`` by:

```python
df.rdd.mapPartitions(my_func).toDF()
```
In most cases we would want to control the number of partitions, like 100, or even group by a column, let's say ``country``, in which case we would write:

```python
df.repartition(100).rdd.mapPartitions(my_func).toDF()
```

or

```python
df.repartition('country').rdd.mapPartitions(my_func).toDF()
```

This simply approach solves our main problem by doing the translation of each partition to Python at once, then calling the function ``my_func`` with it and translating back to Scala whatever the function returns. Therefore we have reduced the number of translations to two times (back and forth) the number of partitions and of course we should keep the number of partitions to a reasonable number.

Having solved one problem, as it is in life, we have introduced another problem. As we are working now with the low-level RDD interface our function ``my_func`` will be passed an iterator of PySpark [Row][] objects and needs to return them as well. A ``Row`` object itself is only a container for the column values in one row as you might have guessed and also the data types of these values must be interpretable by Spark in order to translate them back to Scala. This is a lot of low-level stuff to deal with when we actually only want to work with a **Pandas** DataFrame since gives us great flexibility to define our actual UD(A)F. Therefore we make a wish to the coding fairy, cross two fingers that someone else already solved this and start googling... and here we are ;-) 

So first we define a nice function that will convert a ``Row`` iterator into a Pandas DataFrame:

```python
import logging
import pandas as pd


def rows_to_pandas(rows):
    """Converts a Spark Row iterator of a partition to a Pandas DataFrame

    Args:
        rows: iterator over PySpark Row objects

    Returns:
        Pandas DataFrame
    """
    logger = logging.getLogger()

    first_row, rows = peek(rows)
    if not first_row:
        _logger.warning("Spark DataFrame is empty! Returning empty Pandas DataFrame!")
        return pd.DataFrame()

    first_row_info = ["{} ({}): {}".format(k, rtype(rtype(first_row[k])), first_row[k]) for k in first_row.__fields__]
    logger.debug("First partition row: {}".format(first_row_info))
    df = pd.DataFrame.from_records(rows, columns=first_row.__fields__)
    logger.debug("Converted partition to DataFrame of shape {} with types:\n{}".format(df.shape, df.dtypes))
    return df
```

So what the function actually does is calling ``df = pd.DataFrame.from_records(rows, columns=first_row.__fields__)`` in order to generate a DataFrame but we also need to make sure that the iterator is not empty and for debugging reasons we also peek into the first row and print the value as well as the datatype of each column. This has proven in practice to be extremely helpful in case something goes wrong and one needs to debug what's going on in the UD(A)F. The functions ``peek`` and ``rtype`` are defined as follows:

```python
from itertools import chain


def peek(iterable):
    """Peek into the first element and return the whole iterator again

    Args:
        iterable: iterable object like list or iterator

    Returns:
        tuple of first element and original iterable
    """
    iterable = iter(iterable)
    try:
        first_elem = next(iterable)
    except StopIteration:
        return None, iterable
    iterable = chain([first_elem], iterable)
    return first_elem, iterable
    
    
def rtype(var):
    """Heuristic representation for nested types/containers

    Args:
        var: some (nested) variable

    Returns:
        str: string representation of nested datatype (NA=Not Available)
    """
    def etype(x):
        return type(x).__name__

    if isinstance(var, list):
        elem_type = etype(var[0]) if var else "NA"
        return "List[{}]".format(elem_type)
    elif isinstance(var, dict):
        keys = list(var.keys())
        if keys:
            key = keys[0]
            key_type, val_type = etype(key), etype(var[key])
        else:
            key_type, val_type = "NA", "NA"
        return "Dict[{}, {}]".format(key_type, val_type)
    elif isinstance(var, tuple):
        elem_types = ', '.join(etype(elem) for elem in var)
        return "Tuple[{}]".format(elem_types)
    else:
        return etype(var)
```

The next part is to actually convert the result of our UD(A)F back to an iterator of Row objects. Since our results will most likely be Pandas DataFrames or Series, we define the following:

```python
from pyspark.sql.types import Row


def convert_dtypes(rows):
    """Converts some Pandas data types to pure Python data types

    Args:
        rows (array): numpy recarray holding all rows

    Returns:
        Iterator over lists of row values
    """
    dtype_map = {pd.Timestamp: lambda x: x.to_pydatetime(),
                 np.int8: lambda x: int(x),
                 np.int16: lambda x: int(x),
                 np.int32: lambda x: int(x),
                 np.int64: lambda x: int(x),
                 np.float16: lambda x: float(x),
                 np.float32: lambda x: float(x),
                 np.float64: lambda x: float(x),
                 np.float128: lambda x: float(x)}
    for row in rows:
        yield [dtype_map.get(type(elem), lambda x: x)(elem) for elem in row]
        
        
def pandas_to_rows(df):
    """Converts Pandas DataFrame to iterator of Row objects

    Args:
        df: Pandas DataFrame

    Returns:
        Iterator over PySpark Row objects
    """
    logger = logging.getLogger()
    if df is None:
        logger.debug("Returning nothing")
        return iter([])
    if type(df) is pd.Series:
        df = df.to_frame().T
    if df.empty:
        logger.warning("Pandas DataFrame is empty! Returning nothing!")
        return iter([])
    logger.debug("Convert DataFrame of shape {} to partition with types:\n{}".format(df.shape, df.dtypes))
    records = df.to_records(index=False)
    records = convert_dtypes(records)
    first_row, records = peek(records)
    first_row_info = ["{} ({}): {}".format(k, rtype(v), v) for k, v in zip(df.columns, first_row)]
    logger.debug("First record row: {}".format(first_row_info))
    row = Row(*df.columns)
    return (row(*elems) for elems in records)
```

This looks a bit more complicated but essentially we convert a Pandas Serie to a DataFrame if necessary and handle the case edge cases of an empty DataFrame or ``None`` as return value. We then convert the DataFrame to records, convert some Numpy data types to the Python equivalent and create an iterator over Row objects from the converted records. 

With these function at hand we can define a [Python decorator][] that will allow us to automatically call the functions ``rows_to_pandas`` and ``pandas_to_rows`` at the right time:

```python
from functools import wraps

class pandas_udaf(object):
    """Decorator for PySpark UDAFs using Pandas

    Args:
        loglevel (int): minimum loglevel for emitting messages
    """
    def __init__(self, loglevel=logging.INFO):
        self.loglevel = loglevel

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args):
            # use *args to allow decorating methods (incl. self arg)
            args = list(args)
            setup_logger(loglevel=self.loglevel)
            args[-1] = rows_to_pandas(args[-1])
            df = func(*args)
            return pandas_to_rows(df)
        return wrapper
```

The code is pretty much self-explanatory if you have ever written a Python decorator otherwise you should read about it since it takes some time to wrap your head around it. The only thing that might still raise questions is the usage of ``args[-1]``. This is due to the fact that ``func`` might also be a method of an object. In this case the first argument would be ``self`` but the last argument is in either cases the actual argument that ``mapPartitions`` will pass to us. 
 
Now having all parts in place we could define our UD(A)F ``my_func`` for instance as:

```python
import pyspark_utils

@pyspark_utils.pandas_udaf(loglevel=logging.DEBUG)
def my_func(df):
    if df.empty:
        return
    df = df.groupby('country').apply(lambda x: x.drop('country', axis=1).describe())
    return df.reset_index()
```

As this is just an example it is of course not really useful in practice to return some statistics with the help of a UD(A)F that could also be retrieved with basic PySpark functionality but you get the point. We now apply the function to each partition as above with:

```python
import pyspark_utils


stats_df = df.repartition('country').rdd.mapPartitions(my_func).toDF()
```

One should also note that this proposed method allows the definition of a UDF as well as an UDAF since it is up to the function ``my_func`` if it returns a data frame having as many rows as the input data frame (think [Pandas transform][]), a data frame of only a single row or optionally a series (think [Pandas aggregate][]) or a data frame with an arbitrary number of rows (think [Pandas apply][]) with even varying columns. Therefore, we can conclude that the proposed method is not only drastically faster than the official way in case of a UDF, it also even flexible enough to allow the definition of UDAFs.  

```python

VENV_DIR = '/user/fwilhelm/my_venv'
sc.addFile('./pyspark_utils.py')
distribute_hdfs_files('hdfs://' + VENV_DIR)

df = sc.parallelize(
    [('DEU', 2, 1.0), ('DEU', 3, 8.0), ('FRA', 2, 6.0), 
     ('FRA', 0, 8.0), ('DEU', 3, 8.0), ('FRA', 1, 3.0)]
    , 1).toDF(['country', 'feature1', 'feature2']).cache()
    
@pyspark_utils.pandas_udaf(loglevel=logging.DEBUG)
def my_func2(df):
    if df.empty:
        return
    return df.groupby('country').apply(lambda x: x.drop('country', axis=1).describe()).reset_index()
    
stats_df2 = df.repartition('country').rdd.mapPartitions(my_func2).toDF()
stats_df2.toPandas()
```


[PySpark]: https://spark.apache.org/docs/latest/api/python/index.html
[Dask]: http://dask.pydata.org/en/latest/index.html
[Wes McKinney]: http://wesmckinney.com/
[Apache Arrow]: http://arrow.apache.org/
[SPARK-13534]: https://issues.apache.org/jira/browse/SPARK-13534
[SPARK-21190]: https://issues.apache.org/jira/browse/SPARK-21190
[databricks documentation]: https://docs.databricks.com/spark/latest/spark-sql/udf-in-python.html
[Spark.ml]: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html
[DataFrame]: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
[Row]: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Row
[RDD]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD
[Python decorator]: https://wiki.python.org/moin/PythonDecorators#What_is_a_Decorator
[Pandas transform]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.transform.html
[Pandas aggregate]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.aggregate.html
[Pandas apply]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
