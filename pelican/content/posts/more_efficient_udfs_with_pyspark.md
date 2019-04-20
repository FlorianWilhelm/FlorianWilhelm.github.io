---
title: More Efficient UD(A)Fs with PySpark
date: 2019-04-19 12:30
modified: 2019-04-19 12:30
category: post
tags: spark, python, big data
authors: Florian Wilhelm
status: published
summary: With the release of Spark 2.3 implementing user defined functions with PySpark became a lot easier and faster. Unfortunately, there are still some rough edges when it comes to complex data types that need to be worked around.
---

Some time has passed since my blog post on [Efficient UD(A)Fs with PySpark]({filename}/posts/efficient_udfs_with_pyspark.md) which demonstrated how to define *User-Defined Aggregation Function* (UDAF) with [PySpark][] 2.1 that allow you to use [Pandas][]. Meanwhile, things got a lot easier with the release of Spark 2.3 which provides the [pandas_udf][] decorator. This decorator gives you the same functionality as our custom `pandas_udaf` in the former post but performs much faster if [Apache Arrow][] is activated. *Nice, so life is good now? No more workarounds!? Well, almost...*

If you are just using simple data types in your Spark dataframes everything will work and even blazingly fast if you got Arrow activated but don't you dare dealing with complex data types like maps (dictionaries), arrays (lists) and structs. In that case, all you will get is a `TypeError: Unsupported type in conversion to Arrow` which is already tracked under issue [SPARK-21187]. Even a simple `toPandas()` does not work which might get you to deactivate Arrow support altogether but this would also keep you from using `pandas_udf` which is really nice... 

To save you from this dilemma, this blog post will demonstrate how to work around the current limitations of Arrow without too much hassle. I tested this on Spark 2.3 and it should also work on Spark 2.4. But before we start, let's first take a look into which features `pandas_udf` provides and why we should make use of it.


## Features of Spark 2.3's pandas\_udf

Just to give you a little overview about the functionality, take a look at the table below.

| function type | Operation   | Input → Output        | Pandas equivalent   |
|---------------|-------------|-----------------------|---------------------|
| `SCALAR`      | Mapping     | Series → Series       | `df.transform(...)` |
| `GROUPED_MAP` | Group & Map | DataFrame → DataFrame | `df.apply(...)`     |
| `GROUPED_AGG` | Reduce      | Series → Scalar       | `df.aggregate(...)` |

Besides the return type of your UDF, the `pandas_udf` needs you to specify a function type which describes the general behavior of your UDF. If you just want to map a scalar onto a scalar or equivalently a vector onto a vector with the same length, you would pass `PandasUDFType.SCALAR`. This would also determine that your UDF retrieves a Pandas series as input and needs to return a series of the same length. It basically does the same as the `transform` method of a Pandas dataframe. A `GROUPED_MAP` UDF is the most flexible one since it gets a Pandas dataframe and is allowed to return a modified or new dataframe with an arbitrary shape. From Spark 2.4 on you also have the reduce operation `GROUPED_AGG` which takes a Pandas Series as input and needs to return a scalar. Read more details about `pandas_udf` in the [official Spark documentation][].

## Basic idea

Our workaround will be quite simple. We make use of the [to_json][] function and convert all columns with complex data types to JSON strings. Since Arrow can easily handle strings, we are able to use the [pandas_udf][] decorator. Within our UDF, we convert these columns back to their original types and do our actual work. If we want to return columns with complex types, we just do everything the other way around. That means we convert those columns to JSON within our UDF, return the Pandas dataframe and convert eventually the corresponding columns in the Spark dataframe from JSON to complex types with [from_json][]. The following figure illustrates the process.

<figure>
<p align="center">
<img class="noZoom" src="/images/pandas_udf_complex.png" alt="Converting complex data types to JSON before applying the UDF">
</p>
</figure>

Our workaround involves a lot of bookkeeping and surely is not that user-friendly. Like we did in the last blog post, it is again possible to hide much of the details with the help of a [Python decorator][] from a user. So let's get started!

## Implementation

We split our implementation into three different kinds of functionalities: 1. functions that convert a Spark dataframe to and from JSON, 2. functions that do the same for Pandas dataframes and 3. we combine all of them in one decorator. The final and extended implementation can be found in the file [pyspark23_udaf.py][] where also some logging mechanism for easier debugging of UDFs was added. 

### 1. Conversion of Spark Dataframe

```python
from pyspark.sql.types import MapType, StructType, ArrayType, StructField
from pyspark.sql.functions import to_json, from_json

def is_complex_dtype(dtype):
    """Check if dtype is a complex type

    Args:
        dtype: Spark Datatype

    Returns:
        Bool: if dtype is complex
    """
    return isinstance(dtype, (MapType, StructType, ArrayType))


def complex_dtypes_to_json(df):
    """Converts all columns with complex dtypes to JSON

    Args:
        df: Spark dataframe

    Returns:
        tuple: Spark dataframe and dictionary of converted columns and their data types
    """
    conv_cols = dict()
    selects = list()
    for field in df.schema:
        if is_complex_dtype(field.dataType):
            conv_cols[field.name] = field.dataType
            selects.append(to_json(field.name).alias(field.name))
        else:
            selects.append(field.name)
    df = df.select(*selects)
    return df, conv_cols


def complex_dtypes_from_json(df, col_dtypes):
    """Converts JSON columns to complex types

    Args:
        df: Spark dataframe
        col_dtypes (dict): dictionary of columns names and their datatype

    Returns:
        Spark dataframe
    """
    selects = list()
    for column in df.columns:
        if column in col_dtypes.keys():
            schema = StructType([StructField('root', col_dtypes[column])])
            selects.append(from_json(column, schema).getItem('root').alias(column))
        else:
            selects.append(column)
    return df.select(*selects)
```

The function `complex_dtypes_to_json` converts a given Spark dataframe to a new dataframe with all columns that have complex types replaced by JSON strings. Besides the converted dataframe, it also returns a dictionary with column names and their original data types which where converted. This information is used by `complex_dtypes_from_json` to convert exactly those columns back to their original type. You might find it strange that we define some `root` node in the schema. This is necessary due to some restrictions of Spark's [from_json][] that we circumvent by this. After the conversion, we drop this `root` struct again so that `complex_dtypes_to_json` and `complex_dtypes_from_json` are inverses of each other. We can now also easily define a `toPandas` which also works with complex Spark dataframes.

```python
def toPandas(df):
    """Same as df.toPandas() but converts complex types to JSON first

    Args:
        df: Spark dataframe

    Returns:
        Pandas dataframe
    """
    return complex_dtypes_to_json(df)[0].toPandas()
```

### 2. Conversion of Pandas Dataframe

Analogously, we define the same functions as above but for Pandas dataframes. The difference is that we need to know which columns to convert to complex types for our actual UDF since we want to avoid probing every column containing strings. In the conversion to JSON, we add the `root` node as explained above. 

```python
import json

def cols_from_json(df, columns):
    """Converts Pandas dataframe colums from json

    Args:
        df (dataframe): Pandas DataFrame
        columns (iter): list of or iterator over column names

    Returns:
        dataframe: new dataframe with converted columns
    """
    for column in columns:
        df[column] = df[column].apply(json.loads)
    return df
    

def ct_val_to_json(value):
    """Convert a scalar complex type value to JSON

    Args:
        value: map or list complex value

    Returns:
        str: JSON string
    """
    return json.dumps({'root': value})


def cols_to_json(df, columns):
    """Converts Pandas dataframe columns to json and adds root handle

    Args:
        df (dataframe): Pandas DataFrame
        columns ([str]): list of column names

    Returns:
        dataframe: new dataframe with converted columns
    """
    for column in columns:
        df[column] = df[column].apply(ct_val_to_json)
    return df
```


### 3. Decorator

At this point we got everything we need for our final decorators named `pandas_udf_ct` combining all our ingredients. Like Spark's official [pandas_udf][], our decorator takes the arguments `returnType` and `functionType`. It's just a tad more complicated in the sense that you first have to pass `returnType`, `functionType` which leaves you with some special decorator. A function decorated with such a decorator takes the parameters `cols_in` and `cols_out` which specify which columns need to be converted to and from JSON. Only after passing those you end up with the actual UDF that you defined. No need to despair, an example below illustrates the usage but first we take a look at the implementation.

```python
import json
from functools import wraps
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

class pandas_udf_ct(object):
    """Decorator for UDAFs with Spark >= 2.3 and complex types

    Args:
        returnType: the return type of the user-defined function. The value can be either a 
                    pyspark.sql.types.DataType object or a DDL-formatted type string.
        functionType: an enum value in pyspark.sql.functions.PandasUDFType. Default: SCALAR.

    Returns:
        Function with arguments `cols_in` and `cols_out` defining column names having complex 
        types that need to be transformed during input and output for GROUPED_MAP. In case of 
        SCALAR, we are dealing with a series and thus transformation is done if `cols_in` or 
        `cols_out` evaluates to `True`. 
        Calling this functions with these arguments returns the actual UDF.
    """

    def __init__(self, returnType=None, functionType=None):
        self.return_type = returnType
        self.function_type = functionType

    def __call__(self, func):
        @wraps(func)
        def converter(*, cols_in=None, cols_out=None):
            if cols_in is None:
                cols_in = list()
            if cols_out is None:
                cols_out = list()

            @pandas_udf(self.return_type, self.function_type)
            def udf_wrapper(values):
                if isinstance(values, pd.DataFrame):
                    values = cols_from_json(values, cols_in)
                elif isinstance(values, pd.Series) and cols_in:
                    values = values.apply(json.loads)
                res = func(values)
                if self.function_type == PandasUDFType.GROUPED_MAP:
                    if isinstance(res, pd.Series):
                        res = res.to_frame().T
                    res = cols_to_json(res, cols_out)
                elif cols_out and self.function_type == PandasUDFType.SCALAR:
                    res = res.apply(ct_val_to_json)
                elif (isinstance(res, (dict, list)) and 
                      self.function_type == PandasUDFType.GROUPED_AGG):
                    res = ct_val_to_json(res)
                return res

            return udf_wrapper

        return converter
```

It's just a typical decorator-with-parameters implementation but with one more layer of wrapping for `cols_in` and `cols_out`.  

## Usage

An example says more than one thousand words of explanation. Let's first create some dummy Spark dataframe with complex data types:

```python
from pyspark.sql.types import Row
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([(1., {'a': 1}, ["a", "a"], Row(a=1)),
                            (2., {'b': 1}, ["a", "b"], Row(a=42)),
                            (3., {'a': 1, 'b': 3}, ["d","e"], Row(a=1))],
                           schema=['vals', 'maps', 'lists', 'structs'])
df = df.drop('lists')  # only Spark 2.4 supports ArrayTypes in to_json!
df.show()
```

For sake of simplicity, let's say we just want to add to the dictionaries in the `maps` column a key `x` with value `42`. We define a UDF `normalize` and decorate it with our `pandas_udf_ct` specifying the return type using `dfj_json.schema` (since we only want simple data types) and the function type `GROUPED_MAP`. 

```python
def change_vals(dct):
    dct['x'] = 42
    return dct

@pandas_udf_ct(df_json.schema, PandasUDFType.GROUPED_MAP)
def normalize(pdf):
    pdf['maps'].apply(change_vals)
    return pdf

```
To apply this UDF, we first use `complex_dtypes_to_json` to get a converted Spark dataframe `df_json` and the converted columns `ct_cols`. Just for demonstration, we now group by the `vals` column and apply our `normalize` UDF on each group. Instead of just passing `normalize` we have to call it first with parameters `cols_in` and `cols_out` as explained before. As input columns we pass the output `ct_cols` from our `complex_dtypes_to_json` function and since we do not change the shape of our dataframe within the UDF, we use the same for the output `cols_out`. In case your UDF removes columns or adds additional ones with complex data types, you would have to change `cols_out` accordingly. As a final step we use `complex_dtypes_from_json` to convert the JSON strings of our transformed Spark dataframe back to complex data types.
```python
df_json, ct_cols = complex_dtypes_to_json(df)

df_json = df_json.groupby("vals").apply(normalize(cols_in=ct_cols, cols_out=ct_cols))

df_final = complex_dtypes_from_json(df_json, ct_cols)
df_final.show()
```

## Conclusion

We have shown a practical workaround to deal with UDFs and complex data types for Spark 2.3/4. As with every workaround, it's far from perfect and hopefully the issue [SPARK-21187] will be resolved soon rendering this workaround unnecessary. That being said, the presented workaround has been running smoothly in production for quite a while now and my data science colleagues adapted this framework to write their own UDFs based on it.


[from_json]: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=to_json#pyspark.sql.functions.from_json
[to_json]: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=to_json#pyspark.sql.functions.to_json
[PySpark]: https://spark.apache.org/docs/latest/api/python/index.html
[Pandas]: http://pandas.pydata.org/
[pandas_udf]: http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf
[SPARK-21187]: https://jira.apache.org/jira/browse/SPARK-21187
[Python decorator]: https://wiki.python.org/moin/PythonDecorators#What_is_a_Decorator
[Pandas transform]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.transform.html
[Pandas aggregate]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.aggregate.html
[Pandas apply]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
[pyspark23_udaf.py]: {filename}/src/pyspark23_udaf.py
[official Spark documentation]: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf
[Apache Arrow]: https://arrow.apache.org/
