# -*- coding: utf-8 -*-
"""
UDFs with complex data types for Spark version 2.3 and higher.

If you want to use `pandas_udf` for Spark UDFs the current problem is that no complex
data types are supported as cell values like MapType, ArrayType or StructType.

In order to still make use of it we convert those columns to strings for the UDF.
A minimal example is::

    # create some dummy data
    df = spark.createDataFrame([(1., {'a': 1}, ["a", "a"], Row(a=1)),
                                (2., {'b': 1}, ["a", "b"], Row(a=42)),
                                (3., {'a': 1, 'b': 3}, ["d","e"], Row(a=1))],
                               schema=['vals', 'maps', 'lists', 'structs'])
    df = df.drop('lists')  # only Spark 2.4 supports ArrayTypes in to_json!
    df.show()

    # convert complex columns to JSON strings
    df_json, ct_cols = complex_dtypes_to_json(df)

    # define your UDF
    def change_vals(dct):
        dct['x'] = 42
        return dct

    @pandas_udf_ct(df_json.schema, PandasUDFType.GROUPED_MAP)  # doctest: +SKIP
    def normalize(pdf):
        pdf['maps'].apply(change_vals)
        return pdf

    # apply the udf as usual
    df_json = df_json.groupby("vals").apply(normalize(cols_in=ct_cols, cols_out=ct_cols))

    # convert back to complex types from string
    df_final = complex_dtypes_from_json(df_json, ct_cols)
    df_final.show()
"""
import os
import sys
import json
from functools import wraps
import logging
import logging.config
from pkg_resources import parse_version

import pandas as pd
import pyspark
from pyspark.sql.types import MapType, StructType, ArrayType, StructField
from pyspark.sql.functions import pandas_udf, PandasUDFType, to_json, from_json

__author__ = "Florian Wilhelm"
__copyright__ = "Florian Wilhelm"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def pyspark_version():
    """Retrieves the PySpark version as object

    Returns:
        PySpark version
    """
    version = pyspark.version.__version__
    return parse_version(version)


def setup_logger(loglevel=logging.INFO, logfile="pyspark.log"):
    """Setup basic logging for logging on the executor

    Args:
        loglevel (int): minimum loglevel for emitting messages
        logfile (str): name of the logfile
    """
    logformat = "%(asctime)s %(levelname)s %(module)s.%(funcName)s: %(message)s"
    datefmt = "%y/%m/%d %H:%M:%S"
    try:
        logfile = os.path.join(os.environ['LOG_DIRS'].split(',')[0], logfile)
    except (KeyError, IndexError):
        logging.basicConfig(level=loglevel,
                            stream=sys.stdout,
                            format=logformat,
                            datefmt=datefmt)
        logger = logging.getLogger(__name__)
        logger.error("LOG_DIRS is not in environment variables or empty, using STDOUT instead.")
    else:
        logging.basicConfig(level=loglevel,
                            filename=logfile,
                            format=logformat,
                            datefmt=datefmt)


class DropDuplicates(logging.Filter):
    """Avoids duplicate debug messages when using UDAFs

    Usage:
        logger.addFilter(DropDuplicates())
    """
    MAX_SEEN = 100_000
    already_seen = set()

    def filter(self, record):
        first_line = record.getMessage().splitlines()[0]
        if len(self.already_seen) > self.MAX_SEEN:
            self.already_seen.clear()
        not_seen = first_line not in self.already_seen
        if not_seen:
            self.already_seen.add(first_line)
        return not_seen


def is_complex_dtype(dtype):
    """Check is dtype is a complex type

    Args:
        dtype: Spark data type

    Returns:
        Bool: if dtype is complex
    """
    return isinstance(dtype, (MapType, StructType, ArrayType))


def complex_dtypes_to_json(df):
    """Converts all columns with complex dtypes to JSON

    Args:
        df: Spark dataframe

    Returns:
        Spark dataframe and optionally list of converted columns
    """
    conv_cols = dict()
    selects = list()
    for field in df.schema:
        if is_complex_dtype(field.dataType):
            if pyspark_version() < parse_version("2.4") and isinstance(field.dataType, ArrayType):
                raise RuntimeError("to_json conversion of ArrayType is only supported for Spark 2.4 and higher!")
            conv_cols[field.name] = field.dataType
            selects.append(to_json(field.name).alias(field.name))
            _logger.info("Converted column {} from {} to JSON string".format(field.name, field.dataType))
        else:
            selects.append(field.name)
    df = df.select(*selects)
    return df, conv_cols


def complex_dtypes_from_json(df, col_dtypes):
    """Converts JSON columns to complex types

    Args:
        df: Spark dataframe
        col_dtypes (dict): dictionary of columns names and their data type

    Returns:
        Spark dataframe
    """
    selects = list()
    for column in df.columns:
        if column in col_dtypes.keys():
            schema = StructType([StructField('root', col_dtypes[column])])
            selects.append(from_json(column, schema).getItem('root').alias(column))
            _logger.info("Converted column {} from JSON string to {}".format(column, col_dtypes[column]))
        else:
            selects.append(column)
    return df.select(*selects)


def toPandas(df):
    """Same as df.toPandas() but converts complex types to JSON first

    Args:
        df: Spark dataframe

    Returns:
        Pandas dataframe
    """
    return complex_dtypes_to_json(df)[0].toPandas()


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
        columns (iter): list of or iterator over column names

    Returns:
        dataframe: new dataframe with converted columns
    """
    for column in columns:
        df[column] = df[column].apply(ct_val_to_json)
    return df


def cols_from_json(df, columns):
    """Converts Pandas dataframe colums from json

    Args:
        df (dataframe): Pandas DataFrame
        columns ([str]): list of column names

    Returns:
        dataframe: new dataframe with converted columns
    """
    for column in columns:
        df[column] = df[column].apply(json.loads)
    return df


class pandas_udf_ct(object):
    """Decorator for UDAFs with Spark >= 2.3 and complex types

    Args:
        returnType: the return type of the user-defined function. The value can be either a pyspark.sql.types.DataType
                    object or a DDL-formatted type string.
        functionType: an enum value in pyspark.sql.functions.PandasUDFType. Default: SCALAR.
        logLevel: logging level for UDF

    Returns:
        Function with arguments `cols_in` and `cols_out` defining column names having complex types
        that need to be transformed during input and output for GROUPED_MAP. In case of SCALAR, we are
        dealing with a series and thus transformation is done if `cols_in`/`cols_out` evaluates to `True`.
        Calling this functions with these arguments returns the actual UDF.

    """

    def __init__(self, returnType=None, functionType=None, logLevel=logging.INFO):
        self.return_type = returnType
        self.function_type = functionType
        self.log_level = logLevel

    def __call__(self, func):
        @wraps(func)
        def converter(*, cols_in=None, cols_out=None):
            if cols_in is None:
                cols_in = list()
            if cols_out is None:
                cols_out = list()

            @pandas_udf(self.return_type, self.function_type)
            def udf_wrapper(values):
                setup_logger(loglevel=self.log_level)
                fname = func.__name__
                logger = logging.getLogger(__name__)
                logger.addFilter(DropDuplicates())
                if isinstance(values, pd.DataFrame):
                    values = cols_from_json(values, cols_in)
                    logger.info("UDF-{} data types of dataframe when entering:\n{}".format(fname, values.dtypes))
                    logger.info("UDF-{} first row of dataframe when entering:\n{}".format(fname, values.iloc[0]))
                elif isinstance(values, pd.Series) and cols_in:
                    values = values.apply(json.loads)
                    logger.info("UDF-{} data type of series when entering:\n{}".format(fname, values.dtypes))
                    logger.info("UDF-{} series when entering:\n{}".format(fname, values))
                res = func(values)
                if self.function_type == PandasUDFType.GROUPED_MAP:
                    if isinstance(res, pd.Series):
                        res = res.to_frame().T
                    logger.info("UDF-{} data types of dataframe when leaving:\n{}".format(fname, res.dtypes))
                    logger.info("UDF-{} first row of dataframe when leaving:\n{}".format(fname, res.iloc[0]))
                    res = cols_to_json(res, cols_out)
                elif cols_out and self.function_type == PandasUDFType.SCALAR:
                    logger.info("UDF-{} data types of series when leaving:\n{}".format(fname, res.dtypes))
                    logger.info("UDF-{} series when leaving:\n{}".format(fname, res))
                    res = res.apply(ct_val_to_json)
                elif isinstance(res, (dict, list)) and self.function_type == PandasUDFType.GROUPED_AGG:
                    res = ct_val_to_json(res)
                return res

            return udf_wrapper

        return converter
