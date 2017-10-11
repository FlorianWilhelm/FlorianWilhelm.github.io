# -*- coding: utf-8 -*-
"""
User-defined Aggregation Functions (UDAF) for PySpark

Usage example assuming that pyspark_udaf.py is available to executors and driver:

import pyspark_udaf
import logging


@pyspark_udaf.pandas_udaf(loglevel=logging.DEBUG)
def my_func(df):
    if df.empty:
        return
    df = df.groupby('country').apply(lambda x: x.drop('country', axis=1).describe())
    return df.reset_index()
    

# make pyspark_udaf.py available to the executors
sc.addFile('./pyspark_udaf.py') 

df = sc.parallelize(
    [('DEU', 2, 1.0), ('DEU', 3, 8.0), ('FRA', 2, 6.0), 
     ('FRA', 0, 8.0), ('DEU', 3, 8.0), ('FRA', 1, 3.0)]
    , 1).toDF(['country', 'feature1', 'feature2']).cache()
    
stats_df = df.repartition('country').rdd.mapPartitions(my_func).toDF()
print(stats_df.toPandas())
"""

import os
import sys
import logging
from itertools import chain
from functools import wraps

import numpy as np
import pandas as pd
from pyspark.sql.types import Row


_logger = logging.getLogger(__name__)


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


def rows_to_pandas(rows):
    """Converts a Spark Row iterator of a partition to a Pandas DataFrame

    Args:
        rows: iterator over PySpark Row objects

    Returns:
        Pandas DataFrame
    """
    first_row, rows = peek(rows)
    if not first_row:
        _logger.warning("Spark DataFrame is empty! Returning empty Pandas DataFrame!")
        return pd.DataFrame()

    first_row_info = ["{} ({}): {}".format(k, rtype(first_row[k]), first_row[k])
                      for k in first_row.__fields__]
    _logger.debug("First partition row: {}".format(first_row_info))
    df = pd.DataFrame.from_records(rows, columns=first_row.__fields__)
    _logger.debug("Converted partition to DataFrame of shape {} with types:\n{}".format(df.shape, df.dtypes))
    return df


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
    if df is None:
        _logger.debug("Returning nothing")
        return iter([])
    if type(df) is pd.Series:
        df = df.to_frame().T
    if df.empty:
        _logger.warning("Pandas DataFrame is empty! Returning nothing!")
        return iter([])
    _logger.debug("Convert DataFrame of shape {} to partition with types:\n{}".format(df.shape, df.dtypes))
    records = df.to_records(index=False)
    records = convert_dtypes(records)
    first_row, records = peek(records)
    first_row_info = ["{} ({}): {}".format(k, rtype(v), v) for k, v in zip(df.columns, first_row)]
    _logger.debug("First record row: {}".format(first_row_info))
    row = Row(*df.columns)
    return (row(*elems) for elems in records)


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
        logger.error("LOG_DIRS is not in environment variables or empty, using {} instead.".format(logfile))

    logging.basicConfig(level=loglevel,
                        filename=logfile,
                        format=logformat,
                        datefmt=datefmt)


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
