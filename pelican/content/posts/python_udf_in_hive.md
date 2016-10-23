---
title: Hive UDFs and UDAFs with Python
date: 2016-10-23 11:00
modified: 2016-10-23 11:00
category: post
tags: python, hadoop, hive, big data
authors: Florian Wilhelm
---

Sometimes the analytical power of [built-in Hive functions][1] is just not enough.
In this case it is possible to write hand-tailored User-Defined Functions (UDFs)
for transformations and even aggregations which are therefore called User-Defined
Aggregation Functions (UDAFs). In this post we focus on how to write sophisticated
UDFs and UDAFs in Python. By sophisticated we mean that our UD(A)Fs should
also be able to leverage external libraries like Numpy, Scipy, Pandas etc.
This makes things a lot more complicated since we have to provide not only some
Python script but also a full-blown virtual environment including the external
libraries. Therefore, we require only from the actual Hive setup that
a basic installation of Python is available on the data nodes.


## General information

To keep the idea behind UD(A)Fs short, only some general notes are mentioned here.
With the help of the [Transform/Map-Reduce syntax][2], i.e. ``TRANSFORM``, it is
possible to plug in own custom mappers and reducers. This is where we gonna hook
in our Python script. An UDF is basically only a transformation done by a mapper
meaning that each row should be mapped to exactly one row. A UDAF on the
other hand allows us to transform a group of rows into one or more rows so we
can reduce the number of input rows to a single output row by some custom
aggregation. We can control if the script is run in a mapper or reducer step
by the way we formulate our HiveQL query. The statements ``DISTRIBUTE BY`` and
``CLUSTER BY`` allow us to indicate that we want to actually perform an aggregation.
HiveQL feeds its data to the Python script or any other custom script by using
the standard input and reads the result from its standard out. All messages from
standard error are ignored and can therefore be used for debugging.
Since a UDAF is more complex than a UDF and actually can be seen as a generalization
of it, the development of an UDAF is demonstrated here.   


## Overview and our little task

In order to not get lost in the details, here is what we want to achieve from
a high-level perspective.

1.  Set up small example Hive table within some database.
2.  Create a virtual environment and upload it to Hive's distributed cache.
3.  Write the actual UDAF as Python script and a little helper shell script.
4.  Write a HiveQL query that feeds our example table into the Python script.

Our dummy data consists of different types of vehicles (car or bike) and a price. For
each category we want to calculate mean and the standard deviation with the help
of Pandas to keep things simple. It should not be necessary to mention that this
task can be handled in HiveQL directly, so this is really only for demonstration.


## 1. Setting up our dummy table

With the following query we generate our sample data:

```sql
CREATE DATABASE tmp;
USE tmp;
CREATE TABLE foo (id INT, vtype STRING, price FLOAT);
INSERT INTO TABLE foo VALUES (1, "car", 1000.);
INSERT INTO TABLE foo VALUES (2, "car", 42.);
INSERT INTO TABLE foo VALUES (3, "car", 10000.);
INSERT INTO TABLE foo VALUES (4, "car", 69.);
INSERT INTO TABLE foo VALUES (5, "bike", 1426.);
INSERT INTO TABLE foo VALUES (6, "bike", 32.);
INSERT INTO TABLE foo VALUES (7, "bike", 1234.);
INSERT INTO TABLE foo VALUES (8, "bike", null);
```
Note that the last row even contains a null value that we need to handle later.


## 2. Creating and uploading a virtual environment

In order to prepare a proper virtual environment we need to execute the following
steps on an OS that is binary compatible to the OS on the Hive cluster. Typically
any recent 64bit Linux distribution will do.

We start by creating an empty virtual environment with:
> virtualenv --no-site-packages -p /usr/bin/python3 venv

assuming that `virtualenv` was already installed with the help of pip. Note that
we explicitly ask for Python 3. Who uses Python 2 these days anyhow?
We activate the virtual environment and install Pandas in it.
> source venv/bin/activate

> pip install numpy pandas

This should install Pandas and all its dependencies into our virtual environment.
No we package the virtual environment for later deployment in the distributed cache:
> cd venv

> tar cvfhz ../venv.tgz ./

> cd ..

Be aware that the archive was created with the actual content at its root so
when unpacking there will be no directory holding the actual content. We also
used the parameter `h` to package linked files.

Now we push the archive to HDFS so that later Hive's data nodes will be able to
find it:
> hdfs dfs -put venv.tgz /tmp

The directory `/tmp` should be changed accordingly. One should also note that
in principle the same procedure should also be possible with conda environments. In
practice though, it might be a bit more involved since the activation of a conda
environment (what we need to do later) assumes an installation of at least
miniconda which might not be available on the data nodes.


## 3. Writing and uploading the scripts

We start by writing a simple Python script `udaf.py`:

```python
import sys
import logging
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd

SEP = '\t'
NULL = '\\N'

_logger = logging.getLogger(__name__)


def read_input(input_data):
    for line in input_data:
        yield line.strip().split(SEP)


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    data = read_input(sys.stdin)
    for vtype, group in groupby(data, itemgetter(1)):
        _logger.info("Reading group {}...".format(vtype))
        group = [(int(rowid), vtype, np.nan if price == NULL else float(price))
                 for rowid, vtype, price in group]
        df = pd.DataFrame(group, columns=('id', 'vtype', 'price'))
        output = [vtype, df['price'].mean(), df['price'].std()]
        print(SEP.join(str(o) for o in output))


if __name__ == '__main__':
    main()
```
The script should be pretty much self-explanatory. We read from the standard
input with the help of a generator that strips and splits the lines by the
separator `\t`. At any point we want to avoid to have more data in memory as
needed to perform the actual computation. We use the ``groupby`` function that
is shipped with Python to iterate over our two types of vehicles. For each group
we convert the read values to their respective data types and at that point
also take care of `null` values which are encoded as `\N`. After this preprocessing
we finally feed everything into a Pandas dataframe, do our little mean and standard
deviation calculations and print everything as a tabular separated list.
It should also be noted that we set up a logger at the beginning which writes
everything to standard error. This really helps a lot with debugging and should
be used. For demonstration purposes the vehicle type of the group currently
processed is printed.

At this point we would actually be done if it wasn't for the fact that we are
importing external libraries like Pandas. So if we ran this Python script directly
as UDAF we would see import errors if Pandas is not installed on all cluster nodes.
But in the spirit of David Wheeler's "All problems in computer science can be
solved by another level of indirection." we just write a little helper script
called `udaf.sh` that does this job for us and calls the Python script afterwards.

```sh
#!/bin/bash
set -e
(>&2 echo "Begin of script")
source ./venv.tgz/bin/activate
(>&2 echo "Activated venv")
./venv.tgz/bin/python3 udaf.py
(>&2 echo "End of script")
```
Again we use standard error to trace what the script is currently doing.
With the help of `chmod u+x` we make the script executable and now all that's
left is to push both files somewhere on HDFS for the cluster to find:
> hdfs dfs -put udaf.py /tmp

> hdfs dfs -put udaf.sh /tmp


## 4. Writing the actual HiveQL query

After we are all prepared and set we can write the actual HiveQL query:

```sql
DELETE ARCHIVE hdfs:///tmp/venv.tgz;
ADD ARCHIVE hdfs:///tmp/venv.tgz;
DELETE FILE hdfs:///tmp/udaf.py;
ADD FILE hdfs:///tmp/udaf.py;
DELETE FILE hdfs:///tmp/udaf.sh;
ADD FILE hdfs:///tmp/udaf.sh;

USE tmp;
SELECT TRANSFORM(id, vtype, price) USING 'udaf.sh' AS (vtype STRING, mean FLOAT, var FLOAT)
  FROM (SELECT * FROM foo CLUSTER BY vtype) AS TEMP_TABLE;
```

At first we add the zipped virtual environment to the distributed cache that
will be automatically unpacked for us due to the `ADD ARCHIVE` command.
Then we upload the Python and helper script. To make sure the current version
in the cache is actually the latest, so in case changes are made, we
prepended `DELETE` statements before each `ADD`.

The actual query now calls `TRANSFORM` with the three input column we expect
in our Python script. After the `USING` statement our helper script is provided
as the actual UDAF seen by HiveQL. This is followed by `AS` defining the names
and types of the output columns.

At this point we need to make sure that the script is executed in a reducer step.
We assure this by defining a subselect that reads from our `foo` table and clusters
by the `vtype`. `CLUSTER BY` which is a shortcut for `DISTRIBUTE BY` followed by
`SORT BY` asserts that rows having the same `vtype` column are also located on
the same reducer. Furthermore, the implicit `SORT BY` orders within a reducer
the rows with respect to the `vtype` column. The overall result are consecutive
partitions of a given vehicle type (car and bike in our case) whereas each partition resides
on a single reducer. Finally, our script is fed the whole data on a single reducer
and needs to figure out itself where one partition ends and another one starts
(what we did with `itertools.groupby`).


## Finally

Since our little task is now accomplished, it should also be noted that there
are some more Python libraries one should know when working with Hive.
To actually execute the HiveQL query we have written with the help of Python, there
is [impyla][3] by Cloudera with supports Python 3 in contrast to [PyHive][4] by Dropbox.
In order to work with HDFS the best library around is [hdfs3][5]. That would
for instance allow us to push changes in `udaf.py` automatically with a Python
script.

Have fun hacking Hive with the power of Python!

[1]: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF
[2]: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Transform
[3]: https://github.com/cloudera/impyla
[4]: https://github.com/dropbox/PyHive
[5]: https://hdfs3.readthedocs.io/
