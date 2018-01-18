---
title: Using Python packages with PySpark
date: 2017-06-01 12:30
modified: 2017-06-01 19:30
category: article
tags: spark, python, production
authors: Florian Wilhelm
status: draft
---

With the sustained success of the Spark data processing platform even data scientists with a strong focus on the Python ecosystem can no longer ignore it. Fortunately with PySpark, official Python support for Spark is available and easy to use with millions of tutorials on the web explaining you how to count words. In contrast to that I found resources on how to deploy and use Python packages like Numpy, Pandas, Scikit-Learn in a PySpark program quite lacking. For most Spark/Hadoop distributions, in my case Cloudera, the best-practise arroding to Cloudera's [documentation][] and [blog][] seems to be that you (or rather a sysadmin) sets up a dedicated virtual environment (with [virtualenv][] or [conda][]) on all hosts of your cluster. This virtual environment can then be used by your PySpark application. The drawback of this approach are as severe as obvious. Firstly, either your data scientists have permission to access the actual cluster hosts or Cloudera Manager Admin Console with all implications or your sysadmins have a lot of fun setting up hundreds of virtual environments on a daily basis. Secondly, we are introducing a state in your Spark jobs which is always a root cause of errors in production:
 
 **DataScientist**: "I deployed my virtual envs on all hosts two weeks ago, now my production code fails occasionally with missing imports."<br />
 **SysAdmin**: "Well, we added a few more nodes a week ago... did you push your envs to those?"

In order to prevent situations like this we want to deploy our application with all its dependencies bundled every time we run it, just like you would to with a jar file in Scala. Since Python is not a compiled language this task sounds easier than it actually is. This observation is recognized as a problem and several issues ([SPARK-13587][] & [SPARK-16367][]) suggest solutions but none are implemented yet. So we are coming to the point were things get interesting and our goal is set. Coming up with a solution that allows bundling all requirements together with the actual PySpark application and of course it should not be too hacky ;-)

Luckily, PySpark provides the functions [sc.addFile][] and [sc.addPyFile][] which allow us uploading files to every node in our cluster, even Python modules and egg files in case of the latter. Unfortunately, there is no way to upload wheel files which are needed for binary Python packages like Numpy, Pandas and so on. As a data scientist you cannot live without those. At first sight this looks pretty bad but thanks to the wheel format all we have to do is upload with ``sc.addFile`` and unpack them, even the ``PYTHONPATH`` will be correctly set for us by PySpark. So in theory, we have already all the tools we need but how do we get the proper wheel files? First we check the Python version we want to use on Spark, in my case that is Python 3.4 on Cloudera cdh5.11.0. Now on some Linux, which needs to be compatible with the Linux on your Spark distribution, we create an Anaconda environment with the exact same Python version:

```bash
conda create -n py34 python=3.4
source activate py34
```

Having activated the environment, we just use ``pip download`` to download all the requirements of our PySpark application as well as the requirements of the requirements and so on. In case there is no wheel file available, ``pip`` will download a source-based ``tar.gz`` file instead but we can easily generate a wheel from it. To do so, we just unpack the archive, change into the directory and type ``python setup.py bdist_wheel``. A wheel file should now reside in the `dist` folder. Thereafter, we push all wheel files into some hdfs directory that is accessible by spark. For this example we will use ``hdfs:///absolute/path/to/wheelhouse``. At this point one should also be aware that some wheel files come with low-level Linux dependencies that just need to be installed by a sysadmin on every host, e.g. ``python3-dev`` and ``unixodbc-dev``.   

Up until now was only preliminary skirmishing, so let's get coding Python. We stick to the plan we laid out before, all we need to do is adding the files from our hdfs directory to the Spark context. Then, we unzip the files since wheel files are just plain zip files with a special structure and some meta information and that's about it. The code below will demonstrate this and often serves me as a basic template for a typical PySpark script. Therefore, the template also generates a ``SparkSession`` with Hive support, fires a query and converts it into a Pandas dataframe which can be removed of course if not needed. To execute the code just name it ``pyspark_with_py_pgks.py`` and run it with a command similar to this one:

```bash
PYSPARK_PYTHON=python3.4 /opt/spark/bin/spark-submit --master yarn --deploy-mode cluster \
--num-executors 4 --driver-memory 12g --executor-memory 4g --executor-cores 1 \
--files /etc/spark/conf/hive-site.xml --queue default --conf spark.yarn.maxAppAttempts=1 \
pyspark_with_py_pgks.py
```

The code is pretty much self-explanatory. If not, just drop me a line in the comments below:
```python
import os
import sys
import logging
from zipfile import ZipFile

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkFiles


_logger = logging.getLogger(__name__)


def add_pkg(path):
    SparkSession.builder.getOrCreate().sparkContext.addFile(path)
    root_dir = SparkFiles.getRootDirectory()
    file_name = os.path.basename(path)
    file_path = os.path.join(root_dir, file_name)
    with ZipFile(file_path, 'r') as zip_file:
        zip_file.extractall(root_dir)
    _logger.info("Added package {}".format(file_name))    


sess = (SparkSession
         .builder
         .appName("Python Spark Data Science Stack Check")
         .enableHiveSupport()
         .getOrCreate())

# setup basic logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# upload all dependencies
wheelhouse = 'hdfs:///absolute/path/to/wheelhouse'
pkgs = ['turbodbc-1.1.1-cp34-cp34m-linux_x86_64.whl', 
        'numpy-1.12.1-cp34-cp34m-manylinux1_x86_64.whl',
        'six-1.10.0-py2.py3-none-any.whl',
        'Jinja2-2.9.6-py2.py3-none-any.whl',
        'MarkupSafe-1.0-cp34-cp34m-linux_x86_64.whl',
        'cycler-0.10.0-py2.py3-none-any.whl',
        'pandas-0.20.1-cp34-cp34m-manylinux1_x86_64.whl',
        'pybind11-2.1.1-py2.py3-none-any.whl',
        'pyparsing-2.2.0-py2.py3-none-any.whl',
        'python_dateutil-2.6.0-py2.py3-none-any.whl',
        'pytz-2017.2-py2.py3-none-any.whl',
        'scikit_learn-0.18.1-cp34-cp34m-manylinux1_x86_64.whl',
        'scipy-0.19.0-cp34-cp34m-manylinux1_x86_64.whl']
for pkg in pkgs:
    add_pkg(os.path.join(wheelhouse, pkg))

# do the actual importing after adding the package
import numpy as np
import pandas as pd
import scipy as sp
import sklearn

# just some random query 
spark_df = sess.sql("SELECT * FROM my_database.my_table LIMIT 10")
pd_df = spark_df.toPandas()
print(pd_df)

# print some versions to check with above
print('Python', sys.version)
print('PySpark', pyspark.__version__)
print('NumPy', np.__version__)
print('SciPy', sp.__version__)
print('Pandas', pd.__version__)
print('Scikit-Learn', sklearn.__version__)

# check that NumPy really works ;-)
print(np.arange(10))
```

Das hier noch verwerten
http://blog.cloudera.com/blog/2015/09/how-to-prepare-your-apache-hadoop-cluster-for-pyspark-jobs/
https://stackoverflow.com/questions/37343437/how-to-run-a-function-on-all-spark-workers-before-processing-data-in-pyspark


sc._jsc.sc().getExecutorMemoryStatus().size()
'-'.join(a.split('-')[:2]) + ".dist-info"

```python
import os
import logging
from zipfile import ZipFile
from functools import wraps

from pyspark import SparkFiles
from pyspark.context import SparkContext


_logger = logging.getLogger(__name__)


class PyEnv(object):
    def __init__(self, wheelhouse, pkgs):
        self.wheelhouse = wheelhouse
        self.pkgs = pkgs
        self._init = False

    def init(self):
        for pkg in self.pkgs:
            path = os.path.join(self.wheelhouse, pkg)
            self.add_pkg(path)
            self.unpack_pkg(pkg)
        self._init = True
        return self

    @staticmethod
    def add_pkg(path):
        SparkContext.getOrCreate().addFile(path)
        _logger.info("Added file {}".format(path))

    @classmethod
    def unpack_pkg(cls, pkg):
        root_dir = SparkFiles.getRootDirectory()
        path = os.path.join(root_dir, pkg)
        with ZipFile(path, 'r') as zip_file:
            zip_file.extractall(root_dir)
        _logger.info("Extracted package {}".format(pkg))

    def __call__(self, func):
        assert self._init, "You need to run .init() first"

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError:
                for pkg in self.pkgs:
                    self.unpack_pkg(pkg)
                return func(*args, **kwargs)
        return wrapper
```


```python
import sys
import logging

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, collect_list, isnull, struct
from pyspark.sql.types import IntegerType, LongType, FloatType, MapType, StringType
from pyspark import SparkFiles

from pyenv import PyEnv

sess = (SparkSession
         .builder
         .appName("Python Spark SQL Hive integration example")
         .enableHiveSupport()
         .getOrCreate())
sc = sess.sparkContext

# setup basic logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

wheelhouse = 'hdfs:///company/MOBILE/data-platform/wheelhouse/'
pkgs = ['turbodbc-1.1.1-cp34-cp34m-linux_x86_64.whl',
        'numpy-1.12.1-cp34-cp34m-manylinux1_x86_64.whl',
        'six-1.10.0-py2.py3-none-any.whl',
        'Jinja2-2.9.6-py2.py3-none-any.whl',
        'MarkupSafe-1.0-cp34-cp34m-linux_x86_64.whl',
        'cycler-0.10.0-py2.py3-none-any.whl',
        'pandas-0.20.1-cp34-cp34m-manylinux1_x86_64.whl',
        'pybind11-2.1.1-py2.py3-none-any.whl',
        'pyparsing-2.2.0-py2.py3-none-any.whl',
        'python_dateutil-2.6.0-py2.py3-none-any.whl',
        'pytz-2017.2-py2.py3-none-any.whl',
        'scikit_learn-0.18.1-cp34-cp34m-manylinux1_x86_64.whl',
        'scipy-0.19.0-cp34-cp34m-manylinux1_x86_64.whl']

env = PyEnv(wheelhouse, pkgs).init()

# do the actual importing after adding the package
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
import turbodbc

@env
def squared(s):
    import numpy as np
    import pandas as pd
    rnd = np.random.randn()
    return s * rnd


@env
def count(struct):
    import pandas as pd
    series = pd.Series([elem.fuel for elem in struct])    
    dct =  series.value_counts().to_dict()
    return {k: int(v) for k, v in dct.items()}


squared_udf = udf(squared, FloatType())
count_udf = udf(count, MapType(StringType(), LongType()))
df = sess.table("user_profiles_prod.monitor_profiles")
df.select("sub_count", squared_udf("sub_count").alias("sub_count_squared")).show()

df = sess.table("fwilhelm.profile_test")
df = (df.groupby("uid")
        .agg(collect_list(struct('fuel')).alias('struct'))
        .withColumn('count', count_udf('struct')))
df.show()

pd_df = df.toPandas()
print(pd_df)
print('PySpark', pyspark.__version__)
print('Numpy', np.__version__)
print('Pandas', pd.__version__)
print('Scikit-Learn', sklearn.__version__)
print('turbodbc', turbodbc.api_constants.apilevel)
print('SciPy', sp.__version__)

# check that numpy really works ;-)
print(np.arange(10))
```

```bash
PYSPARK_PYTHON=python3.4 /opt/spark/bin/spark-submit --master yarn --deploy-mode cluster --num-executors 4 --driver-memory 12g --executor-memory 4g --executor-cores 2 --files /etc/spark/conf/hive-site.xml --queue default --conf spark.yarn.maxAppAttempts=1 --py-files pyenv.py test.py
```

ToDo:
- erwaehnen wie man die replication auf der venv erhoeht.
- unterschied driver und executor erwaehnen

[documentation]: https://www.cloudera.com/documentation/enterprise/5-6-x/topics/spark_python.html#spark_python__section_kr2_4zs_b5
[blog]: http://blog.cloudera.com/blog/2016/02/making-python-on-apache-hadoop-easier-with-anaconda-and-cdh/
[sc.addFile]: http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext.addFile
[sc.addPyFile]: http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext.addPyFile
[SPARK-13587]: https://issues.apache.org/jira/browse/SPARK-13587
[SPARK-16367]: https://issues.apache.org/jira/browse/SPARK-16367
[virtualenv]: https://virtualenv.pypa.io/en/stable/
[conda]: https://conda.io/docs/intro.html



```python
17/10/16 09:01:41 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/monkey.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp3019861664343394710.tmp
17/10/16 09:01:42 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/msvc.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp445035189056040521.tmp
17/10/16 09:01:42 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/namespaces.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp7793235992674656634.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/package_index.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp6781084544681609696.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/py26compat.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp8768907866875545192.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/py27compat.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp829494871114433661.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/py31compat.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp6429354488562794884.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/py33compat.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp5117886574805787520.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/py36compat.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp696156835681758986.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/sandbox.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp3609628508002798111.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/script (dev).tmpl to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp4267162028247811916.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/script.tmpl to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp8644342590871886186.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/site-patch.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp4409440928622133996.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/ssl_support.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp6855231276625909069.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/unicode_utils.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcache/application_1506607653040_28748/spark-cba2d06c-17c3-4bdc-8822-4c9e580082a0/-12144712641508143654572_cache/fetchFileTemp1280669822013434575.tmp
17/10/16 09:01:43 INFO util.Utils: Fetching hdfs://nameservice1/user/mo-hadoop-uc-perso/venv/v1.0.x/setuptools/version.py to /data/dn01/yarn/usercache/mo-hadoop-uc-perso/appcac
```


Siehe auch die Praesentation des Hadoop summits

dann auch handling von config files mit 

def spark_files_path(path='./'):
    """Given a file path return the actual filepath as stored with SparkFiles

    Args:
        path (str): conical path as used to push a file

    Returns:
        str: actual path where the file was stored
    """
    return os.path.join(SparkFiles.getRootDirectory(), path)


