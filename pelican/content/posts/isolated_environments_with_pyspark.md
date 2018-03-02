---
title: Managing isolated Environments with PySpark
date: 2018-03-01 12:30
modified: 2018-03-01 19:30
category: article
tags: spark, python, production
authors: Florian Wilhelm
status: draft
---

With the sustained success of the Spark data processing platform even data scientists with a strong focus on the Python ecosystem can no longer ignore it. Fortunately with PySpark, official Python support for Spark is available and easy to use with millions of tutorials on the web explaining you how to count words. 

In contrast to that, resources on how to deploy and use Python packages like Numpy, Pandas, Scikit-Learn in an isolated environment with PySpark are scarce. A nice exception to that is a [blog post by Eran Kampf][].

For most Spark/Hadoop distributions, in my case Cloudera, the best-practise according to Cloudera's [documentation][] and two blog posts by post by [Juliet Hougland et al.][] and [Juliet Hougland][] seems to be that you (or rather a sysadmin) sets up a dedicated virtual environment (with [virtualenv][] or [conda][]) on all nodes of your cluster. This virtual environment can then be used by your PySpark application. The drawback of this approach are as severe as obvious. Either your data scientists have permission to access the actual cluster hosts or the Cloudera Manager Admin Console with all implications following from this or your sysadmins have a lot of fun setting up hundreds of virtual environments on a daily basis. 

Therefore, we need to empower the data scientists developing a predictive application to manage isolated environments with their dependencies themselves. This was also recognized as a problem and several issues ([SPARK-13587][] & [SPARK-16367][]) suggest solutions but none are implemented yet. The most matured solution is actually [coffee boat] which is still in beta and not meant for production. Therefore, we want to present a simple but viable solution for this problem that we have in production for more than a year.

So how can we distribute Python modules and whole packages on our executors? Luckily, PySpark provides the functions [sc.addFile][] and [sc.addPyFile][] which allow us to upload files to every node in our cluster, even Python modules and egg files in case of the latter. Unfortunately, there is no way to upload wheel files which are needed for binary Python packages like Numpy, Pandas and so on. As a data scientist you cannot live without those. 

At first sight this looks pretty bad but thanks to the simplicity of the wheel format it's not so bad at all. So here is what we do in a nutshell: We create a directory named ``venv`` which will serve us as isolated environment and copy the wheel files of all our dependencies into it. Then we unpack the wheel files with ``unzip`` since they are just normal zip files with a strange suffix. After this we copy the ``venv`` directory to some HDFS location and run ``sc.addFile`` from PySpark for the directory every time we run our application.

Let's go through this in more details. Since wheel files contain compiled code they are dependent on the exact Python version and platform. For us this means that when we create the ``venv`` environment locally we have to make sure that we use the same platform and Python version. Let's say we are running Cloudera cdh5.11.0 which ships Python 3.4. That means on a Linux system we would first create an Anaconda environment with the exact same version.

```bash
conda create -n py34 python=3.4
source activate py34
```

Having activated the environment, we just use ``pip download`` to download all the requirements of our PySpark application as wheel files. In case there is no wheel file available, ``pip`` will download a source-based ``tar.gz`` file instead but we can easily generate a wheel from it. To do so, we just unpack the archive, change into the directory and type ``python setup.py bdist_wheel``. A wheel file should now reside in the `dist` folder. At this point one should also be aware that some wheel files come with low-level Linux dependencies that just need to be installed by a sysadmin on every host, e.g. ``python3-dev`` and ``unixodbc-dev``.   

Now we unpack those wheel files in the ``venv`` directory, push it to HDFS, e.g. ``/my_venvs/venv``, using ``hdfs dfs -put ...`` and make sure that the files are readable by anyone. Now we call ``sc.addFile`` on every file in ``/my_venvs/venv`` and subsequently importing Pandas fori instance inside a Python UDF on PySpark will work since ``PYTHONPATH`` will be automatically set.

If our Python application itself is also nicely structured as a Python package (maybe using [PyScaffold][]) we can also push it to ``/my_venvs/venv``. This allows us to split the code that sets ups the environment on PySpark with ``sc.addFile`` from our actual application. Let's say our actual PySpark application is a Python package called ``my_pyspark_app``. The boilerplate code to bootstrap ``my_pyspark_app``, i.e. to activate the isolated environment on Spark, will be in the module ``activate_env.py``. When we submit our Spark job will specify this module and specify the environment as argument, e.g.:

```bash
PYSPARK_PYTHON=python3.4 /opt/spark/bin/spark-submit --master yarn --deploy-mode cluster \
--num-executors 4 --driver-memory 12g --executor-memory 4g --executor-cores 1 \
--files /etc/spark/conf/hive-site.xml --queue default --conf spark.yarn.maxAppAttempts=1 \
activate_env.py /my_venvs/venv
```

And here is how ``activate_env.py`` looks like:
```python
"""
Bootstrap file for running the `my_pyspark_app` on Spark
"""
import os
import sys
import logging

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

_logger = logging.getLogger(__name__)


def list_path_names(path):
    """List files and directories in an HDFS path

    Args:
        path (str): HDFS path to directory

    Returns:
        [str]: list of file/directory names
    """
    sc = SparkContext.getOrCreate()
    # low-level access to hdfs driver
    hadoop = sc._gateway.jvm.org.apache.hadoop
    path = hadoop.fs.Path(path)
    config = hadoop.conf.Configuration()

    status = hadoop.fs.FileSystem.get(config).listStatus(path)
    return (path_status.getPath().getName() for path_status in status)


def distribute_hdfs_files(hdfs_path):
    """Distributes recursively a given directory in HDFS to Spark

    Args:
        hdfs_path (str): path to directory
    """
    sc = SparkContext.getOrCreate()

    for path_name in list_path_names(hdfs_path):
        path = os.path.join(hdfs_path, path_name)
        _logger.info("Distributing {}...".format(path))
        sc.addFile(path, recursive=True)


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    # setup logging for driver
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    _logger = logging.getLogger(__name__)
    _logger.info("Starting up...")

    # Create the singleton instance
    spark = (SparkSession
             .builder
             .appName("My PySpark App in its own environment")
             .enableHiveSupport()
             .getOrCreate())

    # For simplicity we assume that the first argument is the environment on HDFS
    VENV_DIR = args[0]
    # make sure we have the latest version available on HDFS
    distribute_hdfs_files('hdfs://' + VENV_DIR)

    from my_pyspark_app import main
    main(args[1:])


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
```

It is actually easier than it looks. In the ``main`` function we initialize the ``SparkSession`` the first time so that later calls to the session builder will use this instance. Thereafter, the passed path argument when doing the ``spark-submit`` is extracted. Subsequently, this is passed to ``distribute_hdfs_files`` which calls ``sc.addFile`` recursively on every file to set up the isolated environment on the driver and executors. After this we are`able to import our ``my_pyspark_app`` package and call for instance its ``main`` method. The following graphic illustrates the whole concept: 

<figure>
<p align="center">
<img class="noZoom" src="/images/pyspark_venv.png" alt="Isolated environment with PySpark">
<figcaption><strong>Figure:</strong>Executing <em>spark-submit</em> uploads our <em>activate_env.py</em> module and starts a Spark driver process. Thereafter, <em>activate_env.py</em> bootstraps our <em>venv</em> environment on the Spark driver as well as on the executors. Finally, <em>activate_env.py</em> relinquishes control to <em>my_pyspark_app</em>.</figcaption>
</p>
</figure>

Setting up an isolated environment like this is a bit cumbersome and surely also somewhat hacky. Still, in our use-case it served us quite well and allowed the data scientists to set up their specific environments without admin rights. Since the explained method also works with [Jupyter][] this is not only useful for production but also for proof-of-concepts. That being said, we still hope that there will be soon an official solution by the Spark project itself.


[documentation]: https://www.cloudera.com/documentation/enterprise/5-6-x/topics/spark_python.html#spark_python__section_kr2_4zs_b5
[Juliet Hougland et al.]: http://blog.cloudera.com/blog/2016/02/making-python-on-apache-hadoop-easier-with-anaconda-and-cdh/
[Juliet Hougland]: http://blog.cloudera.com/blog/2015/09/how-to-prepare-your-apache-hadoop-cluster-for-pyspark-jobs/
[sc.addFile]: http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext.addFile
[sc.addPyFile]: http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext.addPyFile
[SPARK-13587]: https://issues.apache.org/jira/browse/SPARK-13587
[SPARK-16367]: https://issues.apache.org/jira/browse/SPARK-16367
[virtualenv]: https://virtualenv.pypa.io/en/stable/
[conda]: https://conda.io/docs/intro.html
[blog post by Eran Kampf]: https://developerzen.com/best-practices-writing-production-grade-pyspark-jobs-cb688ac4d20f
[coffee boat]: https://github.com/nteract/coffee_boat
[PyScaffold]: http://pyscaffold.org/
[Jupyter]: http://jupyter.org/
