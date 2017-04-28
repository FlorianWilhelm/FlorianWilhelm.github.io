---
title: Separation of Concerns and the Decorator Pattern
date: 2017-04-15 18:00
modified: 2017-04-24 18:00
category: post
tags: programming, python, decorator
authors: Florian Wilhelm
status: draft
---

Data scientists are the proclaimed unicorns of the 21st century  because of the many skills an ideal, and thus rare, representative of its genus should have. A data scientist should excel in mathematics and statistics, tell the most beautiful stories about data, is at the same time a profound domain expert and ... well, ... is able to program of course. Of those four skills the skill of programming is quite often seen as granted. If you have constructed some neural network in TensorFlow, ran a word count (what else?) in Spark and did some analysis in Jupyter or RStudio, you have successfully demonstrated that you are able to program, right? Well, no, skilled programming is so much more and highly under-appreciated in data science. Hacked together scripts, several screen long functions which can never be practically unit-tested seem to be just acceptable if the accuracy of the model is high enough... until the day your data product goes live and you or some other poor soul needs to fix and maintain the code. 
     

  


Tracing, Logging, Transaktionalität, Caching
Separation of Concerns führt zu loser Kopplung und hoher Kohäsion
Python besser als R weil produktiver etc. mehr Fokus auf guten Code.
aspect-oriented programming (AOP) https://de.wikipedia.org/wiki/Aspektorientierte_Programmierung

```python
def pretty_timedelta(seconds):
    """Converts timedelta in seconds to human-readable string

    Caution: Taken from https://gist.github.com/thatalextaylor/7408395

    Args:
        seconds (int): time delta in seconds

    Returns:
        str: timedelta as pretty string
    """
    sign = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '{}{}d{}h{}m{}s'.format(sign, days, hours, minutes, seconds)
    elif hours > 0:
        return '{}{}h{}m{}s'.format(sign, hours, minutes, seconds)
    elif minutes > 0:
        return '{}{}m{}s'.format(sign, minutes, seconds)
    else:
        return '{}{}s'.format(sign, seconds)


def log_time(msg=None, level=logging.DEBUG):
    """Decorator for logging the run time of a function

    Args:
        msg (str): alternative log message containing {time}
        level (int): log level, e.g. logging.INFO, logging.WARN etc.

    Returns:
        wrapped function
    """
    def wraps(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            resp = func(*args, **kwargs)
            secs = time.time() - start_time
            logger = logging.getLogger(func.__module__)
            if msg is None:
                message = "Runtime of {}: {{time}}".format(func.__name__)
            else:
                message = msg
            logger.log(level, message.format(time=pretty_timedelta(secs)))
            return resp
        return wrapper
    return wraps 
```


```python
def throttle(calls, seconds=1):
    """Decorator for throttling a function to number of calls per seconds

    Args:
        calls (int): number of calls per interval
        seconds (int): number of seconds in interval

    Returns:
        wrapped function
    """
    assert isinstance(calls, int), 'number of calls must be integer'
    assert isinstance(seconds, int), 'number of seconds must be integer'

    def wraps(func):
        # keeps track of the last calls
        last_calls = list()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            curr_time = time.time()
            if last_calls:
                # remove calls from last_calls list older then interval in seconds
                idx_old_calls = [i for i, t in enumerate(last_calls) if t < curr_time - seconds]
                if idx_old_calls:
                    del last_calls[:idx_old_calls[-1]]
            if len(last_calls) >= calls:
                idx = len(last_calls) - calls
                delta = fabs(1 - curr_time + last_calls[idx])
                logger = logging.getLogger(func.__module__)
                logger.debug("Stalling call to {} for {}s".format(func.__name__, delta))
                time.sleep(delta)
            resp = func(*args, **kwargs)
            last_calls.append(time.time())
            return resp

        return wrapper

    return wraps
```