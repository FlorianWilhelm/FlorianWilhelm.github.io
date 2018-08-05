---
title: Handling GPS Data with Python
date: 2016-07-22 11:15
modified: 2016-07-22 11:15
category: talk
tags: python, jupyter, kalman filter, gps
authors: Florian Wilhelm
status: published
summary: This talk presented at the [EuroPython 2016](https://ep2016.europython.eu/conference/talks/handling-gps-data-with-python) introduces several Python libraries related to the handling of GPS data. The slides of this talk are available on [Github](https://github.com/FlorianWilhelm/gps_data_with_python) or on [nbviewer](http://nbviewer.jupyter.org/format/slides/github/FlorianWilhelm/gps_data_with_python/blob/master/talk.ipynb#/).
---

This talk presented at the [EuroPython 2016][] introduces several Python libraries
related to the handling of GPS data. The slides of this talk are available on
[Github][] or on [nbviewer][].

If you have ever happened to need to deal with GPS data in Python you may have
felt a bit lost. There are many libraries at various states of maturity and scope.
Finding a place to start and to actually work with the GPS data might not be as
easy and obvious as you might expect from other Python domains.

Inspired from my own experiences of dealing with GPS data in Python, I want to
give an overview of some useful libraries. From basic reading and writing GPS
tracks in the GPS Exchange Format with the help of [gpxpy][] to adding missing
elevation information with [srtm.py][]. Additionally, I will cover mapping and
visualising tracks on [OpenStreetmap] with [mplleaflet] that even supports
interactive plots in a Jupyter notebook.

Besides the tooling, I will also demonstrate and explain common algorithms like
Douglas-Peucker to simplify a track and the famous Kalman filters for smoothing.
For both algorithms I will give an intuition about how they work as well as their
basic mathematical concepts. Especially the Kalman filter that is used for all
kinds of sensor, not only GPS, has the reputation of being hard to understand.
Still, its concept is really easy and quite comprehensible as I will also
demonstrate by presenting an implementation in Python with the help of Numpy and
Scipy. My presentation will make heavy use of the Jupyter notebook which is a
wonderful tool perfectly suited for experimenting and learning.

{% youtube 9Q8nEA_0ccg 800 500 %}

[EuroPython 2016]: https://ep2016.europython.eu/conference/talks/handling-gps-data-with-python
[Github]: https://github.com/FlorianWilhelm/gps_data_with_python
[nbviewer]: http://nbviewer.jupyter.org/format/slides/github/FlorianWilhelm/gps_data_with_python/blob/master/talk.ipynb#/
[gpxpy]: https://github.com/tkrajina/gpxpy
[srtm.py]: https://github.com/tkrajina/srtm.py
[mplleaflet]: https://github.com/jwass/mplleaflet
[OpenStreetmap]: https://www.openstreetmap.org/
