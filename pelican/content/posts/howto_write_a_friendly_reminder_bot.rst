====================================
How to write a friendly reminder bot
====================================

:date: 2015-07-24 12:30
:modified: 2015-12-22 19:30
:category: Talks
:tags: python, google hangouts, asyncio, event-driven, asynchronous
:slug: howto-write-a-friendly-reminder-bot
:authors: Florian Wilhelm

In this presentation given at the `EuroPython 2015 <https://ep2015.europython.eu/>`_ in Bilbao, I show how the `hangups <https://github.com/tdryer/hangups>`_
library can be used in order to write a small chatbot that connects to Google Hangouts
and reminds you or someone else to take his/her medication.
The secure and recommended OAuth2 protocol is used to authorize the bot application
in the Google Developers Console in order to access the Google+ Hangouts API.
Subsequently, I explain how to use an event-driven library to write a bot
that sends scheduled messages, waits for a proper reply and repeats the question if need be.
Thereby, a primer on event-driven, asynchronous architectures is given.

The source code can be downloaded on `GitHub <https://github.com/blue-yonder/medbot>`_
and the slides are available as `html preview <http://htmlpreview.github.io/?https://github.com/blue-yonder/medbot/blob/master/medbot.slides.html?theme=solarized#/>`_.

.. youtube:: ztfdv9jcxtw
  :width: 800
  :height: 500
