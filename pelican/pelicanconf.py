#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import os
import inspect

__location__ = os.path.join(os.getcwd(), os.path.dirname(
    inspect.getfile(inspect.currentframe())))

AUTHOR = u'Florian Wilhelm'
SITENAME = 'Florian Wilhelm'
#SITEURL = 'http://florianwilhelm.info      '


PATH = 'content'
TIMEZONE = 'Europe/Paris'
DEFAULT_LANG = u'en'
OUTPUT_PATH = os.path.join(__location__, '../')
TYPOGRIFY = True

# Do not publish articles set in the future
WITH_FUTURE_DATES = False

THEME = 'themes/plumage'

STATIC_PATHS = ['images', 'documents', 'extras']

EXTRA_PATH_METADATA = {
    'extras/favicon.ico': {'path': 'favicon.ico'},
    'extras/robots.txt': {'path': 'robots.txt'},
}

# Feed generation
FEED_RSS = 'feed/index.html'
FEED_ATOM = 'feed/atom/index.html'
FEED_ALL_RSS = None
FEED_ALL_ATOM = None
TRANSLATION_FEED_RSS = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('PyScaffold', 'http://pyscaffold.readthedocs.org/'),
         ('Data Science Central', 'http://www.datasciencecentral.com/'),
         ('KDNuggetes', 'http://www.kdnuggets.com/'),
         ('Analytics Vidhya', 'http://www.analyticsvidhya.com/'))

# Social widget
SOCIAL = (('@FlorianWilhelm', 'https://twitter.com/FlorianWilhelm'),
          ('LinkedIn', 'https://linkedin.com/in/florian-wilhelm-621ba834'),
          ('GitHub', 'https://github.com/FlorianWilhelm'))

# Pagination
DEFAULT_ORPHANS = 2
DEFAULT_PAGINATION = 5

FEED_MAX_ITEMS = 5
USE_FOLDER_AS_CATEGORY = False
DEFAULT_CATEGORY = 'English'
DEFAULT_DATE_FORMAT = '%b. %d, %Y'
REVERSE_ARCHIVE_ORDER = True
DISPLAY_PAGES_ON_MENU = False

# Google Analytics
GOOGLE_ANALYTICS_UNIVERSAL = 'UA-7169420-1'
GOOGLE_ANALYTICS_UNIVERSAL_PROPERTY = 'auto'

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# Force Pelican to use the file name as the slug, instead of derivating it from
# the title.
FILENAME_METADATA = '(?P<slug>.*)'

# Force the same URL structure as WordPress
ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
ARTICLE_SAVE_AS = ARTICLE_URL + 'index.html'
ARTICLE_PATHS = ['posts']

PAGE_URL = '{slug}/'
PAGE_SAVE_AS = PAGE_URL + 'index.html'
PAGE_PATHS = ['pages']

TAG_URL = 'tag/{slug}/'
TAG_SAVE_AS = TAG_URL + 'index.html'

CATEGORY_URL = 'category/{slug}/'
CATEGORY_SAVE_AS = CATEGORY_URL + 'index.html'

YEAR_ARCHIVE_SAVE_AS = '{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = '{date:%Y}/{date:%m}/index.html'

# Deactivate author URLs
AUTHOR_SAVE_AS = False
AUTHORS_SAVE_AS = False

# Deactivate localization
ARTICLE_LANG_SAVE_AS = None
PAGE_LANG_SAVE_AS = None

# Tags, categories and archives are Direct Templates, so they don't have a
# <NAME>_URL option.
TAGS_SAVE_AS = 'tags/index.html'
CATEGORIES_SAVE_AS = 'categories/index.html'
ARCHIVES_SAVE_AS = 'archives/index.html'

# TEMPLATE_PAGES = {
#     'templates/videos.html': 'video/index.html',
#     'templates/code.html': 'code/index.html',
#     'templates/themes.html': 'themes/index.html',
# }

DIRECT_TEMPLATES = ['index', 'categories', 'authors', 'archives', 'search', 'tags']

# plumage settings
SITESUBTITLE = 'Data Scientist'
SITE_THUMBNAIL = '/images/myself.jpeg'
MENUITEMS = (
    ('Home', '/'),
    ('About me', '/about/'),
)

SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.5,
        'indexes': 0.5,
        'pages': 0.5
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly'
    }
}

# Plugins
PLUGIN_PATHS = ['plugins-core']
PLUGINS = [
#     # Core plugins
#     'related_posts',
#     # 'thumbnailer',
#     'tipue_search',
#     'neighbors',
     'sitemap',
      'pelican_youtube'
]
