#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import os
import inspect

__location__ = os.path.join(os.getcwd(), os.path.dirname(
    inspect.getfile(inspect.currentframe())))

AUTHOR = u'Florian Wilhelm'
SITENAME = 'Florian Wilhelm'
SITEURL = 'https://www.florianwilhelm.info'
# SITEURL = ''

PATH = 'content'
MARKUP = ('md',)
TIMEZONE = 'Europe/Paris'
DEFAULT_LANG = u'en'
OUTPUT_PATH = os.path.join(__location__, '../')
TYPOGRIFY = True

# Do not publish articles set in the future
WITH_FUTURE_DATES = False

THEME = 'themes/pelican-bootstrap3'
SHOW_ARTICLE_AUTHOR = False
JINJA_EXTENSIONS = ['jinja2.ext.i18n']

STATIC_PATHS = ['images', 'documents', 'extras', 'notebooks', 'src']

DEFAULT_METADATA = {
    'author': 'Florian Wilhelm',
    'description': 'Description'
}

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
#LINKS = (('PyScaffold', 'http://pyscaffold.readthedocs.org/'),
#         ('Data Science Central', 'http://www.datasciencecentral.com/'),
#         ('KDNuggets', 'http://www.kdnuggets.com/'),
#         ('Analytics Vidhya', 'http://www.analyticsvidhya.com/'))

# Social widget
SOCIAL = (('Twitter', 'https://twitter.com/FlorianWilhelm'),
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
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_TAGS_ON_SIDEBAR = True
DISPLAY_TAGS_INLINE = True
DISPLAY_CATEGORIES_ON_SIDEBAR = False
TWITTER_CARDS = True
USE_OPEN_GRAPH = True
TWITTER_USERNAME = 'FlorianWilhelm'
NOTEBOOK_DIR = 'notebooks'
#GITHUB_USER = 'FlorianWilhelm'
#GITHUB_SKIP_FORK = True
#BANNER = ''
#BANNER_SUBTITLE = 'This is my subtitle'

# Google Analytics
GOOGLE_ANALYTICS = 'UA-71694209-1'
GOOGLE_ANALYTICS_PROPERTY = 'auto'

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
TAGS_SAVE_AS = 'tags.html'
CATEGORIES_SAVE_AS = 'categories.html'
ARCHIVES_SAVE_AS = 'archives.html'

# TEMPLATE_PAGES = {
#     'templates/videos.html': 'video/index.html',
#     'templates/code.html': 'code/index.html',
#     'templates/themes.html': 'themes/index.html',
# }

DIRECT_TEMPLATES = ['index', 'categories', 'authors', 'archives', 'search', 'tags']


#LEFT_SIDEBAR = """"""

# plumage settings
SITESUBTITLE = 'Data Scientist'
SITE_THUMBNAIL = '/images/thumbnail.jpeg'
SITE_THUMBNAIL_TEXT = 'Sometimes with hip glasses 😎'
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
PLUGIN_PATHS = ['plugins-core', 'plugins']
PLUGINS = [
     # Core plugins (order actually matters)
     'render_math',
     'i18n_subsites',
     'tag_cloud',
     'related_posts',
     'thumbnailer',
#   'tipue_search',
     'neighbors',
     'sitemap',
     'liquid_tags.img',
     'liquid_tags.video',
     'liquid_tags.youtube',
     'liquid_tags.vimeo',
     'liquid_tags.include_code',
     'liquid_tags.notebook',  # be careful, this inserts mathjax due to the theme
]

# check https://github.com/barrysteyn/pelican_plugin-render_math for options
MATH_JAX = {'align': 'center'}
BOOTSTRAP_NAVBAR_INVERSE = True

DISQUS_SITENAME = 'florianwilhelmblog'
TIPUE_SEARCH = True
