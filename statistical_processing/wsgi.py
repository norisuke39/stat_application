"""
WSGI config for statistical_processing project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from socket import gethostname
from dj_static import Cling
from whitenoise.django import DjangoWhiteNoise

hostname = gethostname()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "statistical_processing.settings")
if 'local' in hostname:

    application = get_wsgi_application()

else:
    #application = get_wsgi_application()
    #application = DjangoWhiteNoise(application)
    application = Cling(get_wsgi_application())