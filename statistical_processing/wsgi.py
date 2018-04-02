"""
WSGI config for statistical_processing project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""
#ローカルサーバ
import os

from django.core.wsgi import get_wsgi_application
from socket import gethostname

hostname = gethostname()

if 'local' in hostname:

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "statistical_processing.settings")

    application = get_wsgi_application()

else:
    from dj_static import Cling

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stat_application.settings")

    application = Cling(get_wsgi_application())