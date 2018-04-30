"""file_uploader URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from stat_application import views

urlpatterns = [
    url(r'^$', views.index, name = 'index'),
    url(r'^index.html$', views.index, name = 'index'),
    url(r'^progress/', views.progress, name = 'progress'),
    url(r'^login.html', views.login, name = 'login'),
    url(r'^state_space.html', views.state_space, name = 'state_space'),
    url(r'^sarima.html', views.sarima, name = 'sarima'),
    url(r'^prophet.html', views.prophet, name = 'prophet'),
    url(r'^rnn.html', views.rnn, name = 'rnn'),
    url(r'^multiple_regression.html', views.multiple_regression, name = 'multiple_regression'),
    url(r'^choice_column/', views.choice_column, name = 'choice_column'),
    url(r'^result/', views.result, name = 'result'),
    url(r'^contact.html', views.contact, name = 'contact'),
]