# -*- coding: utf-8 -*-
# Generated by Django 1.11.8 on 2018-03-03 18:36
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stat_application', '0003_auto_20180304_0300'),
    ]

    operations = [
        migrations.AlterField(
            model_name='resultfilemodel',
            name='file_obj',
            field=models.FileField(upload_to='statistical_processing/static/result/'),
        ),
    ]
