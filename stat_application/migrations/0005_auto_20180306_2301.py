# -*- coding: utf-8 -*-
# Generated by Django 1.11.8 on 2018-03-06 14:01
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stat_application', '0004_auto_20180304_0336'),
    ]

    operations = [
        migrations.CreateModel(
            name='SummaryModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('aic', models.FloatField(default=0)),
                ('bic', models.FloatField(default=0)),
                ('hqic', models.FloatField(default=0)),
                ('method', models.CharField(max_length=50)),
            ],
        ),
        migrations.DeleteModel(
            name='BudgetModel',
        ),
    ]
