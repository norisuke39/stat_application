# Generated by Django 2.0.2 on 2018-04-04 11:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('stat_application', '0005_auto_20180404_1934'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='filenamemodel',
            name='session_id',
        ),
        migrations.RemoveField(
            model_name='methodmodel',
            name='session_id',
        ),
        migrations.RemoveField(
            model_name='summarymodel',
            name='session_id',
        ),
    ]