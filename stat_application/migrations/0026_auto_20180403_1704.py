# Generated by Django 2.0.2 on 2018-04-03 08:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stat_application', '0025_auto_20180403_1657'),
    ]

    operations = [
        migrations.AlterField(
            model_name='methodmodel',
            name='id',
            field=models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.AlterField(
            model_name='summarymodel',
            name='id',
            field=models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]
