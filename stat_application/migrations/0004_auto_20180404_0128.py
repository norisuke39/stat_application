# Generated by Django 2.0.2 on 2018-04-03 16:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stat_application', '0003_auto_20180404_0112'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='filenamemodel',
            name='session',
        ),
        migrations.AddField(
            model_name='filenamemodel',
            name='session_id',
            field=models.CharField(default='DUMMY', max_length=50),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='methodmodel',
            name='session_id',
            field=models.CharField(max_length=50),
        ),
        migrations.AlterField(
            model_name='summarymodel',
            name='session_id',
            field=models.CharField(max_length=50),
        ),
    ]