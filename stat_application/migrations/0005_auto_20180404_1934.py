# Generated by Django 2.0.2 on 2018-04-04 10:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stat_application', '0004_auto_20180404_0128'),
    ]

    operations = [
        migrations.AlterField(
            model_name='filenamemodel',
            name='file_obj',
            field=models.FileField(upload_to='stat_application/temp/files/'),
        ),
    ]
