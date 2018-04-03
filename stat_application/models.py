from django.db import models
import uuid
from django.contrib.postgres.functions import RandomUUID

# Create your models here.
from datetime import datetime

class FileNameModel(models.Model):
    #id = models.UUIDField(primary_key=True, default=uuid.uuid4(), editable=False)
    file_name = models.CharField(max_length = 50)
    upload_time = models.DateTimeField(default = datetime.now)
    file_obj = models.FileField(upload_to = 'statistical_processing/static/files/')
    session_id = models.CharField(max_length = 100)
    
class MethodModel(models.Model):
    id = models.UUIDField(primary_key=True)
    model_ja = models.CharField(max_length = 50)
    model_en = models.CharField(max_length = 50)
    upload_time = models.DateTimeField(default = datetime.now)
    session_id = models.CharField(max_length = 100)
    
class ProgressModel(models.Model):
    progress = models.IntegerField(default = 0)
    
class SummaryModel(models.Model):
    id = models.UUIDField(primary_key=True)
    model = models.CharField(max_length = 50)
    upload_time = models.DateTimeField(default = datetime.now)
    aic = models.FloatField(default = 0)
    bic = models.FloatField(default = 0)
    hqic = models.FloatField(default = 0)
    p = models.IntegerField(default = 0)
    d = models.IntegerField(default = 0)
    q = models.IntegerField(default = 0)
    sp = models.IntegerField(default = 0)
    sd = models.IntegerField(default = 0)
    sq = models.IntegerField(default = 0)
    rsq = models.FloatField(default = 0)
    rsq_adj = models.FloatField(default = 0)
    dw = models.FloatField(default = 0)
    holdout = models.IntegerField(default = 0)
    method = models.CharField(max_length = 50)
    session_id = models.CharField(max_length = 100)