# Register your models here.
from django.contrib import admin
from stat_application.models import FileNameModel
from stat_application.models import MethodModel
from stat_application.models import SummaryModel
from stat_application.models import ProgressModel

class FileNameAdmin(admin.ModelAdmin):
    list_display = ('id', 'file_name', 'upload_time','file_obj')
    list_display_links = ('id', 'file_name')
    
class MethodAdmin(admin.ModelAdmin):
    list_display = ('id', 'model_ja','model_en', 'upload_time')
    list_display_links = ('id', 'model_ja','model_en')
    
class ProgressAdmin(admin.ModelAdmin):
    list_display = ('id', 'progress')
    list_display_links = ('id', 'progress')
    
class SummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'model','aic', 'bic' ,'hqic' , 'p','d','q','sp','sd','sq','method','upload_time','rsq','rsq_adj','holdout','dw')
    list_display_links = ('id', 'model','aic', 'bic' ,'hqic' , 'p','d','q','sp','sd','sq','method','upload_time','rsq','rsq_adj','holdout','dw')

admin.site.register(FileNameModel, FileNameAdmin)
admin.site.register(MethodModel, MethodAdmin)
admin.site.register(ProgressModel, ProgressAdmin)
admin.site.register(SummaryModel, SummaryAdmin)