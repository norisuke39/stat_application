# Create your views here.
import csv
import time
from multiprocessing import Process
import multiprocessing as mp
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.template.context_processors import csrf
from django.conf import settings
from stat_application.models import FileNameModel
from stat_application.models import MethodModel
from stat_application.models import ProgressModel
from stat_application.models import SummaryModel
import sys, os
import pandas as pd
import numpy as np
import random
from . import forms as forms
from . import calculate as cl
from django import forms as fm_d
UPLOADE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/static/files/'
#UPLOADE_DIR = 'http://resort-travel.jp/stat_application/result/'

###HOME画面のレンダリング
def index(request):

    return render(request, 'stat_application/index.html')
    
#プログレスバー用の関数
def progress(request):
    progress = ProgressModel.objects.latest('id')
    d = {
    'progress':progress.progress   
    }
    if request.is_ajax():
        return JsonResponse(d)
    else:
        return render(request,'stat_application/progress.html',d)

###状態空間モデルのinput画面
def state_space(request):
    #通常時state_space.htmlを表示
    if request.method != 'POST':
        return render(request, 'stat_application/state_space.html')    
    try:
        #ファイル取得し、データをcsv_dataに格納
        file = request.FILES['file']
        if file.name.split('.')[-1].lower() != 'csv':
            return render(request, 'stat_application/state_space.html')
        path = os.path.join(UPLOADE_DIR, file.name)
        destination = open(path, 'wb')
        #Fileをアップロード先に保存
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        #File名をサーバーに保存
        insert_data = FileNameModel(file_name = file.name,file_obj = file,session_id=request.session.session_key)
        insert_data.save()
        #UUIDを付与
        uuid = FileNameModel.objects.latest('upload_time')
        #request.session['uuid'] = str(uuid.id)
        insert_data = MethodModel(model_ja = '状態空間モデル',model_en = 'state_space',session_id=request.session.session_key)
        insert_data.save()
        return redirect('stat_application:choice_column')  
    except:
        return render(request, 'stat_application/state_space.html')
    
###SARIMAモデルのinput画面
def sarima(request):   
    #通常時state_space.htmlを表示
    if request.method != 'POST':
        return render(request, 'stat_application/sarima.html')
    try:
        #ファイル取得し、データをcsv_dataに格納
        file = request.FILES['file']
        if file.name.split('.')[-1].lower() != 'csv':
            return render(request, 'stat_application/sarima.html')
        path = os.path.join(UPLOADE_DIR, file.name)
        destination = open(path, 'wb')
        #Fileをアップロード先に保存
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        #File名をサーバーに保存
        insert_data = FileNameModel(file_name = file.name,file_obj = file,session_id=request.session.session_key)
        insert_data.save()
        #UUIDを付与
        uuid = FileNameModel.objects.latest('upload_time')
        #request.session['uuid'] = str(uuid.id)
        insert_data = MethodModel(model_ja = 'SARIMAモデル',model_en = 'sarima',session_id=request.session.session_key)
        insert_data.save()
        return redirect('stat_application:choice_column')   
    except:
        return render(request, 'stat_application/sarima.html')
    
###Prophetモデルのinput画面
def prophet(request):   
    #通常時state_space.htmlを表示
    if request.method != 'POST':
        return render(request, 'stat_application/prophet.html')
    try:
        #ファイル取得し、データをcsv_dataに格納
        file = request.FILES['file']
        if file.name.split('.')[-1].lower() != 'csv':
            return render(request, 'stat_application/prophet.html')
        path = os.path.join(UPLOADE_DIR, file.name)
        destination = open(path, 'wb')
        #Fileをアップロード先に保存
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        #File名をサーバーに保存
        insert_data = FileNameModel(file_name = file.name,file_obj = file,session_id=request.session.session_key)
        insert_data.save()
        #UUIDを付与
        uuid = FileNameModel.objects.latest('upload_time')
        #request.session['uuid'] = str(uuid.id)
        insert_data = MethodModel(model_ja = 'Prophetモデル',model_en = 'prophet',session_id=request.session.session_key)
        insert_data.save()
        return redirect('stat_application:choice_column')   
    except:
        return render(request, 'stat_application/prophet.html')
    
###RNNモデルのinput画面
def rnn(request):   
    #通常時state_space.htmlを表示
    if request.method != 'POST':
        return render(request, 'stat_application/rnn.html')
    try:
        #ファイル取得し、データをcsv_dataに格納
        file = request.FILES['file']
        if file.name.split('.')[-1].lower() != 'csv':
            return render(request, 'stat_application/rnn.html')
        path = os.path.join(UPLOADE_DIR, file.name)
        destination = open(path, 'wb')
        #Fileをアップロード先に保存
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        #File名をサーバーに保存
        insert_data = FileNameModel(file_name = file.name,file_obj = file,session_id=request.session.session_key)
        insert_data.save()
        #UUIDを付与
        uuid = FileNameModel.objects.latest('upload_time')
        #request.session['uuid'] = str(uuid.id)
        insert_data = MethodModel(model_ja = 'RNNモデル',model_en = 'rnn',session_id=request.session.session_key)
        insert_data.save()
        return redirect('stat_application:choice_column')   
    except:
        return render(request, 'stat_application/rnn.html')
    
    ###MultipleRegressionモデルのinput画面
def multiple_regression(request):   
    #通常時state_space.htmlを表示
    if request.method != 'POST':
        return render(request, 'stat_application/multiple_regression.html')
    try:
        #ファイル取得し、データをcsv_dataに格納
        file = request.FILES['file']
        if file.name.split('.')[-1].lower() != 'csv':
            return render(request, 'stat_application/rnn.html')
        path = os.path.join(UPLOADE_DIR, file.name)
        destination = open(path, 'wb')
        #Fileをアップロード先に保存
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        #File名をサーバーに保存
        insert_data = FileNameModel(file_name = file.name,file_obj = file,session_id=request.session.session_key)
        insert_data.save()
        #UUIDを付与
        uuid = FileNameModel.objects.latest('upload_time')
        #request.session['uuid'] = str(uuid.id)
        insert_data = MethodModel(model_ja = '重回帰モデル',model_en = 'mlr',session_id=request.session.session_key)
        insert_data.save()
        return redirect('stat_application:choice_column')   
    except:
        return render(request, 'stat_application/multiple_regression.html')
    
###アップしたファイルのカラム名からアロケしたい要素を選択
def choice_column(request):    
    #データベースに格納されたファイルオブジェクトを抽出→URLを抽出→ファイルデータを格納
    #temp = FileNameModel.objects.latest('id')
    temp = FileNameModel.objects.filter(session_id =request.session.session_key).latest('upload_time')
    csv_data = pd.read_csv(temp.file_obj.url, encoding = 'ms932')
    csv_data.isnull = '-'  
    #どのモデルでの計算かを取得
    #calc_model = MethodModel.objects.latest('id')
    calc_model = MethodModel.objects.filter(session_id =request.session.session_key).latest('upload_time')
    #選択されたファイルのカラム名をリスト化(アロケ粒度用)
    group1 = []
    for factor in csv_data.select_dtypes(include=['number']).columns:#本来はexclude=['number']
        group1.append((factor,factor))     
    #選択されたファイルのカラム名をリスト化(Day選択用)
    group2 = []
    for factor in csv_data.select_dtypes(exclude=['number']).columns:
        group2.append((factor,factor))          
    #選択されたファイルのカラム名をリスト化(入力項目選択用)
    group6 = []
    for factor in csv_data.select_dtypes(include=['number']).columns:
        group6.append((factor,factor))
    #forms.pyで定義されたフォームをファイルのカラム名で再定義
    form = forms.DfColumnForm()
    form.fields['df_data'].choices = group1
    form.fields['df_date'].choices = group2
    form.fields['df_predict'].choices = group6
    form.fields['df_goal'].choices = group1
    #選択されたカラム名/date項目を受け取り
    obj_choices = list(['hoge'])
    obj_predict = request.POST.getlist('df_predict') 
    obj_goal = request.POST.getlist('df_goal') 
    #進捗バー用に0%をDBに格納
    insert_data = ProgressModel(progress = 0)
    insert_data.save()
    #データテーブル表示用
    #重回帰には日付がなく、画面上だと自動的にソートされてしまうので、強制的にindexを付与
    #indexは最後尾に付与されるため、先頭に持ってくるために面倒な処理
    if calc_model.model_en == 'mlr':
        header_before = csv_data.columns#オリジナルのカラム名を保存
        header = ['index'] + list(header_before)#headerにindex行を追加
        csv_data['index'] = [i for i in range(len(csv_data))]#index行を付与(最後尾)
        csv_data = csv_data.ix[:,['index']+list(header_before)]#index行を先頭に持ってくる
        data_print = [csv_data.iloc[i] for i in range(10)]
    else:
        header = csv_data.columns
        data_print = [csv_data.iloc[i] for i in range(10)]
    
    ###回帰モデルの計算結果
    if obj_goal and calc_model.model_en == 'mlr':
        obj_holdout = request.POST['df_holdout']
        obj_choices = request.POST.getlist('df_data')
        obj_date = 'hoge'
        obj_method = request.POST['df_regression_method']
        #定数項が選択されれば、フラグを格納
        try:
            obj_constflag = request.POST['df_constflag']
        except:
            obj_constflag = 0
        obj_option = list([obj_holdout,obj_method,calc_model.model_ja,obj_constflag])
        queue = mp.Queue()
        p = Process(target = cl.calculate,args =(obj_choices,obj_date,obj_goal,calc_model.model_en,obj_option,request.session.session_key,queue))
        p.start()
        return redirect('stat_application:progress')
    ###時系列予測の計算結果
    elif obj_predict:
        obj_holdout = request.POST['df_holdout']
        obj_predictspan = request.POST['df_predictspan']
        obj_date = request.POST.getlist('df_date')
        #状態空間モデル
        if calc_model.model_en == 'state_space':
            obj_seasonal = request.POST['df_seasonal']
            obj_method = request.POST['df_state_space_method']
            obj_option = list([obj_holdout,obj_predictspan,obj_seasonal,obj_method,calc_model.model_ja])
        #SARIMAモデル
        if calc_model.model_en == 'sarima':
            obj_seasonal = request.POST['df_seasonal']
            obj_method = 'SARIMA'
            obj_parameta = pd.DataFrame(columns = [])
            #自動最適化フラグが選択されれば、フラグを格納
            try:
                obj_autoflag = request.POST['df_autoflag']
                obj_parameta = [request.POST['df_p'],request.POST['df_d'],request.POST['df_q'],request.POST['df_sp'],request.POST['df_sd'],request.POST['df_sq']]
            except:
                obj_autoflag = False
                obj_parameta = 0
            obj_option = list([obj_holdout,obj_predictspan,obj_seasonal,obj_method,calc_model.model_ja,obj_autoflag,obj_parameta])   
        #prophetモデル
        if calc_model.model_en == 'prophet':
            obj_seasonal = request.POST['df_seasonal']
            obj_method = request.POST['df_prophet_method']
            obj_date_unit = request.POST['df_date_unit']
            #キャップフラグが選択されれば、フラグを格納
            try:
                obj_capflag = request.POST['df_flag']
                obj_cap = request.POST['df_cap']
            except:
                obj_capflag = False
                obj_cap = False
            obj_option = list([obj_holdout,obj_predictspan,obj_seasonal,obj_method,calc_model.model_ja,obj_date_unit,obj_capflag,obj_cap])
        #RNNモデル
        if calc_model.model_en == 'rnn':
            obj_seasonal = request.POST['df_seasonal']
            obj_method = 'RNN'
            obj_option = list([obj_holdout,obj_predictspan,obj_seasonal,obj_method,calc_model.model_ja])
        #Ajaxでプログレスバーを非同期的に更新するため、マルチプロセッシング計算
        queue = mp.Queue()
        p = Process(target = cl.calculate,args =(obj_choices,obj_date,obj_predict,calc_model.model_en,obj_option,request.session.session_key,queue))
        #result,result_file_name = cl.calculate(obj_choices,obj_date,obj_predict,calc_model.model_en,obj_option,request.session['uuid'],request.session.session_key)
        p.start()
        return redirect('stat_application:progress')
    else:
        data = {
            'input_data' : form,
            'header' : header,
            'data_heads' : np.array(data_print[:10]),
            'model' :  calc_model,
        }
        return render(request,'stat_application/choice_column.html',data)

#出力結果レンダリング関数
def result(request):  
    #AIC、BIC、HQIC等のモデルの評価指標を格納
    summary = SummaryModel.objects.filter(session_id =request.session.session_key).latest('upload_time')
    #予測値のみ
    result_file_name = './result/forecast_result_preview.csv'
    _data_predict = pd.read_csv(result_file_name, encoding='ms932')
    #オリジナル
    result_file_name = './result/original_data.csv'
    _data_ori = pd.read_csv(result_file_name, encoding='ms932')
    
    #グラフ用予測部分だけ色を変えるために綺麗にする処理
    if summary.model != '重回帰モデル':
        _data_predict = _data_predict.append(_data_ori).sort_values(['date','original'],ascending = [1,0])
        _data_predict['predict'] = (_data_predict.fillna(0)['original']+_data_predict.fillna(0)['predict']).round(2)
        _data_temp = _data_predict
        _data_predict = _data_predict.drop_duplicates('date',keep = 'last').reset_index()
        _data_temp = _data_temp.drop_duplicates('date',keep = 'first').reset_index()
        _data_predict['original'] = _data_temp['original']
        _data_predict = _data_predict.drop('index',axis = 1).fillna(0)
        date = sorted(list(set(list(_data_ori['date']) + list(_data_predict['date']))))
        #Data Preview用
        header = _data_predict.columns
        _data_print =np.array(_data_predict.iloc[0:])
        _data_ori = _data_ori.round(2)
        _data_predict = _data_predict.round(2)
        _data_predict_ho = _data_predict.round(2)#使用しないけどエラー防止のため入れている
    else:
        #Data Preview用
        header = _data_predict.columns
        _data_print =np.array(_data_predict.iloc[0:]).round(2)
        _data_ori = _data_ori.round(2)
        _data_predict_ho = _data_predict.round(2)
        _data_predict = _data_predict[:-(summary.holdout)].round(2)
        date = list(_data_ori['index'].round(0))        

    data = {
      'header' : header,
      'data_preview' : _data_print,
      'date' : date,
      'original' : list(_data_ori['original']),
      'predict' : list(_data_predict['predict']),
      'holdout' : list(_data_predict_ho['predict']),
      'summary' : summary,
    }  
    return render(request,'stat_application/result.html',data)
    
#ログイン画面レンダリング関数
def login(request):
    
    return render(request,'stat_application/login.html')