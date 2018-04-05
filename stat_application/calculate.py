import os
import numpy as np
import pandas as pd
import scipy.stats 
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg') # 追加
import matplotlib.pyplot as plt
from operator import itemgetter
import datetime
import itertools
from multiprocessing import Process
import multiprocessing as mp
from stat_application.models import FileNameModel
from stat_application.models import MethodModel
from stat_application.models import ProgressModel
from . import predict_method as pr

UPLOADE_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate(_col,date,predict,method,obj_option,session_id):
    
    #進捗バー用に0%をDBに格納
    insert_data = ProgressModel(progress = 0)
    insert_data.save()
    ##ファイル読み込み
    file = FileNameModel.objects.latest('id')
    #file = FileNameModel.objects.filter(session_id =session_id).latest('upload_time')
    _data = pd.read_csv(file.file_obj.url, encoding = 'ms932')
    ##欠損値除去
    _data = _data.replace(np.NaN,0)
    ##変数選択
    predict = predict[0]  
    #説明変数が0の時は除外
    _data = _data[_data[predict] != 0]
    #ダミーで格納(複数項目が入ることを見据えて※重要)
    _data['hoge'] = 'hoge'
    #回帰分析用に面倒な処理(時系列用の流れを壊さぬよう、一旦説明変数を退避。)
    evar = _col
    _col = list(['hoge'])

    ##月単位への変換
    if method != 'mlr':
        _data['date'] = pd.to_datetime(_data[date[0]],format = '%Y-%m-%d')
        _data['month'] = _data['date'].map(lambda x : x.month)
        _data['year'] = _data['date'].map(lambda x : x.year)
        _data['Y-M'] = pd.to_datetime(_data['year'].astype(str)+'-'+_data['month'].astype(str) + '-'+'1',format = '%Y-%m-%d')
    
        #オブジェクト型と数値型のカラムを別々に格納
        _col_obj = _data.select_dtypes(include=['object']).columns
        _col_num = _data.select_dtypes(include=['number']).fillna(0).columns
        #_data_dat = date
        _col_dat = ['Y-M','date']

        #UIで選択されたカラムの要素をユニークな配列に変換
        _element_list = np.array(_data[_col])
        _unique_list = np.vstack({tuple(row) for row in _element_list})

        #UIで選択されたカラムと日付でサマル
        data_wk = _data.groupby(_col + list(_col_dat),as_index = False).sum().fillna(0)    
    else:
        data_wk = _data
        
    #insert_data = ProgressModel(progress = 10)
    #insert_data.save()
    #RNNモデル
    if method == 'rnn':
        data_result,data_preview,data_ori = pr.forecast_rnn(data_wk,_unique_list,_col,predict,obj_option,session_id)
    #prophet予測計算
    if method == 'prophet':
        data_result,data_preview,data_ori = pr.forecast_prophet(data_wk,_unique_list,_col,predict,obj_option,session_id)
    #状態空間モデル
    if method == 'state_space':
        data_result,data_preview,data_ori = pr.forecast_state_space(data_wk,_unique_list,_col,predict,obj_option,session_id)
    #SARIMAモデル
    if method == 'sarima':
        data_result,data_preview,data_ori = pr.forecast_SARIMA(data_wk,_unique_list,_col,predict,obj_option,session_id)
    #重回帰分析モデル
    if method == 'mlr':
        _col = evar
        data_result,data_preview,data_ori = pr.forecast_mlr(data_wk,_col,predict,obj_option,session_id)

    ###ファイル保存
    #サーバー
    result_file_name = 'http://resort-travel.jp/stat_application/result/forecast_result.csv'
    data_result.to_excel('http://resort-travel.jp/stat_application/result/forecast_result.xlsx',index = False)
    data_result.to_csv(result_file_name,index = False)
    #ローカル
    result_file_name = UPLOADE_DIR+'/temp/result/forecast_result.csv'
    data_result.to_excel(UPLOADE_DIR+'/temp/result/forecast_result.xlsx',index = False)
    data_result.to_csv(result_file_name,index = False)
    ##preview用データ保存
    #サーバー
    #result_file_name = 'http://resort-travel.jp/stat_application/result/forecast_result_preview.csv'
    #data_preview.to_excel('http://resort-travel.jp/stat_application/result/forecast_result_preview.xlsx',index = False)
    #data_preview.to_csv(result_file_name,index = False)
    #ローカル
    result_file_name = UPLOADE_DIR+'/temp/result/forecast_result_preview.csv'
    data_preview.to_excel(UPLOADE_DIR+'/temp/result/forecast_result_preview.xlsx',index = False)
    data_preview.to_csv(result_file_name,index = False)
    ##オリジナル+予測データ保存
    #サーバー
    #result_file_name = 'http://resort-travel.jp/stat_application/result/original_data.csv'
    #data_preview.to_excel('./result/forecast_result_preview.xlsx',index = False)
    #data_ori.to_csv(result_file_name,index = False)
    #ローカル
    result_file_name = UPLOADE_DIR+'/temp/result/original_data.csv'
    #data_preview.to_excel('./result/forecast_result_preview.xlsx',index = False)
    data_ori.to_csv(result_file_name,index = False)
    
    #q.put([data_result,result_file_name])
    
    insert_data = ProgressModel(progress = 100)
    insert_data.save()
    return data_result,result_file_name