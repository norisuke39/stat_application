import os
import numpy as np
import pandas as pd
#import itertools
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg') # 追加
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
from numba.decorators import jit
from stat_application.models import SummaryModel
from stat_application.models import ProgressModel

###時系列予測
from fbprophet import Prophet #FacebookのProphetモデル
import statsmodels.api as sm #状態空間モデル、#SARIMAモデル、#ARモデル、#VARモデル、#ARIMAモデル
#from sklearn import linear_model #重回帰モデル
import statsmodels.formula.api as smf #重回帰モデル2
from sklearn.preprocessing import PolynomialFeatures #重回帰モデル3
#from scipy.optimize import curve_fit#線形回帰モデル、#非線形回帰モデル、#対数回帰モデル
#RNN
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
###分類予測
#ロジスティック回帰モデル
#決定木モデル
#ランダムフォレスト
#状態空間モデルでの予測
UPLOADE_DIR = os.path.dirname(os.path.abspath(__file__))
#Prophetでの予測
def forecast_prophet(_data,_unique_list,_col,predict,option,session_id):
    ###変数まとめ
    estimate_method = option[3]
    growth = option[3] #線形or非線形
    seasonality = int(option[2]) #シーズナル間隔
    capflag = option[6]#capフラグ
    cap = float(option[7])#cap値
    
    #データフレーム初期化
    _data_pr = pd.DataFrame(columns = [])
    data_rsp = pd.DataFrame(columns = [])
    data_stat = pd.DataFrame(columns = [])
    data_all = pd.DataFrame(columns = [])
    data_stat_preview = pd.DataFrame(columns = [])
    data_ori = pd.DataFrame(columns = [])
    
    for row in _unique_list:
        model = Prophet(growth = 'linear',yearly_seasonality = seasonality)
        model = Prophet()

        ###選択された粒度ごとで計算
        mask = np.array(_data[_col]) == row
        mask = np.prod(mask, axis=1).astype(np.bool)
        _data_mask = _data[mask].dropna()
        #_data_sum = _data_mask.groupby('Y-M',as_index = False).sum()
        _data_sum = _data_mask.groupby('date',as_index = False).sum()
        holdout = len(_data_sum) - int(option[0]) #ホールドアウト期間
        predict_span = int(option[1]) #予測期間
        freq = option[5] # 日付の単位
        
        _date = _data_sum['date'].iloc[-1]
        _date_ori = pd.DataFrame(_data_sum['date'])
        if freq == 'M':
            temp = [_date + relativedelta(months = t) for t in range(predict_span)]
            _date = pd.DataFrame(pd.to_datetime(temp,format = '%Y-%m-%d'),columns = ['date'])
        if freq == 'W':
            temp = [_date + datetime.timedelta(weeks = t) for t in range(predict_span)]
            _date = pd.DataFrame(pd.to_datetime(temp,format = '%Y-%m-%d'),columns = ['date'])
        if freq == 'D':
            temp = [_date + datetime.timedelta(days = t) for t in range(predict_span)]
            _date = pd.DataFrame(pd.to_datetime(temp,format = '%Y-%m-%d'),columns = ['date'])
        if freq == 'H':
            temp = [_date + datetime.timedelta(hours = t) for t in range(predict_span)]
            _date = pd.DataFrame(pd.to_datetime(temp,format = '%Y-%m-%d %H'),columns = ['date'])
        _date = _date.append(_date_ori).sort_values('date',ascending = True).reset_index()
        #cap =_data_sum[predict].median() * 3

        #_data_pr['ds'] = _data_sum['Y-M'].dropna()
        
        _data_pr['ds'] = _data_sum['date'].dropna()
        _data_pr['y'] = _data_sum[predict].dropna()
        _data_pr = _data_pr.dropna(axis = 0,how = 'any')
        if capflag:
            _data_pr['cap'] = cap
        model.fit(_data_pr)
        future_df = model.make_future_dataframe(predict_span,freq = freq)
        if capflag:
            future_df['cap'] = cap
        forecast_df = model.predict(future_df)
        model.plot(forecast_df)
        plt.savefig(UPLOADE_DIR + '/temp/prophet/pic/'+'_'.join(row) +'.jpg')
        
        #保存用に形式揃え
        pred_df = pd.DataFrame(forecast_df)
        
        #insert_data = SummaryModel(model = option[4],method = estimate_method,session_id=session_id)
        insert_data = SummaryModel(model = option[4],method = estimate_method)
        insert_data.save()
        
        for (i,j) in zip(_col,row):
            data_rsp[i] = pd.Series(j)
        for (d,v) in zip(pred_df['ds'],pred_df['yhat'].astype(int)):
            data_rsp[d] = pd.Series(v)      
        #data_rsp['pre_Imp.'] = forecast_df['yhat'].iloc[-(_fore_span)].astype(int)
        data_stat = data_stat.append(data_rsp)
        #preview用データ格納
        data_rsp = pd.DataFrame(columns = [])
        #data_rsp['date'] = pred_df['ds']
        data_rsp['date'] = _date['date']
        data_rsp['predict'] = pred_df['yhat'].round(2)
        data_rsp['trend'] = pred_df['trend'].round(2)
        data_rsp['seasonal'] = pred_df['seasonal'].round(2)
        data_stat_preview = data_stat_preview.append(data_rsp)
        #Result画面でのグラフ用にオリジナルデータ + 予測データ
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = _data_sum['date']
        data_rsp['original'] = _data_sum[predict].round(2)
        #data_all = data_all.append([data_rsp,data_stat_preview])
        data_ori = data_ori.append(data_rsp)
        
    return data_stat,data_stat_preview,data_ori

#SARIMAモデルでの予測
@jit
def forecast_SARIMA(_data,_unique_list,_col,predict,option,session_id):
    estimate_method = option[3] #推定方法
    seasonal = int(option[2]) #シーズナル間隔
    flag = option[5] #最適化計算か決め計算
    parameta = option[6]#パラメータを一旦格納
    #start = '1960-01-01' #予測スタート時
    #end = '1963-12-01' #予測終了時
    
    #データフレーム初期化
    data_rsp = pd.DataFrame(columns = [])
    data_stat = pd.DataFrame(columns = [])
    data_all = pd.DataFrame(columns = [])
    data_stat_preview = pd.DataFrame(columns = [])
    data_ori = pd.DataFrame(columns = [])
    
    for row in _unique_list:
        ###選択された粒度ごとで計算
        mask = np.array(_data[_col]) == row
        mask = np.prod(mask, axis=1).astype(np.bool)
        _data_mask = _data[mask].dropna()
        _data_sum = _data_mask.groupby('date',as_index = False).sum()
        #cap =_data_sum[predict].median() * 3
        
        holdout = len(_data_sum) - int(option[0]) #ホールドアウト期間
        predict_span = len(_data_sum) + int(option[1]) #予測期間

        _data = _data_sum[['date',predict]].dropna()
        ts = _data.groupby('date').sum()
        ts = ts.dropna(axis = 0,how = 'any')
        # 自動SARIMA選択
        num = 0
        max_p = 3
        max_d = 3
        max_q = 1
        max_sp = 1
        max_sd = 1
        max_sq = 1
        pattern = max_p*(max_q + 1)*(max_d + 1)*(max_sp + 1)*(max_sq + 1)*(max_sd + 1)

        modelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic","p","d","q","sp","sd","sq"])
        if flag == False:

            for p in range(1, max_p + 1):
                for d in range(0, max_d + 1):
                    for q in range(0, max_q + 1):
                        for sp in range(0, max_sp + 1):
                            for sd in range(0, max_sd + 1):
                                for sq in range(0, max_sq + 1):
                                    sarima = sm.tsa.SARIMAX(
                                        ts, order=(p,d,q), 
                                        seasonal_order=(sp,sd,sq,seasonal), 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False
                                    ).fit(method='bfgs', maxiter=100, disp=False)
                                    modelSelection.iloc[num]['model'] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), season=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
                                    modelSelection.iloc[num]['aic'] = sarima.aic
                                    modelSelection.iloc[num]['p'] = p
                                    modelSelection.iloc[num]['d'] = d
                                    modelSelection.iloc[num]['q'] = q
                                    modelSelection.iloc[num]['sp'] = sp
                                    modelSelection.iloc[num]['sd'] = sd
                                    modelSelection.iloc[num]['sq'] = sq
                                    num = num + 1
                                    insert_data = ProgressModel(progress = 10+(num/pattern)*89)
                                    insert_data.save()
        else:
            p = int(parameta[0])
            d = int(parameta[1])
            q = int(parameta[2])
            sp = int(parameta[3])
            sd = int(parameta[4])
            sq = int(parameta[5])
            sarima = sm.tsa.SARIMAX(
            ts, order=(p,d,q), 
            seasonal_order=(sp,sd,sq,seasonal), 
            enforce_stationarity = False, 
            enforce_invertibility = False
            ).fit(method='bfgs', maxiter=100, disp=False)
            modelSelection.iloc[0]['model'] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), season=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
            modelSelection.iloc[0]['aic'] = sarima.aic
            modelSelection.iloc[0]['p'] = p
            modelSelection.iloc[0]['d'] = d
            modelSelection.iloc[0]['q'] = q
            modelSelection.iloc[0]['sp'] = sp
            modelSelection.iloc[0]['sd'] = sd
            modelSelection.iloc[0]['sq'] = sq
            insert_data = ProgressModel(progress = 99)
            insert_data.save()

        # モデルごとの結果確認
        #print(modelSelection)

        # AIC最小モデル
        #print(modelSelection[modelSelection.aic == min(modelSelection.aic)])
        p = modelSelection[modelSelection.aic == min(modelSelection.aic)].p
        d = modelSelection[modelSelection.aic == min(modelSelection.aic)].d
        q = modelSelection[modelSelection.aic == min(modelSelection.aic)].q
        sp = modelSelection[modelSelection.aic == min(modelSelection.aic)].sp
        sd = modelSelection[modelSelection.aic == min(modelSelection.aic)].sd
        sq = modelSelection[modelSelection.aic == min(modelSelection.aic)].sq

        bestSARIMA = sm.tsa.SARIMAX(ts, order=(int(p),int(d),int(q)), 
                                    seasonal_order=(int(sp),int(sd),int(sq),seasonal), 
                                    enforce_stationarity = False, 
                                    enforce_invertibility = False).fit()
        
        #表示用計算結果情報
        aic = round(bestSARIMA.aic,3)
        bic = round(bestSARIMA.bic,3)
        hqic = round(bestSARIMA.hqic,3)
        #insert_data = SummaryModel(id = uuid,model = option[4],aic = aic, bic = bic,hqic = hqic,p=p,d=d,q=q,sp=sp,sd=sd,sq=sq,method = estimate_method,session_id=session_id)
        insert_data = SummaryModel(id = uuid,model = option[4],aic = aic, bic = bic,hqic = hqic,p=p,d=d,q=q,sp=sp,sd=sd,sq=sq,method = estimate_method)
        insert_data.save()

        #予測
        #bestPred = bestSARIMA.predict('2017-12-01', '2017-12-31')
        bestPred = bestSARIMA.predict(holdout, predict_span)
        #plt.savefig('./static/stat_application/'+'_'.join(row) +'.jpg')

        #保存用に形式揃え
        pred_df = pd.DataFrame(bestPred.values,columns = ['yhat'])
        pred_df = pred_df.reset_index().rename(columns = {'index':'ds'})
        pred_df['ds'] = pd.to_datetime(bestPred.index,format = '%Y-%m-%d')
        
        #日付横並びのデータ格納
        for (i,j) in zip(_col,row):
            data_rsp[i] = pd.Series(j)
        for (d,v) in zip(pred_df['ds'],pred_df['yhat'].astype(int)):
            data_rsp[d] = pd.Series(v)      
        data_stat = data_stat.append(data_rsp)
        #preview用データ格納
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = pred_df['ds']
        data_rsp['predict'] = pred_df['yhat']
        data_stat_preview = data_stat_preview.append(data_rsp)
        #Result画面でのグラフ用にオリジナルデータ + 予測データ
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = _data_sum['date']
        data_rsp['original'] = _data_sum[predict]
        #data_all = data_all.append([data_rsp,data_stat_preview])
        data_ori = data_ori.append(data_rsp)
    return data_stat,data_stat_preview,data_ori

##状態空間モデル
def forecast_state_space(_data,_unique_list,_col,predict,option,session_id):

    #変数まとめ
    estimate_method = option[3] #推定方法
    seasonal = int(option[2]) #シーズナル間隔
    #start = '1960-01-01' #予測スタート時
    #end = '1963-12-01' #予測終了時
    # グラフを横長にする
    rcParams['figure.figsize'] = 15, 6
    
    #データフレーム初期化
    data_rsp = pd.DataFrame(columns = [])
    data_stat = pd.DataFrame(columns = [])
    data_all = pd.DataFrame(columns = [])
    data_stat_preview = pd.DataFrame(columns = [])
    data_ori = pd.DataFrame(columns = [])

    for row in _unique_list:

        ###選択された粒度ごとで計算
        mask = np.array(_data[_col]) == row
        mask = np.prod(mask, axis=1).astype(np.bool)
        _data_mask = _data[mask].dropna()
        _data_sum = _data_mask.groupby('date',as_index = False).sum()
        
        holdout = len(_data_sum) - int(option[0]) #ホールドアウト期間
        predict_span = len(_data_sum) + int(option[1]) #予測期間

        ts = _data_sum[['date',predict]]
        ts = ts.set_index('date')

        mod_season_trend = sm.tsa.UnobservedComponents(
            ts,
            estimate_method,
            seasonal = seasonal
        )

        # まずはNelder-Meadでパラメタを推定し、その結果を初期値としてまた最適化する。2回目はBFGSを使用。
        res_season_trend = mod_season_trend.fit(
            method='bfgs', 
            maxiter=500, 
            start_params = mod_season_trend.fit(method='nm', maxiter=500).params,
        )

        # 推定されたパラメタ一覧
        print(res_season_trend.summary())
        #表示用計算結果情報
        aic = round(res_season_trend.aic,3)
        bic = round(res_season_trend.bic,3)
        hqic = round(res_season_trend.hqic,3)
        #insert_data = SummaryModel(model = option[4],aic = aic, bic = bic,hqic = hqic,method = estimate_method,session_id=session_id)
        insert_data = SummaryModel(model = option[4],aic = aic, bic = bic,hqic = hqic,method = estimate_method)
        insert_data.save()

        # 推定された状態・トレンド・季節の影響の描画
        rcParams['figure.figsize'] = 15, 20
        fig = res_season_trend.plot_components()

        # 予測
        #pred = res_season_trend.predict(start,end)
        pred = res_season_trend.predict(holdout,predict_span)

        # 実データと予測結果の図示
        rcParams['figure.figsize'] = 15, 6
        #plt.plot(ts)
        #plt.plot(pred, "r")
        plt.savefig(UPLOADE_DIR +'/temp/state_space/pic/'+'_'.join(row) +'.jpg')
        
        #保存用に形式揃え
        pred_df = pd.DataFrame(pred,columns = ['yhat'])
        pred_df = pred_df.reset_index().rename(columns = {'index':'ds'})
        pred_df['ds'] = pd.to_datetime(pred_df['ds'],format = '%Y-%m-%d')
        
        #日付横並びのデータ格納
        for (i,j) in zip(_col,row):
            data_rsp[i] = pd.Series(j)
        for (d,v) in zip(pred_df['ds'],pred_df['yhat'].astype(int)):
            data_rsp[d] = pd.Series(v)      
        data_stat = data_stat.append(data_rsp)
        #preview用データ格納
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = pred_df['ds']
        data_rsp['predict'] = pred_df['yhat']
        data_stat_preview = data_stat_preview.append(data_rsp)
        #Result画面でのグラフ用にオリジナルデータ + 予測データ
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = _data_sum['date']
        data_rsp['original'] = _data_sum[predict]
        #data_all = data_all.append([data_rsp,data_stat_preview])
        data_ori = data_ori.append(data_rsp)
        
    return data_stat,data_stat_preview,data_ori

def forecast_rnn(_data,_unique_list,_col,predict,option,session_id):
    
    #変数まとめ
    estimate_method = option[3] #推定方法
    seasonal = int(option[2]) #シーズナル間隔
    base_data_num = 64 #入力データ数
    forecast_data_num = 16 #出力データ数
    fore_span = base_data_num + forecast_data_num #
    epochs = 100 #epoch数
    start = 40 # スタート地点
    
    #start = '1960-01-01' #予測スタート時
    #end = '1963-12-01' #予測終了時
    
    #データフレーム初期化
    data_rsp = pd.DataFrame(columns = [])
    data_stat = pd.DataFrame(columns = [])
    data_all = pd.DataFrame(columns = [])
    data_stat_preview = pd.DataFrame(columns = [])
    data_ori = pd.DataFrame(columns = [])
    
    for row in _unique_list:

        ###選択された粒度ごとで計算
        mask = np.array(_data[_col]) == row
        mask = np.prod(mask, axis=1).astype(np.bool)
        _data_mask = _data[mask].dropna()
        _data_sum = _data_mask.groupby('date',as_index = False).sum()
        
        holdout = 0.1
        predict_span = 2
        
        ts = _data_sum[['date',predict]]
        ts = ts.set_index('date')

        timeline = np.arange(len(ts))
        #ts[predict] = ts[predict] + (np.random.rand(len(timeline)) * 2)# ノイズ項
        #plt.plot(_data[imp])
        #plt.show()
        #plt.close()

        input_data = []
        output_data = []
        for n in range(len(ts)-fore_span):
            input_data.append(ts[predict][n:n+base_data_num])
            output_data.append(ts[predict][n+base_data_num:n+fore_span])

        train_X = np.reshape(input_data, (-1, base_data_num, 1))
        train_Y = np.reshape(output_data, (-1, forecast_data_num, 1))

        #model作成
        model = Sequential()
        model.add(Conv1D(64, 8, padding='same', input_shape=(base_data_num, 1), activation='relu'))
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Conv1D(64, 8, padding='same', activation='relu'))
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Conv1D(32, 8, padding='same', activation='relu'))
        model.add(Conv1D(1, 8, padding='same', activation='tanh'))

        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(train_X, train_Y, validation_split=holdout, epochs=epochs)

        #plt.plot(range(epochs), history.history['loss'], label='loss')
        #plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
        #plt.xlabel('epoch')
        #plt.ylabel('loss')
        #plt.legend() 
        #plt.show()
        #plt.close()

        _data_np = np.array(ts)
        sheed = np.reshape(ts[predict][start:start+base_data_num], (1, base_data_num, 1))
        prediction = sheed

        for i in range(predict_span):
            res = model.predict(sheed)
            sheed = np.concatenate((sheed[:, forecast_data_num:, :], res), axis=1)
            prediction = np.concatenate((prediction, res), axis=1)

        #print(prediction.shape)
        predictor = np.reshape(prediction, (-1))
        #print(predictor.shape)
        #plt.plot(range(len(predictor)), predictor, label='predict')
        #plt.plot(range(len(predictor)), _data[imp][start:start + len(predictor)], label='real')
        #plt.legend() 
        #plt.show()
        
        #insert_data = SummaryModel(model = option[4],method = estimate_method,session_id=session_id)
        insert_data = SummaryModel(model = option[4],method = estimate_method)
        insert_data.save()
        
        #保存用に形式揃え
        pred_df = pd.DataFrame(predictor,columns = ['yhat'])
        pred_df = pred_df.reset_index().rename(columns = {'index':'ds'})
        pred_df['ds'] = pd.to_datetime(pred_df['ds'],format = '%Y-%m-%d')
        
        #日付横並びのデータ格納
        for (i,j) in zip(_col,row):
            data_rsp[i] = pd.Series(j)
        for (d,v) in zip(range(len(predictor)),predictor.astype(int)):
            data_rsp[d] = pd.Series(v)      
        data_stat = data_stat.append(data_rsp)
        #preview用データ格納
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = [int(i) for i in range(len(predictor))]
        data_rsp['predict'] = predictor
        data_stat_preview = data_stat_preview.append(data_rsp)
        #Result画面でのグラフ用にオリジナルデータ + 予測データ
        data_rsp = pd.DataFrame(columns = [])
        data_rsp['date'] = [int(i) for i in range(len(_data_sum))]
        data_rsp['original'] = _data_sum[predict]
        #data_all = data_all.append([data_rsp,data_stat_preview])
        data_ori = data_ori.append(data_rsp)
    
    return data_stat,data_stat_preview,data_ori

#重回帰分析での予測
def forecast_mlr(_data,_col,predict,option,session_id):
    ###変数まとめ
    estimate_method = option[1]
    
    #データフレーム初期化
    _data_pr = pd.DataFrame(columns = [])
    data_rsp = pd.DataFrame(columns = [])
    data_stat = pd.DataFrame(columns = [])
    data_all = pd.DataFrame(columns = [])
    data_stat_preview = pd.DataFrame(columns = [])
    data_ori = pd.DataFrame(columns = [])
    
    ###選択された粒度ごとで計算
    _data_sum = _data

    ##model入力値
    holdout = int(option[0])
    _data_model = _data[:-(holdout)]
    X = _data_model[_col].fillna(0)
    if int(option[3]) !=1:
        X = sm.add_constant(X)#定数を入れるか
    Y = _data_model[predict].fillna(0)
    X_all = _data[_col].fillna(0)
    Y_all = _data[predict].fillna(0)

    if estimate_method =='ols':
        model = smf.OLS(Y,X)
    if estimate_method =='wls':
        model = smf.WLS(Y,X)
    if estimate_method =='glm_po':
        model = smf.GLM(Y,X,family=sm.families.Poisson())
    # 予測モデルを作成
    result = model.fit()
    result.summary()
    ##サマリをテキスト保存
    f = open( UPLOADE_DIR +'/temp/mlr/text/summary.txt', 'w' ) 
    f.write( str(result.summary()) ) 
    f.close()

    # 予測値計算
    Y_pre = pd.DataFrame(columns = [])
    for i in _col:
        Y_pre[i] = result.params[i]*X_all[i]
    pred_df = Y_pre.sum(axis = 1) + result.params[0]

    #insert_data = SummaryModel(model = option[2],method = estimate_method,aic=round(result.aic,3),bic=round(result.bic,3),rsq=round(result.rsquared,3),rsq_adj=round(result.rsquared_adj,3),holdout = holdout,session_id=session_id)
    insert_data = SummaryModel(model = option[2],method = estimate_method,aic=round(result.aic,3),bic=round(result.bic,3),rsq=round(result.rsquared,3),rsq_adj=round(result.rsquared_adj,3),holdout = holdout)
    insert_data.save()

    #preview用データ格納
    data_rsp = pd.DataFrame(columns = [])
    data_rsp['index'] = [int(i) for i in range(len(_data))]
    data_rsp['original'] = Y_all
    data_rsp['predict'] = pred_df
    data_stat_preview = data_stat_preview.append(data_rsp)
    #Result画面でのグラフ用にオリジナルデータ + 予測データ
    data_rsp = pd.DataFrame(columns = [])
    data_rsp['index'] = [int(i) for i in range(len(_data))]
    data_rsp['original'] = Y_all
    data_rsp['predict'] = pred_df
    data_ori = data_ori.append(data_rsp)
    return data_stat,data_stat_preview,data_ori