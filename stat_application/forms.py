from django import forms

EMPTY_CHOICES = (
    ('', '-'*10),
)
STATE_SPACE_METHOD_CHOICES = (
    ('local level', 'ローカルレベルモデル'),
    ('local linear trend', 'ローカル線形トレンドモデル'),
    ('local linear deterministic trend', 'ローカル線形トレンドモデル(トレンドの分散なし)'),
    ('random walk with drift', 'ローカル線形トレンドモデル(トレンドと観測誤差の分散なし)')
)

REGRESSION_METHOD_CHOICES = (
    ('ols', '線形回帰モデル(最小二乗法：※デフォルト)'),
    ('wls', '線形回帰モデル(重み付き最小二乗法)'),
    #('glm_po', '一般化線形モデル(ポアソン回帰)'),
)

SCORING_CHOICES = (
    ('roc_auc', 'AUC'),
    ('accuracy', 'Accuracy'),
)

FLAG_CHOICES = (
    (1, 'あり'),
)
AUTOFLAG_CHOICES = (
    (1, 'なし'),
)

PROPHET_METHOD_CHOICE = (
    ('linear','線形モデル'),
    ('logistic','非線形モデル')
)

DATE_CHOICE = (
    ('M','月'),
    ('W','週'),
    ('D','日'),
    ('H','時間')
)

class DfColumnForm(forms.Form):

        df_data = forms.ChoiceField(
        label='説明変数',
        widget=forms.CheckboxSelectMultiple(attrs ={'onClick':'checkRegValue(this);'},),
        choices=[],
        required=True,
    )
        df_goal = forms.ChoiceField(
        label='目的変数',
        widget=forms.RadioSelect(attrs ={'onClick':'checkRegValue(this);'},),
        choices=[],
        required=True,
    )
        df_date = forms.ChoiceField(
        label='日付項目',
        widget=forms.RadioSelect(attrs ={'onClick':'checkSimValue(this);'},),
        choices=[],
        required=True,
    )
        df_holdout = forms.IntegerField(
        label='HoldOut',
        initial=int(10),
        min_value = int(1), 
        max_value = int(500), 
        required=True,
    )
        df_predictspan= forms.IntegerField(
        label='予測期間',
        initial=int(20),
        min_value = int(1), 
        max_value = int(500), 
        required=True,
    )
        df_seasonal= forms.IntegerField(
        label='seasonal',
        initial=int(12),
        min_value = int(0),
        max_value = int(1000),
        required=True,
    )
        df_predict = forms.ChoiceField(
        label='予測対象',
        widget=forms.RadioSelect(attrs ={'onClick':'checkSimValue(this);'},),
        choices=[],
        required=True,
    )
        df_date_unit = forms.ChoiceField(
        label='日付の単位',
        widget=forms.Select,
        choices=DATE_CHOICE,
        required=True,
    )
        df_state_space_method = forms.ChoiceField(
        label='state_space_method',
        widget=forms.Select,
        choices=STATE_SPACE_METHOD_CHOICES,
        required=True,
    )
        df_prophet_method = forms.ChoiceField(
        label='prophet_method',
        widget=forms.Select(attrs ={'onChange':'selectValueProphet(this);'},),
        choices=PROPHET_METHOD_CHOICE,
        required=True,
    )
        df_flag = forms.ChoiceField(
        label='flag',
        widget=forms.CheckboxSelectMultiple(attrs ={'onClick':'checkCapValue(this);'},),
        choices=FLAG_CHOICES,
        required=True,
        disabled = True,
    )
        df_autoflag = forms.ChoiceField(
        label='autoflag',
        widget=forms.CheckboxSelectMultiple(attrs ={'onClick':'checkAutoValue(this);'},),
        choices=AUTOFLAG_CHOICES,
        required=True,
    )
        df_constflag = forms.ChoiceField(
        label='constflag',
        widget=forms.CheckboxSelectMultiple(),
        choices=AUTOFLAG_CHOICES,
        required=True,
    )
        df_cap= forms.IntegerField(
        label='cap',
        initial=int(0),
        min_value = int(0),
        required=True,
        disabled = True,
    )

        df_p = forms.IntegerField(
        label='p',
        initial=int(3),
        min_value = int(0),
        max_value = int(3),
        required=True,
        disabled = True,
    )
        df_d = forms.IntegerField(
        label='d',
        initial=int(3),
        min_value = int(0),
        max_value = int(3),
        required=True,
        disabled = True,
    )
        df_q = forms.IntegerField(
        label='q',
        initial=int(1),
        min_value = int(0),
        max_value = int(3),
        required=True,
        disabled = True,
    )
        df_sp = forms.IntegerField(
        label='sp',
        initial=int(1),
        min_value = int(0),
        max_value = int(3),
        required=True,
        disabled = True,
    )
        df_sd = forms.IntegerField(
        label='sd',
        initial=int(1),
        min_value = int(0),
        max_value = int(3),
        required=True,
        disabled = True,
    )
        df_sq = forms.IntegerField(
        label='sq',
        initial=int(1),
        min_value = int(0),
        max_value = int(3),
        required=True,
        disabled = True,
    )
        df_regression_method = forms.ChoiceField(
        label='state_space_method',
        widget=forms.Select,
        choices=REGRESSION_METHOD_CHOICES,
        required=True,
    )
        df_score = forms.ChoiceField(
        label='score',
        widget=forms.Select,
        choices=SCORING_CHOICES,
        required=True,
    )           
                