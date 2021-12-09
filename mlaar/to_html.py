from catboost import CatBoost, CatBoostClassifier
import datetime
from fastFM import als
from IPython.display import HTML, Image, display
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mpld3
import numpy as np
import os
import pandas as pd
import pdfkit
import re
from scipy import interp
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             explained_variance_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
import sqlite3
from sqlite3 import OperationalError
import sys
import time
import xgboost as xgb
from xgboost import XGBClassifier
from yattag import Doc
import zipfile

try:
    from mlaar import report
    from mlaar import css_js
except:
    import report
    import css_js
    
PKG_PATH = os.path.abspath(os.path.dirname(__file__))
TFF_PATH = os.path.join(PKG_PATH,'ttf/')

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_QPA_FONTDIR'] = TFF_PATH

plot_width=8.27
plot_height=11.69
fontsize=14

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)
    
def print_text(content, bold=False):    
    doc, tag, text = Doc().tagtext()
    doc.stag('br')
    if bold:
        with tag('b'):
            with tag('p', style="color:black", align="left"):
                text(content)
    else:
        with tag('p', style="color:black", align="left"):
            text(content)
    return doc.getvalue()   

def generate_meta_table(name, learning_type, test_data):
    doc, tag, text = Doc().tagtext()
    meta_data = {'Tabulation Time': f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                 'User': os.environ['JUPYTERHUB_USER'].upper(),
                 'Project Name': name,
                 'Learning Type': learning_type,
                 'Model Name': str(type(test_data['model'])).replace("<class '","").replace("'>","")}
    with tag('table', id="customers"):
        for key in meta_data:
            with tag('tr'):
                with tag('th',width=250):
                    text(key)
                with tag('td',width=200): 
                    text(meta_data[key])   
    return doc.getvalue()

def generate_data_shape_value(train_data, test_data):
    train_rows = format(len(train_data['Y']), ',')
    train_cols = format(len(train_data['X_col']), ',')
    test_rows = format(len(test_data['Y']), ',')
    test_cols = format(len(test_data['X_col']), ',')
    return train_rows, train_cols, test_rows, test_cols

def generate_data_shape_table(train_data, test_data):
    train_rows, train_cols, test_rows, test_cols = generate_data_shape_value(train_data, test_data)
    df = pd.DataFrame({'Rows': [train_rows, test_rows],
                       'Columns': [train_cols, test_cols]})
    df.index = ['Training Data','Testing Data']
    return df

def plot_label_dist_regressor(train_data, test_data):
    train_rows, train_cols, test_rows, test_cols = generate_data_shape_value(train_data, test_data)   
    fig = plt.figure(figsize=(8,8))
    fig.tight_layout(pad=1.5)
    for idx, phase in enumerate(['Train', 'Test']):
        locals()['ax'+str(idx+1)] = fig.add_subplot(2,1,idx+1)
        locals()['ax'+str(idx+1)].set_title('Label Distribution - {}ing Data'.format(phase), size=fontsize)
        locals()['ax'+str(idx+1)].hist(locals()[phase.lower()+str('_data')]['Y'], bins=50, alpha=0.75, edgecolor='black')
        locals()['ax'+str(idx+1)].yaxis.set_major_formatter(PercentFormatter(xmax=len(locals()[phase.lower()+str('_data')]['Y'])))
        plt.draw()        # 要加 plt.draw, get_yticklabel才抓得到值
        locals()[phase.lower()+str('_y_ticklabel')] = [item.get_text() for item in locals()['ax'+str(idx+1)].get_yticklabels()]
        percent = len(locals()[phase.lower()+str('_data')]['Y'])/100
        locals()[phase.lower()+str('_y_ticklabel_pos')] = [float(i.replace("%",""))*percent for i in locals()[phase.lower()+str('_y_ticklabel')]]
        locals()['ax'+str(idx+1)].set_yticks(locals()[phase.lower()+str('_y_ticklabel_pos')])
        locals()['ax'+str(idx+1)].set_yticklabels(locals()[phase.lower()+str('_y_ticklabel')])
    return fig

def plot_label_dist_classifier(train_data, test_data):
    train_positive = round(sum(train_data['Y'])/len(train_data['Y'])*100, 2)
    test_positive = round(sum(test_data['Y'])/len(test_data['Y'])*100, 2)
    train_negative = 100 - train_positive
    test_negative = 100 - test_positive
    x_pos = np.linspace(0, 100, 6)
    x = [str(int(i))+'%' for i in x_pos]
    y = ['Train', 'Test']
    y_pos = np.arange(len(y))/2
    positive_y = [train_positive, test_positive]
    negative_y = [train_negative, test_negative]
    fig = plt.figure(figsize=(8,4.5))
    plt.rcParams.update({'legend.fontsize':14})
    ax1 = fig.add_subplot(111)
    fig.tight_layout(pad=3)
    ax1.set_title('Label Distribution',size=fontsize)
    p1 = ax1.barh(y_pos, positive_y, label='positive', height=0.3)
    p2 = ax1.barh(y_pos, negative_y, left=positive_y, label='negative', height=0.3)
    ax1.set_xlim((0, 100))
    ax1.set_ylim((-0.3, 1))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x, fontsize=fontsize)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y, fontsize=fontsize)
    ax1.annotate(str(train_positive)+'%', xy=(train_positive*0.4,y_pos[0]), color = 'white',size=fontsize)
    ax1.annotate(str(test_positive)+'%', xy=(test_positive*0.4,y_pos[1]), color = 'white',size=fontsize)
    ax1.annotate(str(train_negative)+'%', xy=(train_positive+train_negative*0.4,y_pos[0]), color = 'white',size=fontsize)
    ax1.annotate(str(test_negative)+'%', xy=(test_positive+test_negative*0.4,y_pos[1]), color = 'white',size=fontsize)
    ax1.legend(loc = 'upper right')
    return fig

def col_summary(train_data, html_file):    
    if train_data['X_info'] is not None:
        df = train_data['X_info'].copy()
        df['Type Name'] = df.apply(lambda x: x['type'].name,axis=1)
        df = df.groupby('Type Name').apply(lambda x: pd.Series({'Number of Columns':len(x),'Column Name':', '.join(x['name'].tolist())})).reset_index()
        df = df[['Type Name', 'Number of Columns', 'Column Name']]
        html_file.write(print_text('Table 1-3. Column Type{}'.format('s' if len(df)>1 else '')))       
        html_file.write(df_to_html(df, col_width=[150,100,700], layer=1, table_width='100%')) 
        html_file.write(print_text('There {} {} column type{}.'.format('is' if len(df)<=1 else 'are', 
                                                                      len(df), 
                                                                      's' if len(df)>1 else '')))
        return df

def cols_with_na(train_data, html_file):
    df = train_data['X_info'].copy()
    html_file.write('<br /><li><p style="color:black" align="left">Table 1-4. Column{} with NA</p></li>'.format('s' if len(df)>1 else ''))
    df['NA_tag'] = np.where(df.NA_count>0, 'withNA', 'nonNA')
    def count_na(x):
        input_value = x['NA_count']/(x['NA_count']+x['count'])*100
        return pd.Series({'Column Name': x['name'],
                          'NA Ratio': '{}%'.format(report.float_to_display(input_value, n=1)),
                          'Column Type': x['type'],
                          'NA Value': input_value})
    df = df[df.NA_tag=='withNA'].apply(count_na, axis=1).reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame([['no NA', 'no NA', 'no NA']], columns=['Column Name','Column Type','NA Ratio'])
        html_file.write(df_to_html(df, table_width='100%'))
        html_file.write(print_text('There is 0 column with NA.'))
    else:
        df=df.sort_values(['NA Value'], ascending=False)[['Column Name','Column Type','NA Ratio']].reset_index(drop=True)
        html_file.write(df_to_html(df, table_width='100%'))
        html_file.write(print_text('There {} {} column{} with NA.'.format('is' if len(df)<=1 else 'are',
                                                                          len(df), 
                                                                          's' if len(df)>1 else '')))
    return df
    
def cat_col_summary(train_data, html_file):
    df = train_data['X_info'].copy()
    cat_ratio = train_data['cat_ratio'].copy()
    html_file.write(print_text('Table 1-5. Categorical Column{}'.format('s' if len(df)>1 else '')))
    df['Type Name'] = df.apply(lambda x: x['type'].name, axis=1)
    def add_cat_ratio(x):
        lst = []
        for cat in cat_ratio[x['name']].keys():
            lst.append(str(cat) + ' (' + "{0:.1%}".format(cat_ratio[x['name']][cat]) + ')')
        return lst
    def gen_info(x):
        input_value = x['NA_count']/(x['NA_count']+x['count'])*100
        return pd.Series({'Categorical Column':x['name'],
                          'Number of Categories':len(x['type'].categories),
                          'NA Ratio': '{}%'.format(report.float_to_display(input_value, n=1)),
                          'Categories': ', '.join(add_cat_ratio(x))})
    df = df[df['Type Name']=='category'].apply(gen_info, axis=1)
    if df.empty:
        df = pd.DataFrame([['no cat', 'no cat', 'no cat', 'no cat']], columns=['Categorical Column', 'Number of Categories', 'NA Ratio', 'Categories'])
        html_file.write(df_to_html(df, layer=3, table_width='100%'))
        html_file.write(print_text('There is 0 categorical column.'))
    else:
        df = df.sort_values(['Number of Categories'], ascending=False)
        df = df[['Categorical Column', 'Number of Categories', 'NA Ratio', 'Categories']].reset_index(drop=True)
        html_file.write(df_to_html(df, layer=3, table_width='100%')) 
        html_file.write(print_text('There {} {} categorical column{}.'.format('is' if len(df)<=1 else 'are',
                                                                              len(df),
                                                                              's' if len(df)>1 else '')))
    return df

def df_to_html(df, col_width=None, index=False, index_width=100, table_width='80%', layer=0):
    lst = []
    lst.append('<table width={} id="customers" ><tr>'.format(table_width))
    if index:
        lst.append('<th width={}></th>'.format(index_width))
    for idx, colname in enumerate(df.columns):
        if col_width==None:
            lst.append('<th>{}</th>'.format(colname))
        else:
            lst.append('<th width={}>{}</th>'.format(col_width[idx], colname))
    lst.append('</tr>')
    cnt = 1
    row, col = df.shape
    for i in range(row):
        lst.append('<tr{}>'.format('' if i % 2 == 0 else ' class="alt"'))        
        if index:
            lst.append('<th width={}>{}</th>'.format(index_width, df.index[i]))    
        for j in range(col):
            read_more_threshold = 300
            if len(str(df.iloc[i,j])) >= read_more_threshold:
                while str(df.iloc[i,j])[read_more_threshold-1] != ',':
                    read_more_threshold = read_more_threshold - 1
                lst.append('<td>{}<span id="dots{}{}">...</span><span id="more{}{}">{}</span><button class="button" onclick="readMoreFunction(\'dots{}{}\',\'more{}{}\',\'myBtn{}{}\')" id="myBtn{}{}">Read more</button>'.format(df.iloc[i,j][:read_more_threshold-1], layer, cnt, layer, cnt, df.iloc[i,j][:read_more_threshold-1], layer, cnt, layer, cnt, layer, cnt, layer, cnt))
                cnt = cnt + 1            
            else:
                lst.append('<td>{}</td>'.format(df.iloc[i,j]))
        lst.append('</tr>')       
    lst.append('</table>')
    return ''.join(lst)

def show_first_few_decimal(df, n=3):
    for col in df.columns:
        df[col] = pd.Series([round(val,n) for val in df[col]], index=df.index)
    return df
    
def performance_format(df): 
    doc, tag, text = Doc().tagtext()           
    with tag('div', klass="section white"):
        with tag('div', klass="container"):
            with tag('div', klass="p-t-60 p-b-50 "):
                with tag('div', klass="row feature-list"):
                    for i in range(len(df.columns)):    
                        with tag('div', klass="col-md-3"):
                            with tag('h4', klass="custom-font title"):
                                #df variable
                                text(str(df.columns[i]))
                            with tag('h3', klass="custom-font"):
                                with tag('span'):
                                    #df variable
                                    text(str(df.iloc[0,i])+'/'+str(df.iloc[1,i]))
                            with tag('div', klass="col-md-8 no-padding"):
                                with tag('div', klass="progress transparent progress-small no-radius "):
                                    with tag('div', klass="progress-bar progress-bar-black animated-progress-bar "):
                                        pass             
    return doc.getvalue()

def plot_performance_map_html(train_data, test_data, name, threshold, memo, alpha=0.8, initial_threshold=0.5, end_threshold=0.99, step_threshold=0.01):
    splits = np.linspace(0, 1, 100)
    fig1 = plt.figure(figsize=(8,4.5))      
    axs1 = fig1.add_subplot(111)
    fig1.tight_layout(pad=3)
    
    #plot recall precision random curve
    random_precision = len(test_data['Y'][test_data['Y'] ==  1])/len(test_data['Y'])
    axs1.plot([0,1], [random_precision,random_precision], linestyle = '--', lw = 2, color = 'r', label = 'random', alpha = .8)
    
    # plot train recall precision
    train_precisions, train_recalls, train_ts = report.plot_precision_recall(data=train_data,ax=axs1,alpha=alpha,initial_threshold=initial_threshold,end_threshold=end_threshold,step_threshold=step_threshold)
    # plot recall precision
    test_precisions, test_recalls, test_ts = report.plot_precision_recall(data=test_data,ax=axs1,alpha=alpha,initial_threshold=initial_threshold,end_threshold=end_threshold,step_threshold=step_threshold)
    
    axs1.legend(loc="best")
    
    fig2 = plt.figure(figsize=(8,4.5))      
    axs2 = fig2.add_subplot(111)
    fig2.tight_layout(pad=3)

    # plot train validation AUC
    axs2.plot([0, 1], [0, 1], linestyle='--', lw=2,color='r', label='random', alpha=.8)
    train_evaluation_result = report.plot_validation_AUC(data=train_data,name=name,ax=axs2,alpha=alpha,threshold=report.best_threshold(train_precisions, train_recalls, train_ts) if threshold is None else threshold,splits=splits,memo=memo)
        
    # plot validation AUC
    test_evaluation_result = report.plot_validation_AUC(data=test_data,name=name,ax=axs2,alpha=alpha,threshold=report.best_threshold(train_precisions, train_recalls, train_ts) if threshold is None else threshold,splits=splits,memo=memo)
        
    performance_tb = report.gen_model_perform_classifier(train_evaluation_result,test_evaluation_result, test_cols=len(test_data['X_col']))
    
    confusion_matrix_train = report.gen_confusion_matrix(train_data,threshold=train_evaluation_result['threshold'].iloc[0])
    confusion_matrix_train = show_first_few_decimal(confusion_matrix_train)
        
    confusion_matrix_test = report.gen_confusion_matrix(test_data,threshold=test_evaluation_result['threshold'].iloc[0])
    confusion_matrix_test = show_first_few_decimal(confusion_matrix_test)
    
    # plot f1
    fig3 = plt.figure(figsize=(8,4.5))      
    axs3 = fig3.add_subplot(111)
    fig3.tight_layout(pad=3)
    
    report.plot_f1(ds_phase='train', precisions=train_precisions, recalls=train_recalls, ts=train_ts, alpha=alpha, ax=axs3)
    report.plot_f1(ds_phase='test', precisions=test_precisions, recalls=test_recalls, ts=test_ts, alpha=alpha, ax=axs3)
        
    return performance_tb, confusion_matrix_train, confusion_matrix_test, fig1, fig2, fig3

def plot_regressor_performance_map_html(train_data, test_data, name, memo, params):
    test_evaluation_result = report.eval_regressor(data=test_data, name=name, memo=memo)
    train_evaluation_result = report.eval_regressor(data=train_data, name=name, memo=memo)
    if test_data['evals_result'] is not None:
        fig = plt.figure(figsize=(8,4.5))      
        ax1 = fig.add_subplot(111)
        ax1.tick_params(axis='y', which='major', pad=50)
        mapp = {
                'mse':'l2',
                'mae':'l1',
                'rmse':'rmse',
                'root_mean_squared_error':'rmse',
                'l2_root':'rmse',
                'mean_squared_error':'l2',
                'mean_absolute_error':'l1'
                }
        if type(test_data['model']) == lgb.basic.Booster:
            lgb.plot_metric(test_data['evals_result'], metric=mapp[params['metric']], ax=ax1)
            ax1.set_title('Metric During Training', size=fontsize)
            ax1.set_xlabel('Iterations', size=fontsize)
            ax1.set_ylabel(params['metric'], size=fontsize)
            fig.tight_layout()    
    performance_tb = report.gen_model_perform_regressor(train_evaluation_result, test_evaluation_result, test_cols=len(test_data['X_col']))
    return performance_tb, fig  

def plot_feature_importance(test_data):
    fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(111)
    importance = test_data['importance'].sort_values('importance', ascending=True)
    importance_total = sum(importance['importance'].astype('int'))
    importance['importance_ratio'] = round(importance['importance']/importance_total*100, 2)
    importance = importance.tail(20)
    importance = importance[importance.importance != 0]
    importance['ylabel'] = importance['importance'].rank(method='min', ascending=False).astype('int').astype('str')
    importance = importance.reset_index().reset_index()
    ax1.set_title('Feature Importance', size=fontsize)
    ax1.set_xlabel('Importance', size=fontsize)
    ax1.set_ylabel('Feature', size=fontsize)
    ax1.set_yticks(range(len(importance)))
    ax1.set_yticklabels(importance['ylabel'], size=fontsize)
    ax1.barh(importance['feature'], importance['importance'], height=0.5)
    ax1.set_xlim((0, max(importance['importance'])*4))
    for i in range(0, len(importance['importance'])):    
        ax1.annotate(str(importance['feature'][i])+'-'+str(importance['importance_ratio'][i])+'%'+' ('+str(importance['importance'][i])+')', xy=(importance['importance'][i]+2, importance['level_0'][i]), color='black', size=16)
    return fig

def plot_feature_importance_lgb(test_data):
    importance_type_list = ['split','gain']
    fig = plt.figure(figsize=(12,12))
    fig.tight_layout(pad=3)
    fig.suptitle('Feature Importance', fontsize=16)
    for i,j in enumerate(importance_type_list):
        locals()['ax'+str(i+1)] = fig.add_subplot(1,2,i+1)  
        importance = test_data['importance'].sort_values('importance_'+j, ascending=True)
        importance_total = sum(importance['importance_'+j].astype('int'))
        importance['importance_ratio'] = round(importance['importance_'+j]/importance_total*100, 2)
        importance = importance.tail(20)
        importance['importance_'+j] = round(importance['importance_'+j], 2)
        importance = importance[importance['importance_'+j] != 0]
        importance['ylabel'] = importance['importance_'+j].rank(method='min', ascending=False).astype('int').astype('str')
        importance = importance.reset_index().reset_index()
        locals()['ax'+str(i+1)].set_xlabel('Importance_'+j, size=fontsize)
        locals()['ax'+str(i+1)].set_ylabel('Feature', size=fontsize)
        locals()['ax'+str(i+1)].set_yticks(range(len(importance)))
        locals()['ax'+str(i+1)].set_yticklabels(importance['ylabel'], size=fontsize)
        locals()['ax'+str(i+1)].barh(importance['feature'],importance['importance_'+j], height=0.5)
        locals()['ax'+str(i+1)].set_xlim((0,max(importance['importance_'+j])*4))
        for k in range(0, len(importance['importance_'+j])):    
            locals()['ax'+str(i+1)].annotate(str(importance['feature'][k])+'-'+str(importance['importance_ratio'][k])+'%'+' ('+str(importance['importance_'+j][k])+')', xy=(importance['importance_'+j][k]+2, importance['level_0'][k]), color='black', size=16)
    return fig

def plot_single_feature_perform_lgb(test_data, learning_type):
    single_feature_performance = {}
    y_true = test_data['Y'].tolist()
    if learning_type == 'classifier':
        performance_index = 'AUC'
        for col in test_data['single_feature_pred']:
            test_precisions, test_recalls, test_ts = report.gen_precision_recall(data=test_data, alpha=0.8, initial_threshold=0, end_threshold=0.99, step_threshold=0.01, single_feature=col)
            fpr, tpr, thresholds = roc_curve(test_data['Y'], test_data['single_feature_pred'][col])  
            test_auc = auc(fpr, tpr)
            #print('test_auc of {}: {}'.format(col, test_auc))
            single_feature_performance[col] = [test_auc] 
    elif learning_type == 'regressor':
        performance_index = 'RMSE'
        for col in test_data['single_feature_pred']:
            y_pred = test_data['single_feature_pred'][col]
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            rmse = np.sqrt(mse)
            single_feature_performance[col] = [rmse]     
    df = pd.DataFrame(single_feature_performance).T
    df = df.reset_index()
    df.columns = ['feature', performance_index]
    fig = plt.figure(figsize=(8,12))
    fig.tight_layout(pad=3)
    fig.suptitle('Single Feature Importance', fontsize=16)
    ax = fig.add_subplot(111)
    if learning_type == 'classifier':
        df = df.sort_values(performance_index, ascending=True)
        df['ylabel'] = df[performance_index].rank(method='min', ascending=False).astype('int').astype('str')
    elif learning_type == 'regressor':
        df = df.sort_values(performance_index, ascending=False)
        df['ylabel'] = df[performance_index].rank(method='min', ascending=True).astype('int').astype('str')
    df = df.tail(20)
    df[performance_index] = round(df[performance_index],3)
    df = df.reset_index().reset_index()
    ax.set_xlabel('Single feature performance ('+performance_index+')', size=fontsize)
    ax.set_ylabel('Feature', size=fontsize)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['ylabel'], size=fontsize)
    ax.barh(df['feature'], df[performance_index], height=0.5)
    ax.set_xlim((0, max(df[performance_index])*4))
    for k in range(0, len(df[performance_index])):   
        ax.annotate(str(df['feature'][k])+'-'+str(df[performance_index][k]), xy=(df[performance_index][k]+0.15, df['level_0'][k]), color='black', size=16)
    return fig

def feature_importance_summary(data):
    
    feature_name = data['importance'].sort_values('importance_gain',ascending=False).head(20).feature
    X_info_df = data['X_info'].copy()
    X_info_df = X_info_df[X_info_df.name.isin(feature_name)]
    def count_na(x):
        input_value = x['NA_count']/(x['NA_count']+x['count'])*100
        return pd.Series({'Column Name': x['name'],
                            'Column Type': x['type'],
                            'min': '{:.2f}'.format(x['min']),
                            'mean': '{:.2f}'.format(x['mean']),
                            'max': '{:.2f}'.format(x['max']),
                            'NA Ratio': '{}%'.format(report.float_to_display(input_value, n=1))
                            })
        
    X_info_df = X_info_df.apply(count_na, axis=1).reset_index(drop=True)
    return X_info_df

def fig_to_html(fig):
    html_fig = mpld3.fig_to_html(fig)
    # localize the .js path
    html_fig = html_fig.replace('https://mpld3.github.io/','') 
    #solve /n problem
    html_fig = html_fig.replace('></div>',' white-space: pre-line, style="text-align:left;padding: 0"></div>') 
    return html_fig
        
def generate_html(test_data, train_data, threshold, name, file_name, learning_type, memo, output_format, params, model, single_feature_num):
    
    current_date  = datetime.datetime.now()
    if file_name == None:
        file_name = '{}_{}.html'.format(name, current_date.strftime('%Y%m%d'))

    folder_name = name + '_' + current_date.strftime('%Y%m%d')

    create_folder(folder_name)
    css_js.generateCss(folder_name)
    css_js.generateJs(folder_name)
    
    html_file= open(folder_name + '/' + name + ".html", "w")

    doc, tag, text = Doc().tagtext()

    doc.stag('html', lang="zh_TW") 

    with tag('head'):
        with tag('style'):
            font_style = '''
            @font-face{
                font-family: Light;
                src: url(Light.ttf);
            }
            body{
                font-family:Light;
            }
            '''
            text(font_style)
        doc.stag('br')
        with tag('title'):
            text('MLaaR模型效能報表')
        doc.stag('meta', charset="UTF-8")
        doc.stag('meta', name="viewport", content="width=device-width, initial-scale=1")  
        doc.stag('link', rel="stylesheet", href="css/css_content.css")
        with tag('script', type="text/javascript", src="js/mpld3.v0.3.js"):
            pass
        with tag('script', type="text/javascript", src="js/d3.v3.min.js"):
            pass
        with tag('script', type="text/javascript", src="js/js_content.js"):
            pass        
    with tag('body'):
        with tag('div'):
            with tag('strong', style="align:left;font:6"):
                text('【MLaaR - Machine Learning as a Report】')      
    doc.stag('br')       
    
    html_file.write(doc.getvalue())
    
    html_file.write(print_text('1. Brief Description of the Data', bold=True))
    
    html_file.write(print_text('Table 1-1. Meta Data'))
    html_file.write(generate_meta_table(name=name, learning_type=learning_type, test_data=test_data))                        
    html_file.write('<br />')
    
    html_file.write(print_text('Table 1-2. Data Shape'))
    html_file.write(df_to_html(generate_data_shape_table(train_data=train_data, test_data=test_data),
                               col_width=[100,100], index=True, index_width=250, table_width='40%'))
    html_file.write('<br />')
    
    html_file.write(print_text('Figure 1-1. Label Distribution'))
    html_file.write('<br />')

    if learning_type == 'classifier':
        html_file.write(fig_to_html(plot_label_dist_classifier(train_data=train_data, test_data=test_data)))
    if learning_type == 'regressor':
        html_file.write(fig_to_html(plot_label_dist_regressor(train_data=train_data, test_data=test_data)))
        
    btn_style = '''
    <script>
    function readMoreFunction(x,y,z) {
      var dots = document.getElementById(x);
      var moreText = document.getElementById(y);
      var btnText = document.getElementById(z);

      if (dots.style.display === "none") {
        dots.style.display = "inline";
        btnText.innerHTML = "Read more"; 
        moreText.style.display = "none";
      } else {
        dots.style.display = "none";
        btnText.innerHTML = "Read less"; 
        moreText.style.display = "inline";
      }
    }
    </script>
    '''
    
    html_file.write(btn_style)
    html_file.write('<div><ul>')
    
    html_file.write('<li>')
    col_summary(train_data, html_file)
    html_file.write('</li><br />') 
    
    html_file.write('<li>')
    cols_with_na(train_data, html_file)
    html_file.write('</li><br />')
    
    html_file.write('<li>')
    cat_col_summary(train_data, html_file)
    html_file.write('</li><br />')

    html_file.write(print_text('2. Model Hyperparameters and Performance', bold=True))
    data_param_info, model_param_info = report.show_model_params(test_data)
    html_file.write(print_text('Table 2-1. Model Hyperparameters'))
    html_file.write(df_to_html(data_param_info, col_width=[150,250], index=True, index_width=200, table_width='60%')) 
    html_file.write(print_text('Note: * means your hyperparameters must be wrong.'))
    html_file.write('<br />')
    
    if learning_type == 'classifier':
        performance_tb, confusion_matrix_train, confusion_matrix_test, performance_map_fig1, performance_map_fig2, performance_map_fig3 = plot_performance_map_html(train_data=train_data, test_data=test_data, alpha=0.8, name=name, threshold=threshold, initial_threshold=0, end_threshold=0.99, step_threshold=0.01, memo=memo)
            
        html_file.write(print_text('Table 2-2. Model Performance'))
        html_file.write(df_to_html(performance_tb, col_width=[100]*9, index=True, table_width='100%'))
        html_file.write('<br />')
        
        html_file.write('<li>')
        html_file.write(print_text('Table 2-3. Confusion Matrix - Training Data'))
        html_file.write(df_to_html(confusion_matrix_train, col_width=[100,100], index=True, index_width=100, table_width='50%'))
        html_file.write('</li><br />')
        
        html_file.write('<li>')
        html_file.write(print_text('Table 2-4. Confusion Matrix - Testing Data'))
        html_file.write(df_to_html(confusion_matrix_test, col_width=[100,100], index=True, index_width=100, table_width='50%'))
        html_file.write('</li><br />')
        
        html_file.write(print_text('Figure 2-1. ROC Curves'))
        html_file.write(fig_to_html(performance_map_fig2))

        html_file.write(print_text('Figure 2-2. Precision-Recall Curves'))
        html_file.write(fig_to_html(performance_map_fig1))

        html_file.write(print_text('Figure 2-3. F1 Measure Curves'))
        html_file.write(fig_to_html(performance_map_fig3))
        
        html_file.write(print_text('Figure 2-4. Feature Importance'))
        try:
            if type(model) == lgb.basic.Booster:
                html_file.write(fig_to_html(plot_feature_importance_lgb(test_data)))
                html_file.write(print_text('【Importance_split】Age-22.06% (60) means importance of Age is 60 and accounts for 22.06% of total split times.'))
                html_file.write(print_text('【Importance_gain】Sex-61.27% (1091.77) means importance of Sex is 1091.77 and accounts for 61.27% of total gains.'))
                html_file.write('<br />')
                html_file.write(print_text('Table 2-5. Feature Importance(gain top20) summary - Train Data'))
                feature_top20_summary = feature_importance_summary(train_data)
                html_file.write(df_to_html(feature_top20_summary, col_width=[100,100,100,100,100,100], index=False, index_width=100, table_width='50%'))
                html_file.write(print_text('Table 2-6. Feature Importance(gain top20) summary - Test Data'))
                feature_top20_summary = feature_importance_summary(test_data)
                html_file.write(df_to_html(feature_top20_summary, col_width=[100,100,100,100,100,100], index=False, index_width=100, table_width='50%'))
                if single_feature_num > 0:
                    html_file.write(print_text('Figure 2-5. Single Feature Performance'))
                    html_file.write(fig_to_html(plot_single_feature_perform_lgb(test_data, learning_type)))
            else:
                html_file.write(fig_to_html(plot_feature_importance(test_data)))   
        except:
            print('Error in plotting feature importance or single feature performance.')
            html_file.write(print_text('NaN'))
            
        html_file.write('<br />')    
        
        html_file.write('<li>')
        html_file.write(print_text('Table 2-7. Past Model Performance'))
        test_hist_df = report.gen_hist_data(ds_phase='test', learning_type=learning_type, name=name)
        test_hist_df['Phase'] = 'Test'
        test_hist_df = test_hist_df[['ts', 'memo', 'Phase', 'count', 'count_of_col' ,'f1', 'precision', 'recall']]
        test_hist_df.columns = ['Production Time', 'Memo', 'Phase', 'Rows', 'Columns', 'F1', 'Precision', 'Recall']
        for col in ['Rows', 'Columns']:
            test_hist_df[col] = test_hist_df[col].apply(lambda x: format(x, ','))
        html_file.write(df_to_html(test_hist_df, col_width=[150,300,100,100,100,100,100,100], table_width='100%'))
        html_file.write('</li>')
        
    if learning_type == 'regressor':
        performance_tb, performance_map_fig = plot_regressor_performance_map_html(train_data=train_data, test_data=test_data, 
                                                                                  name=name, memo=memo, params=params)
        html_file.write(print_text('Table 2-2. Model Performance'))
        html_file.write(df_to_html(performance_tb, col_width=[100]*6, index=True, table_width='80%'))
                
        html_file.write(print_text('Figure 2-1. Metric During Training'))
        html_file.write(fig_to_html(performance_map_fig))
        
        try:
            html_file.write(print_text('Figure 2-2. Feature Importance'))
            if type(model) == lgb.basic.Booster:
                html_file.write(fig_to_html(plot_feature_importance_lgb(test_data)))
                html_file.write(print_text('【Importance_split】Age-22.06% (60) means importance of Age is 60 and accounts for 22.06% of total split times.'))
                html_file.write(print_text('【Importance_gain】Sex-61.27% (1091.77) means importance of Sex is 1091.77 and accounts for 61.27% of total gains.'))
                html_file.write('<br />')
                html_file.write(print_text('Table 2-3. Feature Importance(gain top20) summary - Train Data'))
                feature_top20_summary = feature_importance_summary(train_data)
                html_file.write(df_to_html(feature_top20_summary, col_width=[100,100,100,100,100,100], index=False, index_width=100, table_width='50%'))
                html_file.write(print_text('Table 2-4. Feature Importance(gain top20) summary - Test Data'))
                feature_top20_summary = feature_importance_summary(test_data)
                html_file.write(df_to_html(feature_top20_summary, col_width=[100,100,100,100,100,100], index=False, index_width=100, table_width='50%'))
                if single_feature_num > 0:
                    html_file.write(print_text('Figure 2-3. Single Feature Performance'))
                    html_file.write(fig_to_html(plot_single_feature_perform_lgb(test_data, learning_type)))
            else:
                html_file.write(fig_to_html(plot_feature_importance(test_data)))
        except:
            print('Error in plotting Figure 2-2 or Figure 2-3.')
            html_file.write(print_text('NaN'))
        html_file.write('<br />')
        
        html_file.write('<li>')
        html_file.write(print_text('Table 2-5. Past Model Performance'))
        test_hist_df = report.gen_hist_data(ds_phase='test', learning_type=learning_type, name=name)
        test_hist_df['Phase'] = 'Test'
        test_hist_df = test_hist_df[['ts', 'memo', 'Phase', 'count', 'count_of_col', 'RMSE', 'MAPE']]
        test_hist_df.columns = ['Production Time', 'Memo', 'Phase', 'Rows', 'Columns', 'RMSE', 'MAPE']
        for col in ['Rows', 'Columns', 'RMSE', 'MAPE']:
            test_hist_df[col] = test_hist_df[col].apply(lambda x: format(x, ','))
        html_file.write(df_to_html(test_hist_df, col_width=[150,300,100,100,100,100,100], table_width='100%'))
        html_file.write('</li>')      
    html_file.close()
    
    if output_format == 'pdf':
        try:
            pdfkit.from_file(folder_name + '/' + name + '.html', folder_name + '.pdf')
        except:
            pass
        shutil.rmtree(folder_name)
        
    
    
    
    