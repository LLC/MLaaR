# -*- coding: utf-8 -*-

"""model performance evaluation module

Example:
        $ python mlaar/examples/titanic_classifier.py

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
"""

from catboost import CatBoost, CatBoostClassifier
import datetime
from fastFM import als
from IPython.display import HTML, Image, display
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.plotting import table as pd_plot_table
import re
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             explained_variance_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
import sqlite3
import sys
from xgboost import XGBClassifier

if 'ipykernel' not in plt.get_backend():
    plt.switch_backend('Agg')
try:
    from mlaar import data_util as du
    from mlaar import esun_automl
    from mlaar import to_html
except:
    import data_util as du
    import esun_automl
    import to_html

plot_width = 8.27
plot_height = 11.69
fontsize = 14

#parameter dictionary
lightgbm_param = {'config':['config_file'],
                 'task':['task_type'],
                 'objective':['objective_type', 'app', 'application'],
                 'boosting':['boosting_type', 'boost'],
                 'data':['train', 'train_data', 'train_data_file', 'data_filename'],
                 'valid':['test', 'valid_data', 'valid_data_file', 'test_data', 'test_data_file', 'valid_filenames'],
                 'num_iterations':['num_iteration', 'n_iter', 'num_tree', 'num_trees', 'num_round', 'num_rounds', 'num_boost_round', 'n_estimators'],
                 'learning_rate':['shrinkage_rate', 'eta'],
                 'num_leaves':['num_leaf', 'max_leaves', 'max_leaf'],
                 'tree_learner':['tree', 'tree_type', 'tree_learner_type'],
                 'num_threads':['num_thread', 'nthread', 'nthreads', 'n_jobs'],
                 'device_type':['device'],
                 'seed':['random_seed', 'random_state'],
                 'max_depth':['None'],
                 'min_data_in_leaf':['min_data_per_leaf', 'min_data', 'min_child_samples'],
                 'min_sum_hessian_in_leaf':['min_sum_hessian_per_leaf', 'min_sum_hessian', 'min_hessian', 'min_child_weight'],
                 'bagging_fraction':['sub_row', 'subsample', 'bagging'],
                 'bagging_freq':['subsample_freq'],
                 'bagging_seed':['bagging_fraction_seed'],
                 'feature_fraction':['sub_feature', 'colsample_bytree'],
                 'feature_fraction_seed':['None'],
                 'early_stopping_round':['early_stopping_rounds', 'early_stopping'],
                 'max_delta_step':['max_tree_output', 'max_leaf_output'],
                 'lambda_l1':['reg_alpha'],
                 'lambda_l2':['reg_lambda'],
                 'min_gain_to_split':['min_split_gain'],
                 'drop_rate':['rate_drop'],
                 'max_drop':['None'],
                 'skip_drop':['None'],
                 'xgboost_dart_mode':['None'],
                 'uniform_drop':['None'],
                 'drop_seed':['None'],
                 'top_rate':['None'],
                 'other_rate':['None'],
                 'min_data_per_group':['None'],
                 'max_cat_threshold':['None'],
                 'cat_l2':['None'],
                 'cat_smooth':['None'],
                 'max_cat_to_onehot':['None'],
                 'top_k':['topk'],
                 'monotone_constraints':['None'],
                 'feature_contri':['None'],
                 'forcedsplits_filename':['fs', 'forced_splits_filename', 'forced_splits_file', 'forced_splits'],
                 'refit_decay_rate':['None'],
                 'cegb_tradeoff':['None'],
                 'cegb_penalty_split':['None'],
                 'cegb_penalty_feature_lazy':['None'],
                 'cegb_penalty_feature_coupled':['None'],
                 'verbosity':['verbose'],
                 'max_bin':['None'],
                 'min_data_in_bin':['None'],
                 'bin_construct_sample_cnt':['bin_construct_sample_cnt'],
                 'histogram_pool_size':['hist_pool_size'],
                 'data_random_seed':['data_seed'],
                 'output_model':['model_output', 'model_out'],
                 'snapshot_freq':['save_period'],
                 'input_model':['model_input', 'model_in'],
                 'output_result':['predict_result', 'prediction_result', 'predict_name', 'prediction_name', 'pred_name', 'name_pred'],
                 'initscore_filename':['init_score_filename', 'init_score_file', 'init_score', 'input_init_score'],
                 'valid_data_initscores':['valid_data_init_scores', 'valid_init_score_file', 'valid_init_score'],
                 'pre_partition':['is_pre_partition'],
                 'enable_bundle':['is_enable_bundle', 'bundle'],
                 'max_conflict_rate':['None'],
                 'is_enable_sparse':['is_sparse', 'enable_sparse', 'sparse'],
                 'sparse_threshold':['None'],
                 'use_missing':['None'],
                 'zero_as_missing':['None'],
                 'two_round':['two_round_loading', 'use_two_round_loading'],
                 'save_binary':['is_save_binary', 'is_save_binary_file'],
                 'header':['has_header'],
                 'label_column':['label'],
                 'weight_column':['weight'],
                 'group_column':['group', 'group_id', 'query_column', 'query', 'query_id'],
                 'ignore_column':['ignore_feature', 'blacklist'],
                 'categorical_feature':['cat_feature', 'categorical_column', 'cat_column'],
                 'predict_raw_score':['is_predict_raw_score', 'predict_rawscore', 'raw_score'],
                 'predict_leaf_index':['is_predict_leaf_index', 'leaf_index'],
                 'predict_contrib':['is_predict_contrib', 'contrib'],
                 'num_iteration_predict':['None'],
                 'pred_early_stop':['None'],
                 'pred_early_stop_freq':['None'],
                 'pred_early_stop_margin':['None'],
                 'convert_model_language':['None'],
                 'convert_model':['convert_model_file'],
                 'num_class':['num_classes'],
                 'is_unbalance':['unbalance', 'unbalanced_sets'],
                 'scale_pos_weight':['None'],
                 'sigmoid':['None'],
                 'boost_from_average':['None'],
                 'reg_sqrt':['None'],
                 'alpha':['None'],
                 'fair_c':['None'],
                 'poisson_max_delta_step':['None'],
                 'tweedie_variance_power':['None'],
                 'max_position':['None'],
                 'label_gain':['None'],
                 'metric':['metrics', 'metric_types'],
                 'metric_freq':['output_freq'],
                 'is_provide_training_metric':['training_metric', 'is_training_metric', 'train_metric'],
                 'eval_at':['ndcg_eval_at', 'ndcg_at', 'map_eval_at', 'map_at'],
                 'num_machines':['num_machine'],
                 'local_listen_port':['local_port'],
                 'time_out':['None'],
                 'machine_list_filename':['machine_list_file', 'machine_list', 'mlist'],
                 'machines':['workers', 'nodes'],
                 'gpu_platform_id':['None'],
                 'gpu_device_id':['None'],
                 'gpu_use_dp':['None'],
                 'class_weight':['None'],
                 'silent':['None'],
                 'subsample_for_bin':['None']}


def generate_report_material(X, Y, G, Y_pred, name, model, execution_time, ds_phase, params, learning_type, evals_result, final, sample_size_gb=1):

    """generate report material
    Args:
        X (pd.Dataframe): x1~xn.
        Y (pd.series): y.
        G (pd.series): group
        Y_pred (pd.series): predict_prob
        name (str): project name
        model (*): input model
        execution_time (int/float): execution time
        ds_phase (str): 'train' or 'test'
        params (dict): model params
        learning_type (str): 'classifier' or 'regressor'
        evals_result (dict): learning curve material
        sample_size_gb(float): default 1 GB
    Returns:
        dict: report material
    """  
    if X is not None:
        X_info, memory_usage = du.get_basic_info(X)
        
        # compute correlation between Y and each features in X
        X_size = X.memory_usage().sum()/1024**3
        if X_size > sample_size_gb:
            X_sample = X.sample(int(len(X)/X_size), random_state=42)
            Y_sample = Y.sample(int(len(X)/X_size), random_state=42)
            Y_pred_sample = pd.DataFrame(Y_pred).sample(int(len(X)/X_size), random_state=42).values.reshape(1, int(len(X)/X_size))[0]        
            corr = du.get_corr_and_pvals(X_sample, Y_sample)
        else:
            corr = du.get_corr_and_pvals(X,Y)               
        X = check_type_of_colname(X)
        columns = X.columns
        
        # generate feature importance
        try:    
            if type(model) == lgb.basic.Booster:
                importance_split = model.feature_importance(importance_type='split')
                importance_gain = model.feature_importance(importance_type='gain')
                df = pd.DataFrame({'feature': columns,
                                   'importance_split': importance_split,
                                   'importance_gain': importance_gain})     
            else: 
                importance_score = model.feature_importances_
                df = pd.DataFrame({'feature': columns,
                                   'importance': importance_score})

            if (ds_phase == 'test') & (final == True):
                shared_platform_df = df.rename(columns={'importance_gain':'importance'})
                shared_platform_df = df[['feature', 'importance']]
                shared_platform_df = shared_platform_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
                shared_platform_df.head(20).to_csv('MLaaR_feature_importance.csv', index=False)
            importance = df
        except:
            print('Can\'t generating feature_importance. Type of model is ' + str(type(model)))
            importance = None               
    else:
        X_info = None
        corr = None
        columns = None
        memory_usage = None   
    
    cat_col = pd.DataFrame(X.dtypes[X.dtypes=='category']).index
    cat_ratio = {}
    for col in cat_col:
        cat_ratio[col] = X[col].value_counts(normalize=True, ascending=False).to_dict()
    
    # generate report material
    data = {
        "name": name,
        "X_col": columns,
        "Y": Y,
        "X_info": X_info,
        "memory_usage": memory_usage,
        "execution_time": execution_time,
        "corr": corr,
        "predict": Y_pred,
        "model": model,
        "ds_phase": ds_phase,
        "evaluation_group": G,
        "params": params,
        "learning_type": learning_type,
        "evals_result": evals_result, 
        "importance": importance,
        "cat_ratio": cat_ratio
    }
    
    return data

def generate_report(Y_train, Y_train_pred, Y_test, Y_test_pred, name, learning_type, params=None, threshold=None, X_train=None, model=None, X_test=None, G_train=None, G_test=None, evals_result=None, train_execution_time=-1, test_execution_time=-1, file_name=None, output_format='html', memo='', final=False,  single_feature_num=0):

    """generate report
    Args:
        Y_train (pd.series):int, 0(N) or 1(P)
        Y_train_pred (pd.series): float, predict_prob
        Y_test (pd.series): int, 0(N) or 1(P)
        Y_test_pred (pd.series): float, predict_prob
        name (str): project name
        learning_type (str): 'classifier' or 'regressor'
        params (dict): model params
        threshold (float): 0~1
        X_train (pd.Dataframe): train x1~xn
        model (*): supported RF, catboost, lightgbm, xgboost
        X_test (pd.Dataframe): test x1~xn
        G_train (pd.series): str (default: None)
        G_test (pd.series): str (default: None)
        evals_result (dict): learning curve material
        train_execution_time (int/float): sec. (default:-1)
        test_execution_time (int/float): sec. (default:-1)
        file_name (str): output path(default: ${name}_${now_date}.pdf)
    Returns:
        test_data: test material
        train_data: train material
    """
    #pre modify user params
    if params is not None:
        params = pd.DataFrame.from_dict(params, orient = 'index')  
        params = modify_param_name(params, model=model)
        params = params.to_dict()
        params = params[0]
    
    learning_type_list = ['classifier', 'regressor']
    assert learning_type in learning_type_list, "learning_type must in {}".format(learning_type_list)
    print("產製報表材料...")
    if(type(Y_train) == pd.core.frame.DataFrame):
        Y_train = Y_train.iloc[:,0]
    if(type(Y_test) == pd.core.frame.DataFrame):
        Y_test = Y_test.iloc[:,0]   
    test_data = generate_report_material(X_test, Y_test, G_test, Y_test_pred, name, model, test_execution_time, ds_phase='test', params=params, learning_type=learning_type, evals_result=evals_result, final=final)
    train_data = generate_report_material(X_train, Y_train, G_train, Y_train_pred, name, model, train_execution_time, ds_phase='train', params=params, learning_type=learning_type, evals_result=evals_result, final=final)  

    if (single_feature_num > 0) & (type(model) == lgb.basic.Booster):
        test_data = single_feature_performance(test_data, single_feature_num, X_train, Y_train, X_test, Y_test, params)
    
    to_html.generate_html(test_data, train_data, threshold=threshold, name=name, file_name=file_name, learning_type=learning_type, memo=memo, output_format=output_format, params=params, model=model, single_feature_num=single_feature_num)
    
    print("結束報表產製")
    return test_data, train_data

def category_to_numeric(df):
    
    """ category to numeric
    Args:
        df (pd.Dataframe): df with category columns
    Returns:
        df (pd.Dataframe):
    """
    
    cat_cols =  df.select_dtypes('category').columns
    for col in cat_cols:
        df[col] = df[col].cat.codes
    return df

def auto_generate_report(y_label, project_name, df=None, df_train=None, df_valid=None, df_test=None, learning_type='classifier', test_size=.25, valid_size=.25, model_list=['RandomForestClassifier', 'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier'], output_format='html', memo='', final=False):

    """auto generate report
    Args:
        df (pd.Dataframe): 
        y_label (str): ground truth column name
        project_name (str): project name
        learning_type (str): 'classifier'
        test_size (float): test size of df (default: 0.25)
        valid_size (float): valid size of df_train (default: 0.25)
        model_list (list): (default: ['RandomForestClassifier' , 'LGBMClassifier','XGBClassifier', 'CatBoostClassifier'])
    Returns:
        model: best model in model_list
        data: test material
        train_data: train material
    """
    
    if df is not None:
        df = check_type_of_colname(df)
        df_train_raw, df_test = train_test_split(df, test_size=test_size, random_state=13)   
        df_train, df_valid = train_test_split(df_train_raw, test_size=valid_size, random_state=13)
    else:
        df_train_raw = pd.concat([df_train, df_valid])
        df_train_raw = check_type_of_colname(df_train_raw)
    best_predictor = esun_automl.auto_select_predictor_by_default_param(df_train, df_valid, y_label, model_list)

    model = best_predictor.trained_final_model.get_params()['model']
    Y_train_pred = best_predictor.predict_proba(category_to_numeric(df_train_raw))[:,1]
    Y_test_pred = best_predictor.predict_proba(category_to_numeric(df_test))[:,1]       
    test_data, train_data = generate_report(X_train=df_train_raw.drop(columns=[y_label]),
                                            Y_train=df_train_raw[y_label],
                                            Y_train_pred=Y_train_pred,
                                            model=model,
                                            X_test=df_test.drop(columns=[y_label]),
                                            Y_test=df_test[y_label],
                                            learning_type=learning_type,
                                            Y_test_pred=Y_test_pred,
                                            name=project_name,
                                            output_format=output_format,
                                            memo=memo,
                                            final=final)
    return model, test_data, train_data

def auto_generate_report_by_lgb(y_label, project_name, g_label=None, df=None, df_train=None, df_valid=None, df_test=None, test_size=0.25, valid_size=0.25, n_estimators=100, early_stopping_rounds=5, random_seed=13, output_format='html', memo='', final=False, cat_feature=None, **kwargs):

    """auto generate report by bayes_parameter_opt_lgb
    Args:
        df (pd.Dataframe): 
        y_label (str): ground truth column name
        project_name (str): project name
        learning_type (str): 'classifier'
        test_size (float): test size of df (default: 0.25)
        valid_size (float): valid size of df_train (default: 0.25)
        n_estimators (int): lgb params (default: 100)
        early_stopping_rounds (int): lgb params (default: 5) 
    Returns:
        opt_params: best params of bayes_parameter_opt_lgb
        model: lgb model
        test_data: test material
        train_data: train material
    """
        
    if df is not None:
        df = check_type_of_colname(df)  
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_seed)
        X_train, X_valid, Y_train, Y_valid = train_test_split(df_train.drop(columns=[y_label]), df_train[y_label], test_size=valid_size, random_state=random_seed)

    else:
        df_train = check_type_of_colname(df_train)
        df_valid = check_type_of_colname(df_valid)
        check_categorical_dtype(df_train, df_valid)
        check_categorical_dtype(df_valid, df_test)
        X_train = df_train.drop(columns=[y_label])
        Y_train = df_train[y_label]
        X_valid = df_valid.drop(columns=[y_label])
        Y_valid = df_valid[y_label]
        df_train = pd.concat([df_train, df_valid])
    
    X_test = df_test.drop(columns=[y_label])
    Y_test = df_test[y_label]
    
    if g_label is not None:
        G_train = pd.concat([X_train[g_label], X_valid[g_label]])
        G_test = X_test[g_label]
        X_train = X_train.drop(columns=[g_label])
        X_valid = X_valid.drop(columns=[g_label])
        df_train = df_train.drop(columns=[g_label])
        X_test = X_test.drop(columns=[g_label])
    else:
        G_train = None
        G_test = None

    print('============= auto-lgbm  =============')
    opt_params = esun_automl.bayes_parameter_opt_lgb(X=df_train.drop(columns=[y_label]), y=df_train[y_label], random_seed=random_seed, n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)
    opt_params['num_leaves'] = int(round(opt_params['num_leaves']))
    opt_params['max_depth'] = int(round(opt_params['max_depth']))
    params = {'num_leaves':opt_params['num_leaves'], 
              'max_depth':opt_params['max_depth'], 
              'reg_alpha':opt_params['reg_alpha'], 
              'reg_lambda':opt_params['reg_lambda'], 
              'learning_rate':opt_params['learning_rate'],
              'n_estimators':n_estimators,
              'metric':'auc'
             }
    print('============= training =============')
    
    X_valid_dataset = lgb.Dataset(X_valid, Y_valid, categorical_feature=cat_feature)
    X_train_dataset = lgb.Dataset(X_train, Y_train, categorical_feature=cat_feature)

    model = lgb.train(params=params,
                      train_set=X_train_dataset,
                      valid_sets=[X_valid_dataset, X_train_dataset],
                      valid_names=['eval', 'train'],
                      early_stopping_rounds=early_stopping_rounds,
                      )
    Y_train_pred = model.predict(df_train.drop(columns=[y_label]))
    Y_test_pred = model.predict(X_test)
    test_data, train_data = generate_report(X_train=df_train.drop(columns=[y_label]),
                                            Y_train=df_train[y_label],
                                            Y_train_pred=Y_train_pred,
                                            G_train=G_train,
                                            model=model,
                                            learning_type='classifier',
                                            X_test=X_test,
                                            Y_test=Y_test,
                                            Y_test_pred=Y_test_pred,
                                            G_test=G_test,
                                            params=params,
                                            name=project_name,
                                            output_format=output_format,
                                            memo=memo,
                                            final=final
                                            )
    return opt_params, model, test_data, train_data

def get_ks_info(Y_test, Y_pred, bins=np.linspace(.0, 1.0, num=101, endpoint=True)):
    
    """get ks and ks_inx info
    Args:
        Y_test (pd.series): int, 0(N) or 1(P)
        Y_test_pred (pd.series): float, predict_prob
        bins (narray): 0 to 1 step by 0.1
    Returns:
        ks: best threshold
        ks_idx: best threshold's index
    """
    
    data = {'Y': Y_test, 'Y_hat': Y_pred}
    df = pd.DataFrame(data)
    df['group'] = pd.cut(df['Y_hat'], bins=bins, right=False)
    df3 = pd.pivot_table(df.sort_values(by='group'), values=['group'], index='group', columns='Y', aggfunc=len, fill_value=0)
    df4 = np.cumsum(df3)
    df5 = df4 / df4.max(axis=0)
    df6 = 1 - df5
    ks = np.max(abs(df6.iloc[:, 0] - df6.iloc[:, 1]))
    ind = abs(df6.iloc[:, 0] - df6.iloc[:, 1]).values.argmax()
    ks_idx = bins[ind+1]
    return ks, ks_idx

def get_perform_index(row):
    
    """get performance index
    Args:
        row (dict): tp, fp, fn, tn, count
    Returns:
        row (dict): precision, recall, acc, f1
    """

    row['precision'] = row['tp'] / (row['tp'] + row['fp'])
    row['recall'] = row['tp'] / (row['tp'] + row['fn'])
    row['acc'] = (row['tp'] + row['tn']) / row['count']
    row['f1'] = 2 * row['precision']*row['recall'] / (row['recall'] + row['precision'])
    return row

def modify_param_name(data, model):
    
    for i in range(0,len(list(data.index))):
        if not (data.index[i] in lightgbm_param.keys()):
            if 'lightgbm' in str(type(model)):
                correct_flag = 0
            else:
                correct_flag = 1
            for key, value in lightgbm_param.items():
                if data.index[i] in lightgbm_param[key]:
                    data = data.rename(index = {data.index[i]: key})
                    correct_flag = 1
                    break
            if correct_flag == 0:
                data.loc[data.index[i]] = data.loc[data.index[i]] + '*'
    return data

def show_model_params(data):
   
    """show model params and return model name and params diff df
    Args:
        data (dict): report material
        name (str): project name
    Returns:
        model_param_df (pd.Dataframe): params diff
        model_param_info (str): model name
    """
    
    model_info = []
    model_param_info = ''
    if 'lightgbm' in str(type(data['model'])):
        model_info.append(pd.Series(LGBMClassifier().get_params(),name='Default'))
        model_info.append(pd.Series(data['params'],name='Current'))
        model_param_info = 'lightgbm'
        #check parameter name in default dictionary
        model_info[0] = model_info[0].astype('str')
        model_info[1] = model_info[1].astype('str')
        model_info[0] = modify_param_name(model_info[0], model=data['model'])#default params modify
    else:
        model_info.append(pd.Series(type(data['model'])().get_params(),name='Default'))
        model_info.append(pd.Series(data['model'].get_params(),name='Current'))
        #capture model name from model type
        tmp = str(type(data['model'])).replace("<class '","")
        tmp = tmp.replace("'>","")
        model_param_info = tmp.split('.')[-1]
        #exception handling
        if model_param_info == 'Booster':
            model_param_info = 'LGBMClassifier'
            
    if model_param_info == '':
        try:
            model_param_info = type(data['model'])
        except:
            model_param_info = 'not supported model'
        model_param_df = pd.DataFrame(['not supported model'],columns=['not supported model'])
    else:       
        model_param_df = pd.concat(model_info, axis=1).fillna("-")
        
    return model_param_df, model_param_info
    
def float_to_display(in_f, n=3):
    
    """ float to substring
    Args:
        in_f (float): input float
    Returns:
        display_str (str): substring of input 
    """
    e_str = '%E' % (in_f)
    r_int = int(e_str.split('E')[1])
    if r_int<0:
        display_str = '{{0:.{}f}}'.format(abs(r_int+1) + n)
    else:
        display_str = '{{0:.{}f}}'.format(n)
    return display_str.format(in_f)

def create_connection(db_file):
    
    """create sqlite connection
    Args:
        db_file (str): sqlite file path
    Returns:
        None
    """
    
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        return conn
    except:
        print(sys.exc_info())
    return None

def insert_df_to_sqlite(df, db_name, tbl_name):
    
    """insert df to sqlite
    Args:
        df (pd.Dataframe): insert value
        db_name (str): sqlite file path
        tbl_name (str): table name of sqlite
    Returns:
        None
    """
    
    conn = create_connection(db_name)
    wild_cards = ','.join(['?']* len(df.columns))
    data = [tuple(i.to_pydatetime() if type(i) == pd._libs.tslib.Timestamp else i for i in x) for x in df.values]
    with conn:
        cur = conn.cursor()
        cur.executemany("INSERT INTO %s values(%s)" % (tbl_name, wild_cards), data)
        
def add_eval_to_history_classifier(df, ds_phase):
    df['ts'] = datetime.datetime.now()
    df['ds_phase'] = ds_phase
    df = df[['acc', 'auc', 'count', 'ds_phase', 'execution_time', 'f1', 'fn', 'fp', 'ks', 'ks_idx',\
             'name', 'precision', 'recall', 'threshold', 'tn', 'tp', 'ts', 'count_of_col', 'memo']]   
    try:
        insert_df_to_sqlite(df, os.path.join(os.environ['HOME'],'.evaluation_history.db'), 'past_performance_classifier')
    except sqlite3.OperationalError:
        create_table_classifier(os.path.join(os.environ['HOME'], '.evaluation_history.db'))
        insert_df_to_sqlite(df, os.path.join(os.environ['HOME'],'.evaluation_history.db'), 'past_performance_classifier')
        
def add_eval_to_history_regressor(df, ds_phase):
    df['ts'] = datetime.datetime.now()
    df['ds_phase'] = ds_phase
    df = df[['MSE', 'RMSE', 'MAE', 'EVS', 'r2', 'MAPE', 'ds_phase', 'name', 'ts', 'count', 'count_of_col', 'memo']]
    try:
        insert_df_to_sqlite(df, os.path.join(os.environ['HOME'],'.evaluation_history.db'), 'past_performance_regressor')
    except sqlite3.OperationalError:
        create_table_regressor(os.path.join(os.environ['HOME'], '.evaluation_history.db'))
        insert_df_to_sqlite(df, os.path.join(os.environ['HOME'],'.evaluation_history.db'), 'past_performance_regressor')
        
def create_table_classifier(db_name):
    conn = create_connection(db_name)
    with conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE past_performance_classifier(acc REAL, auc REAL, count INT, ds_phase TEXT,
        execution_time REAL, f1 REAL, fn INT, fp INT, ks REAL, ks_idx REAL, name TEXT, 
        precision REAL, recall REAL, threshold REAL, tn INT, tp INT, ts datetime, count_of_col INT, memo TEXT)''')
        
def create_table_regressor(db_name):
    conn = create_connection(db_name)
    with conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE past_performance_regressor(MSE REAL, RMSE REAL, MAE REAL, EVS REAL, r2 REAL, MAPE REAL, ds_phase TEXT, name TEXT, ts datetime, count INT, count_of_col INT, memo TEXT)''')
        
def eval_classifier(evaluation_data, name, threshold, memo, verbose=False):
    rows = []
    classifier = np.vectorize(
        lambda x, threshold: 1 if x >= threshold else 0)
    data = evaluation_data  
    row = {}
    row['ks'], row['ks_idx'] = get_ks_info(
        data['Y'], data['predict'])
    
    row['name'] = name
    row['threshold'] = threshold
    row['execution_time'] = data['execution_time']

    # calculate AUC, fp, fn, tp, tn, ...etc on validation data
    row['fpr'], row['tpr'], _ = roc_curve(data['Y'], data['predict'])
    row['auc'] = auc(row['fpr'], row['tpr'])

    predictions = classifier(data['predict'], threshold)
    cfm = confusion_matrix(data['Y'], predictions)
    row['tn'], row['fp'], row['fn'], row['tp'] = cfm.ravel()
    row['count'] = np.sum(
        [row['tn'], row['fp'], row['fn'], row['tp']])
    row['count_of_col'] = len(data['X_col'])
    row['memo'] = memo

    row = get_perform_index(row)
    rows.append(pd.Series(row))
    output = pd.DataFrame(rows)
    columns = ['name', 'execution_time', 'count', 'acc', 'precision', 'recall', 'f1', 'auc', 'fn', 'fp', 'tn', 'tp', 'ks', 'ks_idx']

    evaluation_result = output.drop(columns=['tpr','fpr'])
    add_eval_to_history_classifier(evaluation_result.copy(), data['ds_phase'])
    return evaluation_result

def eval_regressor(data, name, memo, verbose=False):
    y_true = data['Y']
    y_pred = data['predict']
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    mape = sum(abs((y_true - y_pred))/y_true)/len(data)
    rows = pd.Series({
        'name': name,
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true=y_true, y_pred=y_pred),
        'EVS': explained_variance_score(y_true=y_true, y_pred=y_pred),
        'r2': r2_score(y_true=y_true, y_pred=y_pred),
        'MAPE': mape,
        'count': len(data['Y']),
        'count_of_col': len(data['X_col']),
        'memo': memo
    })

    evaluation_result = pd.DataFrame([rows])
    add_eval_to_history_regressor(evaluation_result.copy(), data['ds_phase'])
    return evaluation_result

def gen_validation_AUC(data, name, threshold, memo):
    fpr, tpr, thresholds = roc_curve(data['Y'], data['predict'])
    evaluation_result = eval_classifier(data, name, threshold=threshold, memo=memo)
    return fpr, tpr, threshold, evaluation_result

def plot_validation_AUC(data, name, ax, alpha, threshold, splits, memo):
    axs = ax
    fpr, tpr, threshold, evaluation_result = gen_validation_AUC(data, name, threshold, memo)    
    val_auc = auc(fpr, tpr)
    tprs = interp(splits, fpr, tpr)
    axs.plot(fpr, tpr, lw=2, alpha=alpha, label='%s (AUC = %0.2f)' % (data['ds_phase'],val_auc))
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.grid(True)  
    axs.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axs.yaxis.set_ticks(np.arange(0, 1.1, 0.1))   
    axs.set_xlabel('False Positive Rate', size=fontsize)
    axs.set_ylabel('True Positive Rate', size=fontsize)
    axs.set_title("TPR vs. FPR", size=fontsize)
    axs.legend(loc="best")
    return evaluation_result

def gen_precision_recall(data, alpha, initial_threshold, end_threshold, step_threshold, single_feature=None):
    # plot recall precision
    classifier = np.vectorize(lambda x, threshold: 1 if x >= threshold else 0)
    precisions = []
    recalls = []
    ts = []
    ll = np.arange(initial_threshold, end_threshold, step_threshold)
    for t in ll:
        if single_feature is not None:
            prediction = classifier(data['single_feature_pred'][single_feature], t)
        else:
            prediction = classifier(data['predict'], t)
        # 怕有小數點的問題 原本為if np.sum(prediction) == 0:
        if np.sum(prediction) <= 0.01:
            break
        recall = recall_score(data['Y'], prediction)
        ts.append(t)
        precisions.append(precision_score(data['Y'], prediction))
        recalls.append(recall)
    return precisions, recalls, ts

def plot_precision_recall(data, ax, alpha, initial_threshold, end_threshold, step_threshold):
    axs = ax  
    precisions, recalls, ts = gen_precision_recall(data=data, alpha=alpha, initial_threshold=initial_threshold, end_threshold=end_threshold, step_threshold=step_threshold)
    
    axs.plot(recalls, precisions, label='%s' % (data['ds_phase']))
    axs.set_xlabel('Recall',size=fontsize)
    axs.set_ylabel('Precision',size=fontsize)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.grid(True)
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axs.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axs.set_title("Precision vs. Recall", size=fontsize)
    return precisions, recalls, ts

def plot_f1(ds_phase, precisions, recalls, ts, alpha, ax):
    axs = ax
    pre = np.array(precisions)
    rec = np.array(recalls)
    f1 = (2*pre*rec) / (pre+rec)

    axs.plot(ts, f1,label='%s' % (ds_phase))
    axs.set_xlabel('Threshold', size=fontsize)
    axs.grid(True)
    axs.set_ylabel('F1', size=fontsize)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axs.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axs.set_title('F1 vs. Threshold', size=fontsize)
    axs.legend(loc='best')

def gen_model_perform_classifier(train_evaluation_result, test_evaluation_result, test_cols):
    model_name = test_evaluation_result['name'].iloc[0]
    evaluation_result = pd.concat([train_evaluation_result,test_evaluation_result])
    evaluation_result['tpr'] = evaluation_result.recall
    evaluation_result['tnr'] = evaluation_result.tn/(evaluation_result.tn+evaluation_result.fp)
    evaluation_result['fnr'] = 1-evaluation_result.tpr
    evaluation_result['fpr'] = 1-evaluation_result.tnr
    evaluation_result = np.round(evaluation_result.drop(['name'],axis=1), 3)
    evaluation_result.columns = [c.upper() for c in evaluation_result.columns]
    evaluation_result = evaluation_result[['ACC', 'AUC', 'F1', 'PRECISION', 'RECALL', 'TNR', 'FNR', 'FPR', 'THRESHOLD']]
    evaluation_result.columns = ['ACC', 'AUC', 'F1', 'Precision', 'Recall', 'TNR', 'FNR', 'FPR', 'Threshold']
    evaluation_result.index = ['Train', 'Test']
    return evaluation_result

def gen_model_perform_regressor(train_evaluation_result, test_evaluation_result, test_cols):
    model_name = test_evaluation_result['name'].iloc[0]
    evaluation_result = pd.concat([train_evaluation_result,test_evaluation_result])
    evaluation_result['cols'] = test_cols
    evaluation_result = np.round(evaluation_result.drop(['name'],axis=1), 3)
    evaluation_result['cols'] = evaluation_result['cols'].astype(str)
    evaluation_result = evaluation_result[['MSE', 'RMSE', 'MAE', 'EVS', 'r2', 'MAPE']]
    evaluation_result.columns = ['MSE', 'RMSE', 'MAE', 'EVS', 'r2', 'MAPE']
    evaluation_result.index = ['Train', 'Test']
    return evaluation_result
   
def gen_confusion_matrix(data, threshold):
    cm = pd.DataFrame(confusion_matrix(data['Y'], data['predict']>threshold), columns=['Predicted 0','Predicted 1'], index=['Actual 0','Actual 1'])
    cm = cm/len(data['Y'])
    return cm

def best_threshold(precision, recall, thresholds):
    precision = np.array(precision)
    recall = np.array(recall)
    thresholds = np.array(thresholds)
    f1 = 2*precision*recall/(precision+recall)
    t = thresholds[np.nanargmax(f1)]
    return t

def query_data(db_name, sql_cmd):
    conn = create_connection(db_name)
    try:
        df = pd.read_sql_query(sql_cmd, conn)
        return df
    except:
        print(sys.exc_info())
    finally:
        conn.close()

def gen_hist_data(ds_phase, learning_type, name, db_name=os.path.join(os.environ['HOME'],'.evaluation_history.db')):   
    df = query_data(db_name,"select * from past_performance_" + learning_type + " where ds_phase='{}' and name='{}' order by ts desc limit 20".format(ds_phase, name))
    
    df['ts'] = pd.to_datetime(df.ts).dt.strftime('%Y/%m/%d %H:%M')
    df = np.round(df, 3)
    df = df.sort_values('ts')
    return df

def check_type_of_colname(df):
    for i,col_name in enumerate(df.columns):
        df.columns.values[i] = str(col_name)
    return df

def check_categorical_dtype(df_left, df_right):
    result = []
    for i in df_left.select_dtypes('category').columns:
        if CategoricalDtype(df_left[i].cat.categories) != CategoricalDtype(df_right[i].cat.categories):
            result.append(i)
    if result:
        raise TypeError('TypeError in '+format(', '.join(result))+'.'+' The components of categorical data in training, validation and testing datasets must be the same!')
        
def single_feature_performance(test_data, single_feature_num, X_train, Y_train, X_test, Y_test, params):
    print('跑單因子成效...')
    # use only one feature to train model and predict
    single_feature_dic = {}
    single_feature_importance = test_data['importance'].sort_values('importance_gain',ascending=False).head(single_feature_num)
    for col in single_feature_importance['feature']:
        X_test_dataset = lgb.Dataset(X_test[[col]], Y_test)
        X_train_dataset = lgb.Dataset(X_train[[col]], Y_train)
        single_feature_params = params.copy()
        single_feature_params['min_data_in_bin'] = 1
        single_feature_params['min_data'] = 1
        single_feature_params['min_hessian'] = 0
        single_feature_params['max_bin'] = 1000
        single_feature_model = lgb.train(params=single_feature_params,
                                                train_set=X_train_dataset,
                                                valid_sets=[X_test_dataset, X_train_dataset],
                                                valid_names=['eval', 'train'],
                                                verbose_eval = False)
        y_test_pred = single_feature_model.predict(X_test[[col]])
        single_feature_dic[col] = y_test_pred
    test_data['single_feature_pred'] = single_feature_dic
    return test_data
