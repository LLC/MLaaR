import matplotlib.pylab as plt
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,precision_score
import pandas as pd
import lightgbm as lgb
from bayes_opt import BayesianOptimization
try:
    from mlaar import report
except:
    import report

def auto_select_predictor_by_default_param(df_train, df_valid, y_label,model_list=['RandomForestClassifier' , 'LGBMClassifier','XGBClassifier', 'CatBoostClassifier']):
    df_train = report.category_to_numeric(df_train)
    df_valid = report.category_to_numeric(df_valid)
    column_descriptions = {
        y_label: 'output'
    }
    for c in df_train.select_dtypes('category').columns:
        column_descriptions[c] = 'categorical'
    best_score = -1
    best_predictor = ''
    #model_list = ['RandomForestClassifier' , 'LGBMClassifier','XGBRegressor', 'CatBoostClassifier']
    #model_list = ['RandomForestClassifier']
    for model_name in model_list:
        ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

        #ml_predictor.train(pd.concat([X_train, Y_train],axis=1),model_names=model_name)
        ml_predictor.train(df_train,model_names=model_name)
        ml_predictor.score(df_valid, df_valid[y_label])
        print(model_name)
        try:
            Y_train_pred = ml_predictor.predict_proba(df_valid)[:,1]
        except(TypeError):
            print('============ {} predict_proba error ============ '.format(model_name))
            Y_train_pred = ml_predictor.predict(df_valid)
        cur_F1 = arg_max_F1(Y_train_pred, df_valid[y_label])
        if cur_F1 > best_score:
            #best_model = ml_predictor.trained_final_model.get_params()['model']
            best_predictor = ml_predictor
            best_score = cur_F1

    return best_predictor

def arg_max_F1(Y_train_pred, Y_train):
    y_prob = Y_train_pred
    y_test = Y_train
    tlist = []
    for i in range(100):
        thr = i/100
        y_pred =  y_prob>thr
        the_f1 = f1_score(y_pred,y_test)
        the_recall = recall_score(y_pred,y_test)
        the_precision = precision_score(y_pred,y_test)
        tlist.append({'F1':the_f1,'Recall':the_recall,'Precition':the_precision, 'threshold':thr})

    df = pd.DataFrame(tlist)
    ax = df.plot(x='threshold')
    max_F1 = df.F1.max()
    argmax_F1 = df.F1.argmax()
    ax.axvline(df.loc[argmax_F1]['threshold'],label='Best F1 {:.03f}, THR:{}'.format(max_F1,df.loc[argmax_F1]['threshold']))
    #ax.plot(np.array([df.loc[df.f1.argmax()]['threshold'] for i in range(len(df))]),np.linspace(0,1,100),label='efwf')
    ax.legend()
    #plt.savefig('tmp.png')
    plt.show()
    return max_F1


def bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=5, n_folds=2, random_seed=13, n_estimators=100, early_stopping_rounds=5,output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y)
    # parameters
    def lgb_eval(num_leaves, max_depth, reg_alpha, reg_lambda, learning_rate, bagging_fraction):
        params = {'application':'binary', 'early_stopping_round':early_stopping_rounds, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['learning_rate'] = learning_rate
        params['n_estimators'] = n_estimators
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, verbose_eval = 200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (20, 35),
                                            'max_depth': (5, 9),
                                            'reg_alpha': (0, 5),
                                            'reg_lambda': (0, 5),
                                            'learning_rate': (0.0001, 0.9999),
                                            'bagging_fraction': (0.8, 1),
                                           })
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round, acq='ei')
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    #return lgbBO
    print('----------------------------==')
    print(lgbBO.res['max']['max_params'])
    print('----------------------------==')
    return lgbBO.res['max']['max_params']

