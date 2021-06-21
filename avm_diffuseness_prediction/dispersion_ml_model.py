# encoding: utf-8
# @Author: zhaoqi
# @Date  : 9/23/20
# @Desc  :  
# @license : Copyright(C), Biomind
# @Contact : qi.zhao@biomind.ai

import os
import sys
import sklearn
import pandas as pd
import numpy as np
from numpy import sort
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, roc_auc_score,plot_roc_curve
import xgboost as xgb
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier



def get_train_data(feat_label_df_path, feat_label_df=None, return_df=False):
    if feat_label_df is None:
        feat_label_df = pd.read_csv(feat_label_df_path)
    cols = feat_label_df.columns
    feat_label_cols = ['filename', 'is_dispersion_doc']
    feats = [one for one in feat_label_df.columns if one not in feat_label_cols]
    feat_label_df =  feat_label_df.dropna(axis=0,how='all')
    feat_label_df =  feat_label_df.dropna(axis=1,how='all')
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    feat_label_df[feats] = feat_label_df[feats].astype('float')
    feat_label_df[feats] = feat_label_df[feats].apply(max_min_scaler)
    feat_label_df = feat_label_df.sample(frac=1)
    X = feat_label_df[feats].values
    y0 = feat_label_df[feat_label_cols]
    y = feat_label_df['is_dispersion_doc'].values
    if return_df:
        return feat_label_df, feats, 'is_dispersion_doc'
    return X, y


def draw_roc(mean_tpr_fpr_list, save_path=None):
    fig, ax = plt.subplots()
    for item in seg_mean_tpr_fpr_list:
        model_name = item['model_name']
        mean_fpr, mean_tpr, mean_auc, std_auc = item['items']
        ax.plot(mean_fpr, mean_tpr, label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (model_name, mean_auc, std_auc), lw=2, alpha=0.8)
    cl_x = range(-2, 2)
    cl_y = range(-2, 2)
    ax.plot(cl_x, cl_y, color='gray', alpha=0.5, linestyle='--', linewidth=0.7)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Mean ROC Curve(10-fold cross validation)')
    ax.legend(loc='lower right')
    plt.ylabel('True Positive')
    plt.xlabel('False Positive')
    if save_path is not None:
#         plt.legend(prop={'family': 'Times New Roman', 'size':1})
        plt.savefig(save_path, dpi=300)
    plt.show()


def evaluation(y_true, y_pred_proba, model_name, data_set_name, threshold=0.5):
    if not isinstance(y_true, list):
        y_true = y_true.tolist()
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    dic = {'model_name':model_name, 'acc':acc, 'recall':recall, 'precision':precision, 'auc':auc_score}
    return dic


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def cross_val_model(clf, X, y, model_name, n_splits=10,  threshold=0.5):

    cv = KFold(n_splits=n_splits, shuffle=True)
    recall_list = []
    precision_list = []
    auc_list = []
    acc_list = []
    opt_thre_list = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    opt_thres = []
    for i, (train, test) in enumerate(cv.split(X)):

        clf.fit(X[train], y[train])
        pred_score = clf.predict_proba(X[test])
        y_pred = np.array(pred_score)[:,1]
#         pred_score = clf.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], y_pred, pos_label=1)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        opt_thre, points = Find_Optimal_Cutoff(tpr, fpr, thresholds)
#         opt_thres.append(opt_thre)
        roc_auc = auc(fpr, tpr)
        y_pred_bi = np.where(y_pred > opt_thre, 1, 0)
        test_eval_dict = evaluation(y[test], y_pred_bi, model_name, 'test')
        recall_list.append(test_eval_dict['recall'])
        precision_list.append(test_eval_dict['precision'])
        acc_list.append(test_eval_dict['acc'])
        auc_list.append(roc_auc)
        opt_thre_list.append(opt_thre)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_recall = np.mean(np.array(recall_list))
    mean_precision = np.mean(np.array(precision_list))
    mean_auc = np.mean(np.array(auc_list))
    mean_acc = np.mean(np.array(acc_list))
    std_auc = np.std(np.array(auc_list))

    return mean_recall, mean_precision, mean_auc, mean_acc, (mean_fpr, mean_tpr,mean_auc, std_auc)


def build_xgb(X, y):
#     X, y = get_train_data(feat_label_df_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    print('dispersion ratio: ', sum(y_test) / len(y_test))
    xgb1 = XGBClassifier(learning_rate =0.1, 
                         n_estimators=20, 
                         max_depth=5,
                         min_child_weight=1, 
                         gamma=0, 
                         subsample=0.8, 
                         colsample_bytree=0.8,
                         objective= 'binary:logistic', 
                         nthread=4, 
                         scale_pos_weight=1, 
                         seed=27)
    xgb1.fit(X_train, y_train)
    y_test_pred = xgb1.predict_proba(X_test)[:, 1]
    y_test_pred = np.array(y_test_pred)
    opt_threshold = 0.33
    y_pred_bi = np.where(y_test_pred > opt_threshold, 1, 0)

    y_train_pred = xgb1.predict_proba(X_train)[:, 1]
    y_train_pred = np.array(y_train_pred)
    y_train_bi = np.where(y_train_pred > opt_threshold, 1, 0)
    test_eval_dict = evaluation(y_test, y_pred_bi, 'xgboost', 'test')
    train_eval_dict = evaluation(y_train, y_train_bi, 'xgboost', 'train')
    print('----------------train-----------------')
    for key in train_eval_dict:
        print(key, train_eval_dict[key])
    print('----------------test------------------')
    for key in test_eval_dict:
        print(key, test_eval_dict[key])
    feat_importants = xgb1.feature_importances_
    feat_importants_dic = {}
    for idx, importance in enumerate(feat_importants):
        feat_importants_dic[idx] = importance
    importance_res = sorted(feat_importants_dic.items(), key=lambda x:x[1], reverse=True)

    return xgb1, importance_res

    
def adjust_xgb(feat_label_df_path):
    X, y = get_train_data(feat_label_df_path)
    XGBClassifier(learning_rate=0.1, n_estimators=50)
#     param_test2 = {
#      'max_depth':list(range(3,7,1)),
#      'min_child_weight':list(range(1,2,1))
#     }
    param_test3 = {
        'gamma':[i/10.0 for i in range(0, 5)]
    }
    param_test4 = {'top_k': [i for i in range(0, 100, 20)]}
    xgb1 = XGBClassifier(learning_rate =0.1, 
                         n_estimators=20, 
                         max_depth=5,
                         min_child_weight=1, 
                         gamma=0, 
                         subsample=0.8, 
                         colsample_bytree=0.8,
                         objective= 'binary:logistic', 
                         nthread=4, 
                         scale_pos_weight=1, 
                         
                         seed=27)
    
    gsearch1 = GridSearchCV(estimator=xgb1, param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=10)
    gsearch1.fit(X, y)
    print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    

def compare_model(X, y, n_splits=10, threshold=0.5):
    mean_fpr_tpr_list = []
    xgb = XGBClassifier(learning_rate =0.1, 
                         n_estimators=70, 
                         max_depth=5,
                         min_child_weight=1, 
                         gamma=0, 
                         subsample=0.8, 
                         colsample_bytree=0.8,
                         objective= 'binary:logistic', 
                         nthread=4, 
                         scale_pos_weight=1, 
                         seed=27)
    
    lr = LogisticRegression(C=1)
    
    svm_model = svm.SVC(kernel='linear', C=1, probability=True)
    
    dt_model = DecisionTreeClassifier(max_depth=15, min_samples_split=10)
    
    knn_model = KNeighborsClassifier(n_neighbors=3)
    
    rf_model = RandomForestClassifier(n_estimators=10)

    adb_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)

    gbdt_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=2, min_samples_split=8)
    ensemble_model = VotingClassifier(estimators=[('lr', lr), ('rf', rf_model), ('svm', svm_model), ('adb', adb_model), ('gbdt', gbdt_model)], voting='soft')

    model_list = [('LR', lr), ('SVM', svm_model), ('Decision Tree', dt_model), ('KNN', knn_model), ('Random Forest', rf_model), ('Adaboost', adb_model), ('GBDT', gbdt_model), ('Xgboost', xgb), ('Ensemble', ensemble_model)]
    
    all_eval_list = []
    for item in model_list:
        model_name, model = item
        mean_recall, mean_precision, mean_auc, mean_acc, item = cross_val_model(model, X, y, model_name, n_splits=n_splits,threshold=threshold)
        mean_fpr_tpr_list.append({'model_name':model_name,'items':item})
        eval_dict = {'model_name':model_name, 'acc':mean_acc, 'recall':mean_recall, 'precision':mean_precision, 'auc':mean_auc}
        all_eval_list.append(eval_dict)
    
    eval_df = pd.DataFrame(all_eval_list)
    return eval_df, mean_fpr_tpr_list
    
    
def compare_model_train(X_train, y_train, X_test, y_test):

    # print('dispersion ratio: ', sum(y_test) / len(y_test))
    
    xgb = XGBClassifier(learning_rate =0.1, 
                         n_estimators=50, 
                         max_depth=5,
                         min_child_weight=1, 
                         gamma=0, 
                         subsample=0.8, 
                         colsample_bytree=0.8,
                         objective= 'binary:logistic', 
                         nthread=4, 
                         scale_pos_weight=1, 
                         seed=27)
    
    lr = LogisticRegression(C=2)
    
    svm_model = svm.SVC(kernel='linear', C=1, probability=True)
    
    dt_model = DecisionTreeClassifier(max_depth=15, min_samples_split=10)
    
    knn_model = KNeighborsClassifier(n_neighbors=3)
    
    rf_model = RandomForestClassifier(n_estimators=10)

    adb_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)

    gbdt_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=2, min_samples_split=8)

    model_list = [('lr', lr), ('svm', svm_model), ('decision tree', dt_model), ('knn', knn_model), ('random forest', rf_model), ('adaboost', adb_model), ('gbdt', gbdt_model), ('xgb', xgb)]
    
    all_eval_list = []
    for item in model_list:
        model_name, model = item
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        eval_dict = evaluation(y_test, y_pred_proba, model_name, 'test')
        all_eval_list.append(eval_dict)
    
    eval_df = pd.DataFrame(all_eval_list)
    return eval_df

    
def predict(model, test_data, threshold):
    model.predict(model)

    
    