import pandas as pd
import numpy as np

def feature_engineering(data_dic, apply_on='train'):

    if apply_on=='train':
        data = data_dic['train'].copy()
    elif apply_on=='test':
        data = data_dic['test'].copy()

    bureau = data_dic['bureau'].copy()
    bureau_balance = data_dic['bureau_balance'].copy()
    credit_card_balance = data_dic['credit_card_balance'].copy()
    installments_payments = data_dic['installments_payments'].copy()
    previous_application = data_dic['previous_application'].copy()
    POS_CASH_balance = data_dic['POS_CASH_balance'].copy()
    # BUREAU DATA
    print ('Extracting Features from bureau data ...')
    new_data = process_bureau_data_(data, bureau)
    new_feats_bureau = [x for x in new_data.columns if x not in data.columns]
    print ('Added the {} following features to the main dataset:'.format(len(new_feats_bureau)), new_feats_bureau)

    return new_data

def process_bureau_data_(df, df_bureau):
    df_bureau = df_bureau.copy()
    df = df.copy()
    # CREDIT_ACTIVE
    df_bureau['active_cb'] = df_bureau['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Active' else 0)
    df_bureau['closed_cb'] = df_bureau['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Closed' else 0)
    df_bureau['sold_cb'] = df_bureau['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Sold' else 0)
    df_bureau['bad_debt_cb'] = df_bureau['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Bad debt' else 0)

    active_credit_bureau = pd.DataFrame(df_bureau.groupby('SK_ID_CURR').sum()['active_cb']).reset_index()
    closed_credit_bureau = pd.DataFrame(df_bureau.groupby('SK_ID_CURR').sum()['closed_cb']).reset_index()
    sold_credit_bureau = pd.DataFrame(df_bureau.groupby('SK_ID_CURR').sum()['sold_cb']).reset_index()
    bad_debt_credit_bureau = pd.DataFrame(df_bureau.groupby('SK_ID_CURR').sum()['bad_debt_cb']).reset_index()

    df = df.merge(active_credit_bureau, how='left', on='SK_ID_CURR')
    df = df.merge(closed_credit_bureau, how='left', on='SK_ID_CURR')
    df = df.merge(sold_credit_bureau, how='left', on='SK_ID_CURR')
    df = df.merge(bad_debt_credit_bureau, how='left', on='SK_ID_CURR')

    df['tot_loan_cb'] = (df['active_cb']+df['closed_cb']+
                                    df['sold_cb']+df['bad_debt_cb'])
    # CREDIT_CURRENCY
    num_currency_used_cb = pd.DataFrame(df_bureau.groupby(['SK_ID_CURR',
                'CREDIT_CURRENCY']).count().reset_index()['SK_ID_CURR'].value_counts()).reset_index()
    num_currency_used_cb.columns = ['SK_ID_CURR', 'num_currency_used_cb']
    df = df.merge(num_currency_used_cb, how='left', on='SK_ID_CURR')
    # DAYS_CREDIT
    x=pd.DataFrame(df_bureau.groupby('SK_ID_CURR').agg({'DAYS_CREDIT':['max', 'min', 'mean']}))
    x.columns = x.columns.droplevel()
    x = x.reset_index()
    x.columns = ['SK_ID_CURR', 'DAYS_CREDIT_min_cb', 'DAYS_CREDIT_max_cb', 'DAYS_CREDIT_mean_cb']
    df = df.merge(x, how='left', on='SK_ID_CURR')
    # CREDIT_DAY_OVERDUE
    ### Descriptive Stats
    x=pd.DataFrame(df_bureau.groupby('SK_ID_CURR').agg({'CREDIT_DAY_OVERDUE':['max', 'min', 'mean', 'sum']}))
    x.columns = x.columns.droplevel()
    x = x.reset_index()
    x.columns = ['SK_ID_CURR', 'CREDIT_DAY_OVERDUE_max_cb', 
                 'CREDIT_DAY_OVERDUE_min_cb', 'CREDIT_DAY_OVERDUE_mean_cb', 'CREDIT_DAY_OVERDUE_sum_cb']
    df = df.merge(x, how='left', on='SK_ID_CURR')
    ### Number of Credit with overdue
    x=df_bureau.groupby('SK_ID_CURR').agg({'CREDIT_DAY_OVERDUE_flag':'sum'}).reset_index()
    x.columns = ['SK_ID_CURR', 'num_credit_with_overdue_cb']
    df = df.merge(x, how='left', on='SK_ID_CURR')
    
    
    return df



'''
################################# Script
print ('Loading Data ...')
data_dic={'train':pd.read_csv("../data/application_train.csv"),
          'test':pd.read_csv("../data/application_test.csv"),
          'bureau':pd.read_csv("../data/bureau.csv"),
          'bureau_balance':pd.read_csv("../data/bureau_balance.csv"),
          'credit_card_balance':pd.read_csv("../data/credit_card_balance.csv"),
          'installments_payments':pd.read_csv("../data/installments_payments.csv"),
          'previous_application':pd.read_csv("../data/previous_application.csv"),
          'POS_CASH_balance':pd.read_csv("../data/POS_CASH_balance.csv")}

print ('Feature Engineering')
print ('########### Bureau data')
new_train = feature_engineering(data_dic, apply_on='train')
new_test = feature_engineering(data_dic, apply_on='test')




print ('Saving data ...')
new_train.to_csv('new_train.csv', index=False, sep=',')
new_test.to_csv('new_test.csv', index=False, sep=',')
print ('Done')
'''

