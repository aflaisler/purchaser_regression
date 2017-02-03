import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# Load the data
def load_events():
    '''
    - load events from the preprocessed data
    '''

    # Load all the events files into one dataframe df
    df = pd.DataFrame()
    for file in os.listdir('./data/Events'):
        df_ = pd.read_csv('./data/Events/' + file)
        df = df.append(df_)
    return df


# Prepare the data for the logistic regression
def encoding(df):
    '''
    - take a dataframe of events and returns a dataframe ready for the regression
    '''
    # We are now going to do the one-hot encoding of the brands
    print('One-hot encoding')
    # We select only the event column as a start
    df_event_toframe = df['event'].to_frame()
    # Use get_dummies from pandas to get the one-hot encoding
    df_onehot = pd.get_dummies(df_event_toframe)
    # Then add the one hot encoding back to the original dataframes
    df_onehot_wUser = pd.concat([df['userID'], df_onehot], 1)
    print('Group by')
    # we group by userID
    df_groupby = df_onehot_wUser.groupby(['userID'], as_index=False).sum()
    # we don't need the userIDs and the purchases for the regression: we drop them
    # also the test data does not contain 'event_CustomerSupport' we drop it
    # as well
    userID = df_groupby['userID']
    df_reg = df_groupby.drop(['userID'], 1).reset_index(drop=True)
    # encode the purchase event if present
    if 'event_Purchase' in df_reg.columns:
        # add column purchase True (1) or False (0)
        df_reg['Purchase'] = np.where(df_groupby['event_Purchase'] > 0, 1, 0)
        #remove the column
        df_reg = df_reg.drop(['event_Purchase'], 1).reset_index(drop=True)
    # remove customer support if present
    if 'event_CustomerSupport' in df_reg.columns:
        df_reg = df_reg.drop(['event_CustomerSupport'], 1).reset_index(drop=True)
    return userID, df_reg



def modelCreation():
    '''
    - create the model by doing a logistic regression
    '''
    # Load the data
    df = load_events()
    # format it for the regression
    userID, df_reg = encoding(df)
    # We select the columns that we want to use to predict a purchase
    columns = df_reg.columns
    training_data = [col for col in columns if col != 'Purchase']
    # actual regression using sklearn
    model = LogisticRegression(fit_intercept=True)
    X = df_reg[training_data]
    y = df_reg[['Purchase']]
    mdl = model.fit(X, y)
    # Let's examine the coefficients
    print(pd.DataFrame(zip(X.columns, np.transpose(model.coef_))))
    # check the accuracy of the model:
    print('the model accuracy is %s' % (model.score(X, y)))

    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)

    # generate evaluation metrics
    print('Test on subgroup: %s' %(metrics.accuracy_score(y_test, predicted)))
    # our model is able to predict purchases with an accuracy of 70%
    return model

def evaluateData(filename_test):
    '''
    - return a csv file with the predictions
    '''
    df = pd.read_csv(filename_test, sep='\t', header=None)
    df.columns = ['userID', 'date', 'event']
    # drop duplicates and NA
    df_ = df.drop_duplicates()
    df_ = df_.dropna()
    # encode for regression
    userID, df_reg = encoding(df_)

    # get the model
    model = modelCreation()
    col_ = ['event_EmailClickthrough', 'event_EmailOpen',
            'event_FormSubmit', 'event_PageView', 'event_WebVisit']
    if np.array_equal(df_reg.columns, col_):
        predict = pd.DataFrame(model.predict(df_reg), dtype=int)
        predict.columns = ['prediction']
        df_out = pd.concat([userID, df_reg, predict], 1)
    df_out.to_csv('prediction.csv')
    df_potential_purchasers = df_out[df_out['prediction'] == 1]
    df_potential_purchasers.to_csv('potential_purchasers.csv')

if __name__ == '__main__':
    # This structure (if name == main) makes sure that
    # the following commands are not executed when the script
    # is imported in another script
    filename_test = './Data/test.tsv'
    evaluateData(filename_test)
