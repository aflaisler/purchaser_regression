import pandas as pd
import numpy as np
import os


def load_data(filename):
    '''
    - return dataframe from the input data, rename the columns
    - remove duplicates or NA
    '''

    # Load the events file in a DataFrame
    df = pd.read_csv(filename, header=None, sep='\t', parse_dates=[1])
    df.columns = ['userID', 'date', 'event']

    # drop duplicates and NA
    df_ = df.drop_duplicates()
    df_out = df_.dropna()
    return df_out


def unit_test(filename):
    '''
    - Test if the sum of the extrated events equals the input events
    '''
    # Load all the events files into one dataframe df_
    df_1 = pd.DataFrame()
    for file in os.listdir('./data/Events'):
        df = pd.read_csv('./data/Events/' + file)
        df_1 = df_1.append(df)

    # Test if same shape than original input file and return boolean
    df_2 = load_data(filename)
    if not df_2.shape == df_1.shape:
        raise ValueError('output shape different from input')
        return df_2.shape == df_1.shape
    else:
        return df_2.shape == df_1.shape


def process_Events_file(filename):
    '''
    - Split the Events file into several smaller files
    - Split according to the category of the event

    - This is to allow faster execution/loading and better
    organisation of the project
    '''
    # Create a directory to save the files
    # If it does not already exists
    # This is to avoid crowding the Data directory
    if not os.path.exists('./Data/Events/'):
        os.makedirs('./Data/Events/')

    # load the data
    df = load_data(filename)

    # Loop over each unique event type
    for event in df['event'].unique():

        # Select entries that match the event type
        df_out = df[df.event == event]

        # Save to csv file
        df_out.to_csv('./Data/Events/Events_type_%s.csv' %
                      (event), index=False)
        # index = False : don't save the index as a column
    print('file preprocessed, checking the output...')
    # test the output
    test_result = unit_test(filename)
    print('Test passed: %s' % (test_result))


if __name__ == '__main__':
    # This structure (if name == main) makes sure that
    # the following commands are not executed when the script
    # is imported in another script
    filename = './Data/training.tsv'
    process_Events_file(filename)
