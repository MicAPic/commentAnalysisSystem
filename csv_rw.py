import pandas as pd


def read(filename, sep='\t'):
    # Read a csv file to a dataframe with custom delimiter
    df = pd.read_csv(filename, sep=sep, on_bad_lines='skip', engine="python")
    print('Contents of Dataframe: ')
    print(df)
    return df


def write(df: pd.DataFrame, filename, sep='\t'):
    # Write a dataframe to a csv file with custom delimiter
    df.to_csv(filename, sep=sep, index=False)
    # print('File written successfully')
