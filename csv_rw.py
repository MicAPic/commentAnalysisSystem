import pandas as pd


def read(path: str, sep='\t') -> pd.DataFrame:
    """
    Read a csv/tsv/etc. file as a dataframe with a custom delimiter and print its contents

    :param path: Path to the file
    :param sep: Delimiter, a horizontal tab by default
    :return: pandas DataFrame
    """

    df = pd.read_csv(path, sep=sep, on_bad_lines='skip', engine="python")
    print('Contents of Dataframe: ')
    print(df)
    return df


def write(df: pd.DataFrame, path: str, sep='\t'):
    """
    Write a dataframe to a csv/tsv/etc. file with a custom delimiter

    :param df: pandas DataFrame
    :param path: Path to the file
    :param sep: Delimiter, a horizontal tab by default
    """

    df.to_csv(path, sep=sep, index=False)
