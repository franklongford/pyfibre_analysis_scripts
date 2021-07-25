import os

import pandas as pd
import numpy as np


def load_databases(filename, data_directories, ext='.h5'):

    # Define an empty database to load the files into
    database = pd.DataFrame()

    name, _ext = os.path.splitext(filename)
    if not _ext:
        if ext is not None:
            filename += ext

    def _load_db(db_path, group, label):
        """Helper function to load a database and assign group + label
        columns
        """
        if db_path.endswith('.h5'):
            db = pd.read_hdf(db_path, key='df')
        elif db_path.endswith('.xls'):
            db = pd.read_excel(db_path)
        db['Group'] = group
        db['Label'] = label
        return db

    print("{:20} | {:10} | {:10}".format('Group', 'N', 'Label'))
    print("-" * 42)
    # Loop through the directories to load each database
    for i, directory in enumerate(data_directories):
        # The name of the folder becomes the name of the group
        group = os.path.split(directory)[-1].lower()

        try:
            # Try to load Pandas Dataframe from directory
            db_path = os.path.join(directory, filename)
            db = _load_db(db_path, group, i + 1)
            database = pd.concat([database, db])
        except IOError:
            # Look in sub directories if not database file is present
            for folder in os.listdir(directory):
                db_path = os.path.join(directory, folder, filename)
                db = _load_db(db_path, group, i + 1)
                database = pd.concat([database, db])

        print("{:<20} | {:<10} | {:<10}".format(
            group, len(database['Group'] == group), i+1))

    return database
