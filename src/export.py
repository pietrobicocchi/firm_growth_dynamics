import os
import matplotlib.pyplot as plt
import pickle


def save_figure(figure, filename, fileformat='png', directory='./results/plots'):
    """
    Saves a matplotlib figure to a specified file path.

    Parameters:
        figure (matplotlib.figure.Figure): the figure object to save.
        filename (str): name of the file to save the plot as.
        fileformat (str): format of the file to save the plot as (default is 'png').
        directory (str): directory to save the plot in (default is current directory).

    Returns:
        None
    """
    # set the file path
    filepath = os.path.join(directory, filename)

    # create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the figure
    figure.savefig(filepath, format=fileformat)

    # close the figure
    plt.close(figure)


def save_cleaned_data(data, filepath):
    """
    Saves cleaned data to a file using pickle.

    Args:
        data: The cleaned data to be saved.
        filepath (str): The filepath where the data will be saved.

    Returns:
        None

    Raises:
        IOError: If there is an error in saving the data.

    Example:
        cleaned_data = ...

        save_cleaned_data(cleaned_data, 'path/to/save/data.pkl')
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except IOError as e:
        raise IOError("Error saving data: " + str(e))


def load_cleaned_data(filepath):
    """
    Loads cleaned data from a file using pickle.

    Args:
        filepath (str): The filepath from where the data will be loaded.

    Returns:
        The loaded cleaned data.

    Raises:
        IOError: If there is an error in loading the data.

    Example:
        loaded_data = load_cleaned_data('path/to/load/data.pkl')
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data
    except IOError as e:
        raise IOError("Error loading data: " + str(e))


if __name__ == "__main__":
    # Assuming you have a cleaned DataFrame called `cleaned_df`

    # Path where to save the data
    path_clean = '/Users/pietrobicocchi/Desktop/project/data/cleaned/cleaned_data.pkl'

    # Save the cleaned DataFrame
    save_cleaned_data(cleaned_df, path_clean)

    # Load the saved DataFrame
    loaded_df = load_cleaned_data(path_clean)

    # Now you can use the loaded DataFrame
    print(loaded_df.head())
