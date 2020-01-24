#funkcja poboczna - wczytanie danych
#wykorzystuje format biblioteki sklearn - catboost go rozumie

from sklearn import model_selection
import pandas
def read_data(printing=False):
    """ Zapisuje pod """
    filename_red = '../Data/winequality-red.csv'
    filename_white = '../Data/winequality-white.csv'
    red_data  =pandas.read_csv(filename_red,delimiter=';')
    white_data = pandas.read_csv(filename_white,delimiter=';')

    """A to mniej na żywca, a więcej z dokumentacją"""
    if printing == True:
        print(red_data.sum)
        print(white_data.sum)
        print("Liczba powtórzeń w czerwonych winach")
        print(red_data.duplicated().tolist().count(True))
        print("Liczba powtórzeń w białych winach")
        print(white_data.duplicated().tolist().count(True))
        print("Usunięcie powtórzeń")
    red_data = red_data.drop_duplicates().reset_index()
    white_data = white_data.drop_duplicates().reset_index()
    if printing == True:
        print(red_data.sum)
        print(white_data.sum)
    red_data_values = red_data.drop(columns="quality")
    if printing == True:
        print(red_data_values.head())
    red_data_quality = red_data.drop(columns=(red_data.keys()[1:12]))
    if printing == True:
        print(red_data_quality.head())

    white_data_values = white_data.drop(columns="quality")
    if printing == True:
        print(white_data_values.head())
    white_data_quality = white_data.drop(columns=(white_data.keys()[1:12]))
    if printing == True:
        print(white_data_quality.head())

    red_data_training,red_data_test, red_quality_training, red_quality_test = model_selection.train_test_split(red_data_values,red_data_quality,test_size=0.2,random_state=42,shuffle=False)

    white_data_training,white_data_test, white_quality_training, white_quality_test = model_selection.train_test_split(white_data_values,white_data_quality,test_size=0.2,random_state=42,shuffle=False)
    # koniec uniwersalnej czesci
    return red_data_training,red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test