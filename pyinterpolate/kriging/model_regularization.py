import numpy as np


def calculate_semivariogram_deviation(data_based_semivariogram, theoretically_regularized_semivariogram):
    """Function calculates deviation between experimental and theoretical semivariogram
    over given lags.
    
    INPUT:
    :param data_based_semivariogram: data based semivariogram in the form of numpy array:
    [[lag 0, value 0],
     [lag i, value i],
     [lag z, value z]]
    
    :param theoretically_regularized_semivariogram: array in the same for as data_based_semivariogram,
    where first column represents the same lags as the first column of the data_based_semivariogram array.
    
    OUTPUT:
    :return deviation: scalar which describes deviation between semivariograms.
     """
    
    array_length = len(data_based_semivariogram)
    
    if array_length == len(theoretically_regularized_semivariogram):
        if (data_based_semivariogram[:, 0] == theoretically_regularized_semivariogram[:, 0]).any()
            print('Start of deviation calculation')
            deviation = np.abs(theoretically_regularized_semivariogram[:, 1] - data_based_semivariogram[:, 1])
            deviation = deviation / data_based_semivariogram[:, 1]
            deviation = sum(deviation) / array_length
            print('Calculated deviation is:', deviation)
            return deviation
        else:
            raise ValueError('Semivariograms have a different lags')
    else:
        raise ValueError('Length of data based semivariogram is different than length of theoretical semivariogram')
