from pyinterpolate import ExperimentalVariogram, read_txt


class ExperimentalVariogramController:

    def __init__(self,
                 input_array_file: str,
                 step_size: float,
                 max_range: float,
                 direction: float = None,
                 tolerance: float = 1.0,
                 method: str = 't',
                 is_semivariance: bool = True,
                 is_covariance: bool = True):

        # Read input array file
        try:
            ds = read_txt(input_array_file)
        except Exception as ex:
            print(ex)
            raise TypeError('Not recognized file format!')

        self.experimental_variogram = ExperimentalVariogram(
            input_array=ds,
            step_size=step_size,
            max_range=max_range,
            direction=direction,
            tolerance=tolerance,
            method=method,
            is_semivariance=is_semivariance,
            is_covariance=is_covariance
        )

        print(self.experimental_variogram)

        self.experimental_variogram.plot(
            True, True, True
        )
