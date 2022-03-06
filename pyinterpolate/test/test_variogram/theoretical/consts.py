from dataclasses import dataclass
import numpy as np


@dataclass
class TheoreticalVariogramData:
    zero_variances = np.array(
        list(
            zip(np.arange(0, 10), np.zeros(10))
        )
    )


if __name__ == '__main__':
    tvd = TheoreticalVariogramData
    print(tvd.zero_variances)