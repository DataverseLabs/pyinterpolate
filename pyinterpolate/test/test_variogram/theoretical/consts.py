from dataclasses import dataclass
import numpy as np


@dataclass
class TheoreticalVariogramData:

    lags = np.arange(0, 10)

    nugget0 = 0
    nugget1 = 1
    nugget_random = np.random.random()

    sill0 = 0
    sill1 = 1
    sill_random = np.random.random()

    srange0 = 0
    srange1 = 1
    srange5 = 5
    srange10 = 10
    srange_random = np.random.randint(0, 100)


if __name__ == '__main__':
    tvd = TheoreticalVariogramData
    print(tvd)
