import numpy as np

from pyinterpolate.kriging.point.universal import MultivariateRegression


def test_init():
    mr = MultivariateRegression()
    assert isinstance(mr, MultivariateRegression)


def test_fit():
    number_of_columns = np.random.randint(2, 100)
    random_arr = np.random.random(size=(1000, number_of_columns))

    mr = MultivariateRegression()
    mr.fit(random_arr)
    assert isinstance(mr.features, np.ndarray)
    assert isinstance(mr.intercept, float)
    assert isinstance(mr.coefficients, np.ndarray)


def test_predict():
    ds = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4.5]
    ])

    mr = MultivariateRegression()
    mr.fit(ds)
    predictions = mr.predict(np.array([[2.5], [4]]))
    assert np.allclose(predictions[0], 3.775, rtol=3)
    assert np.allclose(predictions[1], 5.5, rtol=3)
