import cProfile
from pyinterpolate.pipelines.samples import download_air_quality_poland


def profile_air_quality():
    air_q = download_air_quality_poland('PM10', export=True)
    return air_q


if __name__ == '__main__':
    cProfile.run('profile_air_quality()', filename='get_air_quality_v0.3.0.profile')
