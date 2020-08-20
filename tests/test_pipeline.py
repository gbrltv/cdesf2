from cdesf2.utils import read_csv
from cdesf2.core import CDESF


def test_pipeline():
    path = 'demo'
    filename = 'hospital_billing.csv'

    event_stream_test = read_csv(f'{path}/{filename}')
    cdesf = CDESF(name='hospital_billing',
                  time_horizon=432000,
                  lambda_=0.05,
                  beta=0.2,
                  epsilon=0.2,
                  mu=4,
                  stream_speed=100,
                  n_features=2,
                  gen_metrics=True,
                  gen_plot=True)
    cdesf.run(event_stream_test)
