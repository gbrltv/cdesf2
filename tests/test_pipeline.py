from cdesf2.utils import read_csv
from cdesf2.core import CDESF


def test_pipeline():
    event_stream_test = read_csv(f'demo/sudden_trace_noise20_500_cb.csv')
    cdesf = CDESF(name='sudden_trace_noise20_500_cb',
                  time_horizon=259200,
                  lambda_=0.05,
                  beta=0.2,
                  epsilon=0.6,
                  mu=4,
                  stream_speed=100,
                  n_features=2,
                  gen_metrics=True,
                  gen_plot=False)
    cdesf.run(event_stream_test)
