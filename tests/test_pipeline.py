from cdesf.utils import read_csv
from cdesf.core import CDESF


def test_pipeline():
    event_stream_test = read_csv(f"demo/sudden_trace_noise20_500_cb.csv")
    cdesf = CDESF(
        name="sudden_trace_noise20_500_cb",
        time_horizon=259200,
        lambda_=0.05,
        beta=0.2,
        epsilon=0.6,
        mu=4,
        stream_speed=100,
        gen_metrics=True,
        gen_plot=False,
    )
    cdesf.run(event_stream_test)


def test_attributed_pipeline():
    event_stream_test = read_csv(f"demo/sudden_trace_noise20_500_cb_attributed.csv")
    cdesf = CDESF(
        name="sudden_trace_noise20_500_cb_attributed",
        time_horizon=259200,
        lambda_=0.05,
        beta=0.2,
        epsilon=0.6,
        mu=4,
        stream_speed=100,
        gen_metrics=True,
        gen_plot=False,
        additional_attributes=["first", "second", "country"],
    )
    cdesf.run(event_stream_test)
