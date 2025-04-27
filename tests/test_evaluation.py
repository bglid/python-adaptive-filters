import pytest
from numpy import block

from adaptive_filter.algorithms import apa, fd_lms, lms, nlms, rls
from adaptive_filter.evaluation import load_data, noise_evaluation, select_algorithm
from adaptive_filter.filter_models.filter_model import FilterModel


# test for loading data
def test_load_data():
    noise, noisy_speech, clean_speech = load_data("air_conditioner")
    array_shape = noise.shape
    assert noise.shape == array_shape
    assert noisy_speech.shape == array_shape, "Lists are not the same length"
    assert clean_speech.shape == array_shape, "Lists are not the same length"


@pytest.mark.parametrize(
    "mu, filter_order, algorithm, block_size, expected",
    [
        pytest.param(0.01, 16, "LMS", 0, lms.LMS(mu=0.01, n=16), id="Valid_LMS"),
        pytest.param(0.001, 8, "NLMS", 0, nlms.NLMS(mu=0.001, n=8), id="Valid_NLMS"),
        pytest.param(0.99, 32, "RLS", 0, rls.RLS(mu=0.99, n=32), id="Valid_RLS"),
        pytest.param(
            0.01, 16, "APA", 4, apa.APA(mu=0.01, n=16, block_size=4), id="Valid_APA"
        ),
        # test when algorithm is mispelt
        pytest.param(
            0.01,
            16,
            "LMSS",
            0,
            ValueError,
            marks=pytest.mark.xfail(raises=ValueError),
            id="Algorithm mispelling",
        ),
        # test when algorithm is lowercase
        pytest.param(
            0.01,
            16,
            "lms",
            0,
            ValueError,
            marks=pytest.mark.xfail(raises=ValueError),
            id="Algorithm lowercase...",
        ),
        # test when algorithm is random
        pytest.param(
            0.01,
            16,
            "xyZzy",
            0,
            ValueError,
            marks=pytest.mark.xfail(raises=ValueError),
            id="Unknown Algorithm...",
        ),
    ],
)
def test_select_algorithm(filter_order, mu, algorithm, block_size, expected):
    # checking erros are actually an exception
    if isinstance(expected, type) and issubclass(expected, Exception):
        # pytest considers this expected to fail, which it should
        select_algorithm(filter_order, mu, algorithm, block_size)
    else:
        # else assert the result is as expected
        result = select_algorithm(filter_order, mu, algorithm, block_size)
        assert result.__class__ == expected.__class__
        assert result.mu == expected.mu
        assert result.N == expected.N


@pytest.mark.parametrize(
    "filter_order, mu, algorithm, block_size, noise, delay_amount, random_noise_amount, fs, snr_levels, save_result, expected",
    [
        pytest.param(
            16, 0.1, "LMS", 0, "babble", 5.0, 30, 16000, 1, False, dict[str, float]
        ),
    ],
)
def test_noise_evaluation(
    filter_order,
    mu,
    algorithm,
    block_size,
    noise,
    delay_amount,
    random_noise_amount,
    fs,
    snr_levels,
    save_result,
    expected,
):
    result = noise_evaluation(
        filter_order,
        mu,
        algorithm,
        block_size,
        noise,
        delay_amount,
        random_noise_amount,
        fs,
        snr_levels,
        save_result,
    )

    assert isinstance(result, dict)
