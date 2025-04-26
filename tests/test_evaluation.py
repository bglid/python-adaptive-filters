import pytest

from adaptive_filter.algorithms import apa, frequency_domain, fx_lms, lms, nlms, rls
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
    "mu, filter_order, algorithm, expected",
    [
        pytest.param(0.01, 16, "LMS", lms.LMS(mu=0.01, n=16), id="Valid_LMS"),
        # test when algorithm is mispelt
        pytest.param(
            0.01,
            16,
            "LMSS",
            ValueError,
            marks=pytest.mark.xfail(raises=ValueError),
            id="Algorithm mispelling",
        ),
        # test when algorithm is lowercase
        pytest.param(
            0.01,
            16,
            "lms",
            ValueError,
            marks=pytest.mark.xfail(raises=ValueError),
            id="Algorithm lowercase...",
        ),
        # test when algorithm is random
        pytest.param(
            0.01,
            16,
            "xyZzy",
            ValueError,
            marks=pytest.mark.xfail(raises=ValueError),
            id="Unknown Algorithm...",
        ),
    ],
)
def test_select_algorithm(filter_order, mu, algorithm, expected):
    # checking erros are actually an exception
    if isinstance(expected, type) and issubclass(expected, Exception):
        # pytest considers this expected to fail, which it should
        select_algorithm(filter_order, mu, algorithm)
    else:
        # else assert the result is as expected
        result = select_algorithm(filter_order, mu, algorithm)
        assert result.__class__ == expected.__class__
        assert result.mu == expected.mu
        assert result.N == expected.N


@pytest.mark.parametrize(
    "filter_order, mu, algorithm, noise, delay_amount, random_noise_amount, fs, snr_levels, save_result, expected",
    [
        pytest.param(
            16, 0.1, "LMS", "babble", 5.0, 30, 16000, 1, False, dict[str, float]
        ),
    ],
)
def test_noise_evaluation(
    filter_order,
    mu,
    algorithm,
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
        noise,
        delay_amount,
        random_noise_amount,
        fs,
        snr_levels,
        save_result,
    )

    assert isinstance(result, dict)
