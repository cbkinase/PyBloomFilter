import random
import pytest
from src import BloomFilter


@pytest.fixture
def empty_filter():
    return BloomFilter(10_000, fp_rate=0.01)


@pytest.fixture
def inserted_filter():
    f = BloomFilter(10_000, fp_rate=0.01)
    f.put("hello")
    f.put("world")
    return f


def test_bad_insertion_init():
    with pytest.raises(TypeError):
        BloomFilter("hello")

    with pytest.raises(TypeError):
        BloomFilter("0.5")

    with pytest.raises(ValueError):
        BloomFilter(-500)


def test_bad_fp_rate_init():
    with pytest.raises(TypeError):
        BloomFilter(10_000, fp_rate="hello")

    with pytest.raises(TypeError):
        BloomFilter(10_000, fp_rate=0.1 + 3j)

    with pytest.raises(ValueError):
        BloomFilter(10_000, fp_rate=0)

    with pytest.raises(ValueError):
        BloomFilter(10_000, fp_rate=-1)

    with pytest.raises(ValueError):
        BloomFilter(10_000, fp_rate=1)

    with pytest.raises(ValueError):
        BloomFilter(10_000, fp_rate=100)


def test_may_contain(inserted_filter: BloomFilter):
    assert inserted_filter.may_contain("hello") is True
    assert inserted_filter.may_contain("world") is True


def test_no_bogus_coercion():
    bloom = BloomFilter(1000, fp_rate=0.01)
    bloom.put(1)
    assert bloom.may_contain("1") is False


def test_expected_fpp(inserted_filter: BloomFilter):
    assert pytest.approx(inserted_filter.expected_fpp(), abs=10e-5) == 0.01


def test_put_all(inserted_filter: BloomFilter, empty_filter: BloomFilter):
    empty_filter.put_all(inserted_filter)
    assert empty_filter.may_contain("hello") is True
    assert empty_filter.may_contain("world") is True


def test_bad_put_all(inserted_filter: BloomFilter):
    bad_k = BloomFilter(100_000, fp_rate=0.03)
    bad_m = BloomFilter(100_000, fp_rate=0.01)

    with pytest.raises(ValueError):
        inserted_filter.put_all([0, 0, 0])

    with pytest.raises(ValueError):
        inserted_filter.put_all(inserted_filter)

    with pytest.raises(ValueError):
        inserted_filter.put_all(bad_k)

    with pytest.raises(ValueError):
        inserted_filter.put_all(bad_m)

    with pytest.raises(ValueError):
        inserted_filter.put_all(inserted_filter)


def test_in(inserted_filter: BloomFilter):
    assert "world" in inserted_filter


def test_not_in(empty_filter: BloomFilter):
    assert empty_filter.may_contain("hello") is False


# Let's verify the false positive estimation rate with Monte Carlo simulation


def test_false_positive_estimation_accuracy():
    size = 200_000
    offset = random.randint(0, 300_000)
    error_rate = 0.01
    bloom = BloomFilter(size, fp_rate=error_rate)

    for i in range(size):
        bloom.put(i + offset)

    # At this point, any element not in the BloomFilter should have an fp
    # rate of approximately `error_rate` - given exactly by expected_fpp()

    start = size + offset + 1_000_000
    end = size + start
    total = end - start
    hits = 0

    for i in range(start, end):
        if bloom.may_contain(i):
            hits += 1

    measured_fp_rate = hits / total
    # What's the probability this test randomly fails?
    # Probably want to tune it to be less than 1 in a million
    assert (
        pytest.approx(measured_fp_rate, abs=(error_rate / 10))
        == bloom.expected_fpp()
    )
