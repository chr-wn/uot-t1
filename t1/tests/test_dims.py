from uot.dims import Dimension, BASE_SYMBOLS


def test_group_laws():
    L, T, M = Dimension(L=1), Dimension(T=1), Dimension(M=1)
    v = L / T
    assert v * T == L
    assert (v ** 2) / v == v
    assert (L * T) / (T * L) == Dimension()
    assert Dimension().is_dimensionless
    assert (L ** 0) == Dimension()
    assert hash(L / T) == hash(Dimension(L=1, T=-1))


def test_parse_and_vector():
    d = Dimension.parse("M L2/T2")
    assert d == Dimension(M=1, L=2, T=-2)
    assert d.vector() == (2, 1, -2, 0, 0, 0, 0)
    assert d.distance() == 5
    assert Dimension.parse("1/T").vector() == (0, 0, -1, 0, 0, 0, 0)
    assert Dimension.parse("M/(L T2)") == Dimension(M=1, L=-1, T=-2)


def test_names():
    assert Dimension.parse("L/T").name() == "speed"
    assert Dimension.parse("M2 L").name() is None
    assert Dimension.parse("M L/T2").name() == "force"


def test_invented_basis():
    x = Dimension(X1=1)
    assert (x / Dimension(T=1)).vector(basis=BASE_SYMBOLS + ("X1",)) == (0, 0, -1, 0, 0, 0, 0, 1)
