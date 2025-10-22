from edital_extractor.utils import to_quantity_br


def test_to_quantity_br_handles_integers():
    assert to_quantity_br("12") == 12
    assert to_quantity_br("1.234") == 1234


def test_to_quantity_br_handles_decimals():
    assert to_quantity_br("3,5") == 3.5
    assert to_quantity_br("1.250,40") == 1250.4
