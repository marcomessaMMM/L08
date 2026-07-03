from L_08_ex_package_fc.fc_Messa import parse_coords

def test_parse_coords_decimal():
    ra, dec = parse_coords("10", "20")
    assert isinstance(ra, float)
    assert isinstance(dec, float)

def test_parse_coords_sexagesimal():
    ra, dec = parse_coords("13:00:00", "-30:00:00")
    assert 0 <= ra <= 360
    assert -90 <= dec <= 90
