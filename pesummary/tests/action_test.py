from pesummary.core.command_line import ConfigAction

def test_dict_from_str():
    f = ConfigAction.dict_from_str("{'H1':10, 'L1':20}")
    assert sorted(list(f.keys())) == ["H1", "L1"]
    assert f["H1"] == 10
    assert f["L1"] == 20
    f = ConfigAction.dict_from_str("{'H1':'/home/IFO0.dat', 'L1': '/home/IFO1.dat'}")
    assert f["H1"] == "/home/IFO0.dat"
    assert f["L1"] == "/home/IFO1.dat"
    f = ConfigAction.dict_from_str("{'H1'=10, 'L1'=20}")
    assert sorted(list(f.keys())) == ["H1", "L1"]
    assert f["H1"] == 10
    assert f["L1"] == 20
    f = ConfigAction.dict_from_str("{H1=10, L1=20}")
    assert sorted(list(f.keys())) == ["H1", "L1"]
    assert f["H1"] == 10
    assert f["L1"] == 20
    f = ConfigAction.dict_from_str("dict(H1=10, L1=20)")
    assert sorted(list(f.keys())) == ["H1", "L1"]
    assert f["H1"] == 10
    assert f["L1"] == 20
    f = ConfigAction.dict_from_str("{H1$10, L1$20}", delimiter="$")
    assert sorted(list(f.keys())) == ["H1", "L1"]
    assert f["H1"] == 10
    assert f["L1"] == 20
    f = ConfigAction.dict_from_str("{H1=10, L1=20}", dtype=str)
    assert sorted(list(f.keys())) == ["H1", "L1"]
    assert isinstance(f["H1"], str)
    assert isinstance(f["L1"], str)
    f = ConfigAction.dict_from_str("{H1=10, L1:20, V1=2}")
    assert f["H1"] == 10
    assert f["L1"] == 20
    assert f["V1"] == 2
    f = ConfigAction.dict_from_str("{}")
    assert not len(f)

def test_list_from_str():
    f = ConfigAction.list_from_str('[1,2,3]', dtype=int)
    assert f == [1,2,3]
    f = ConfigAction.list_from_str('[1,    2,  3]', dtype=int)
    assert f == [1,2,3]
    f = ConfigAction.list_from_str('1,    2,  3', dtype=int)
    assert f == [1,2,3]
    f = ConfigAction.list_from_str("[/home/IFO0.dat, /home/IFO1.dat]")
    assert f == ["/home/IFO0.dat", "/home/IFO1.dat"]
    f = ConfigAction.list_from_str("['/home/IFO0.dat', '/home/IFO1.dat']")
    assert f == ["/home/IFO0.dat", "/home/IFO1.dat"]
