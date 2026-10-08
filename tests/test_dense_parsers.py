"""Dataset parsers tested on tiny synthetic files with the real layouts."""

import zipfile

import numpy as np
import pytest

from aida_data import dense


def _zip(path, members: dict[str, str | bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def _decode(colors, name):
    j = colors.names.index(name)
    return [colors.labels[j][c] for c in colors.values[:, j]]


def test_adult(tmp_path):
    rows = [
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K",
        "50, ?, 83311, Bachelors, 13, Married-civ-spouse, ?, Husband, White, Female, 0, 0, 13, ?, <=50K",
        "38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, ?, Male, 0, 0, 40, United-States, >50K",
        "",
    ]
    path = _zip(tmp_path / "adult.zip", {"adult.data": "\n".join(rows)})
    data, test, distances, colors = dense._load_adult(path)
    assert test is None and distances is None
    np.testing.assert_array_equal(
        data, [[39, 77516, 13, 2174, 40], [50, 83311, 13, 0, 13]]
    )
    assert colors.names == ("sex", "race", "marital-status")
    # leading spaces stripped, "?" rows dropped only if in a used column
    assert _decode(colors, "sex") == ["Male", "Female"]
    assert colors.labels[0] == ("Female", "Male")
    assert _decode(colors, "marital-status") == ["Never-married", "Married-civ-spouse"]


def test_athlete(tmp_path):
    path = tmp_path / "athlete_events.csv"
    path.write_text(
        '"ID","Name","Sex","Age","Height","Weight","Team"\n'
        '"1","A","M",24,180,80,"China"\n'
        '"2","B","F",NA,170,60,"China"\n'
        '"3","C","F",21,165,55.5,"Italy"\n'
    )
    data, _, _, colors = dense._load_athlete(path)
    np.testing.assert_array_equal(data, [[24, 180, 80], [21, 165, 55.5]])
    assert _decode(colors, "Sex") == ["M", "F"]


def test_diabetes(tmp_path):
    header = "encounter_id,race,gender,age,weight,time_in_hospital,num_lab_procedures,num_procedures,num_medications,diag_1,diag_2,diag_3,number_diagnoses,readmitted"
    rows = [
        "1,Caucasian,Female,[0-10),?,1,41,0,1,250.83,276,255,1,NO",
        "2,Asian,Male,[70-80),?,3,59,0,18,V57,250.01,255,9,>30",  # V code
        "3,?,Male,[10-20),?,2,11,5,13,8,428,250.43,6,NO",  # missing race
        "4,AfricanAmerican,Male,[90-100),?,2,44,1,16,414,411,401,7,<30",
    ]
    path = _zip(
        tmp_path / "diabetes.zip",
        {"diabetic_data.csv": "\n".join([header] + rows) + "\n"},
    )
    data, _, _, colors = dense._load_diabetes(path)
    np.testing.assert_allclose(
        data,
        [
            [0, 1, 41, 0, 1, 250.83, 276, 255, 1],
            [90, 2, 44, 1, 16, 414, 411, 401, 7],
        ],
        rtol=1e-6,
    )
    assert colors.names == ("gender", "race")
    assert _decode(colors, "race") == ["Caucasian", "AfricanAmerican"]


def test_kfc(tmp_path):
    path = tmp_path / "bank_categorized.csv"
    path.write_text(
        "age,balance,duration,job,marital,education\n"
        "30,1787,79,10,1,0\n"
        "33,4789,220,7,2,1\n"
        "35,1350,185,4,1,2\n"
    )
    data, _, _, colors = dense._load_kfc(
        path, features=["age", "balance", "duration"], colors=["marital"]
    )
    np.testing.assert_array_equal(data[:, 0], [30, 33, 35])
    assert colors.labels == (("1", "2"),)
    np.testing.assert_array_equal(colors.values[:, 0], [0, 1, 0])


@pytest.mark.network
def test_adult_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    ds = dense.load("adult")
    assert ds.dataset.shape[1] == 5
    assert 32000 < ds.dataset.shape[0] <= 32561
    assert ds.colors.n_colors("sex") == 2


def _compress_z(data: bytes) -> bytes:
    """Minimal Unix `compress` (.Z) writer emitting only literal 9-bit codes,
    valid as long as fewer than 255 codes are written."""
    assert len(data) < 255
    out, acc, nbits = bytearray(b"\x1f\x9d\x90"), 0, 0
    for byte in data:
        acc |= byte << nbits
        nbits += 9
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc)
    return bytes(out)


def test_breast(tmp_path):
    rows = [
        ",".join(["842302", "M"] + [str(i) for i in range(30)]),
        ",".join(["842517", "B"] + [str(i + 0.5) for i in range(30)]),
    ]
    path = _zip(tmp_path / "breast.zip", {"wdbc.data": "\n".join(rows) + "\n"})
    data, _, _, colors = dense._load_breast(path)
    assert data.shape == (2, 30)
    np.testing.assert_array_equal(data[0], np.arange(30))
    assert _decode(colors, "diagnosis") == ["M", "B"]


def test_wine(tmp_path):
    header = '"fixed acidity";"volatile acidity";"alcohol";"quality"\n'
    path = _zip(
        tmp_path / "wine.zip",
        {
            "winequality-red.csv": header + "7.4;0.7;9.4;5\n",
            "winequality-white.csv": header + "7;0.27;8.8;6\n6.3;0.3;9.5;5\n",
        },
    )
    data, _, _, colors = dense._load_wine(path)
    np.testing.assert_allclose(data, [[7.4, 0.7, 9.4], [7, 0.27, 8.8], [6.3, 0.3, 9.5]])
    assert colors.names == ("type", "quality")
    assert _decode(colors, "type") == ["red", "white", "white"]
    assert _decode(colors, "quality") == ["5", "6", "5"]


def test_shuttle(tmp_path):
    pytest.importorskip("unlzw3")
    train = b"55 0 81 0 -6 11 25 88 64 4\n56 0 96 0 52 -4 40 44 4 1\n"
    test = b"37 0 76 0 28 18 40 48 8 1\n"
    path = _zip(
        tmp_path / "shuttle.zip",
        {"shuttle.trn.Z": _compress_z(train), "shuttle.tst": test},
    )
    data, _, _, colors = dense._load_shuttle(path)
    assert data.shape == (3, 9)
    np.testing.assert_array_equal(data[:, 0], [55, 56, 37])
    assert colors.labels == (("1", "4"),)
    np.testing.assert_array_equal(colors.values[:, 0], [1, 0, 0])


def test_rt_iot(tmp_path):
    content = (
        ",id.orig_p,id.resp_p,proto,service,flow_duration,fwd_pkts_tot,fwd_last_window_size,Attack_type\n"
        "0,38667,1883,tcp,mqtt,32.0,9,502,MQTT_Publish\n"
        "1,51143,1883,tcp,mqtt,31.8,9,502,MQTT_Publish\n"
        "2,44459,53,udp,dns,0.1,1,0,DOS_SYN_Hping\n"
    )
    path = _zip(tmp_path / "rt-iot2022.zip", {"RT_IOT2022": content})
    data, _, _, colors = dense._load_rt_iot(path)
    np.testing.assert_allclose(data, [[32.0, 9, 502], [31.8, 9, 502], [0.1, 1, 0]])
    assert colors.labels == (("DOS_SYN_Hping", "MQTT_Publish"),)


def test_biokdd(tmp_path):
    import io
    import tarfile

    rows = [
        "\t".join(["279", "261532", "0"] + [str(i) for i in range(74)]),
        "\t".join(["279", "261533", "1"] + [str(-i) for i in range(74)]),
    ]
    payload = ("\n".join(rows) + "\n").encode()
    path = tmp_path / "data_kddcup04.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name in ("bio_test.dat", "bio_train.dat"):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    data, _, _, colors = dense._load_biokdd(path)
    assert data.shape == (2, 74)
    np.testing.assert_array_equal(data[1], -np.arange(74))
    np.testing.assert_array_equal(colors.values[:, 0], [0, 1])
    assert (tmp_path / "biokdd.parquet").is_file()


def test_metropt3(tmp_path):
    header = ",timestamp,TP2,TP3,H1,DV_pressure,Reservoirs,Oil_temperature,Motor_current,COMP,DV_eletric,Towers,MPG,LPS,Pressure_switch,Oil_level,Caudal_impulses\n"
    row = "0,2020-02-01 00:00:00,1,2,3,4,5,6,7,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0\n"
    path = _zip(
        tmp_path / "metropt3.zip", {"MetroPT3(AirCompressor).csv": header + row}
    )
    data, _, _ = dense._load_metropt3(path)
    np.testing.assert_array_equal(data, [[1, 2, 3, 4, 5, 6, 7]])


def test_household_power(tmp_path):
    content = (
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
        "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000\n"
        "21/12/2006;11:23:00;?;?;?;?;?;?;\n"
    )
    path = _zip(
        tmp_path / "power.zip", {"household_power_consumption.txt": content}
    )
    data, _, _ = dense._load_household_power(path)
    assert data.shape == (2, 7)
    np.testing.assert_allclose(data[0], [4.216, 0.418, 234.84, 18.4, 0, 1, 17], rtol=1e-6)
    assert np.isnan(data[1]).all()


@pytest.mark.network
def test_breast_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    ds = dense.load("breast")
    assert ds.dataset.shape == (569, 30)
    assert ds.colors.labels == (("B", "M"),)


def test_covertype(tmp_path):
    import gzip

    rows = [
        ",".join(map(str, list(range(54)) + [5])),
        ",".join(map(str, list(range(1, 55)) + [2])),
    ]
    payload = gzip.compress(("\n".join(rows) + "\n").encode())
    path = _zip(tmp_path / "covertype.zip", {"covtype.data.gz": payload})
    data, _, _, colors = dense._load_covertype(path)
    assert data.shape == (2, 54)
    np.testing.assert_array_equal(data[1], np.arange(1, 55))
    assert colors.labels == (("2", "5"),)
    np.testing.assert_array_equal(colors.values[:, 0], [1, 0])


def test_census1990(tmp_path):
    columns = ["caseid", "dAge"] + dense._CENSUS1990_FEATURES[:10] + ["iSex"]
    columns += dense._CENSUS1990_FEATURES[10:]
    rows = [
        [10000, 5] + list(range(10)) + [1] + list(range(10, 66)),
        [10001, 3] + [7] * 10 + [0] + [8] * 56,
    ]
    content = "\n".join(",".join(map(str, r)) for r in [columns] + rows) + "\n"
    path = _zip(tmp_path / "census.zip", {dense._CENSUS1990_MEMBER: content})
    data, _, _, colors = dense._load_census1990(path)
    np.testing.assert_array_equal(data[0], np.arange(66))
    assert colors.names == ("dAge", "iSex")
    assert _decode(colors, "dAge") == ["5", "3"]
    assert _decode(colors, "iSex") == ["1", "0"]


def test_phones(tmp_path):
    import io

    content = (
        "Index,Arrival_Time,Creation_Time,x,y,z,User,Model,Device,gt\n"
        "0,1424696633908,1424696631913248572,-5.958191,0.6880646,8.135345,a,nexus4,nexus4_1,stand\n"
        "1,1424696633909,1424696631918283972,-5.95224,0.6702118,8.136536,b,s3,s3_1,null\n"
    )
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr(dense._PHONES_MEMBER, content)
        zf.writestr("Activity recognition exp/Phones_gyroscope.csv", "unused")
    path = _zip(
        tmp_path / "phones.zip",
        {dense._PHONES_INNER_ZIP: inner.getvalue(), "Still exp.zip": b""},
    )
    data, _, _, colors = dense._load_phones(path)
    np.testing.assert_allclose(
        data, [[-5.958191, 0.6880646, 8.135345], [-5.95224, 0.6702118, 8.136536]]
    )
    assert colors.names == ("gt", "User", "Model", "Device")
    # `null` is kept as an activity, not treated as a missing value
    assert _decode(colors, "gt") == ["stand", "null"]
    assert _decode(colors, "Device") == ["nexus4_1", "s3_1"]


def test_higgs_and_highlevel_share_the_cache(tmp_path):
    import gzip

    rows = [[1.0] + list(np.arange(28) / 10), [0.0] + list(-np.arange(28) / 10)]
    content = "\n".join(",".join(f"{v:.18e}" for v in r) for r in rows) + "\n"
    path = tmp_path / "HIGGS.csv.gz"
    path.write_bytes(gzip.compress(content.encode()))
    data, _, _, colors = dense._load_higgs(path)
    assert data.shape == (2, 28)
    np.testing.assert_allclose(data[0], np.arange(28) / 10, rtol=1e-6)
    assert _decode(colors, "label") == ["1", "0"]
    mtime = (tmp_path / "higgs.parquet").stat().st_mtime_ns
    high, _, _, colors_high = dense._load_higgs_highlevel(path)
    np.testing.assert_array_equal(high, data[:, -7:])
    assert high.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(colors_high.values, colors.values)
    assert (tmp_path / "higgs.parquet").stat().st_mtime_ns == mtime


@pytest.mark.network
def test_covertype_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    ds = dense.load("covertype", deduplicate=False)
    assert ds.dataset.shape == (581012, 54)
    assert ds.colors.n_colors("cover_type") == 7
