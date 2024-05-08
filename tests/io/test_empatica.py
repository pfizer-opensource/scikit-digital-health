import pytest
import numpy as np

from skdh.io.empatica import ReadEmpaticaAvro


def test_get_accel():
    read_empatica_avro = ReadEmpaticaAvro()
    raw_accel_dict = {
        "samplingFrequency": 32.0,
        "timestampStart": 1609459200000000,
        "imuParams": {
            "physicalMin": -2,
            "physicalMax": 2,
            "digitalMin": 0,
            "digitalMax": 1024,
        },
        "x": [0, 512, 1024],
        "y": [0, 512, 1024],
        "z": [0, 512, 1024],
    }
    results_dict = {}
    key = "accel"
    read_empatica_avro.get_accel(raw_accel_dict, results_dict, key)

    assert "accel" in results_dict
    assert np.array_equal(results_dict["accel"]["fs"], 32.0)
    assert np.array_equal(
        results_dict["accel"]["time"],
        np.array([1609459200.0, 1609459200.03125, 1609459200.0625]),
    )
    assert np.array_equal(
        results_dict["accel"]["accel"],
        np.array([[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
    )


def test_get_gyroscope():
    read_empatica_avro = ReadEmpaticaAvro()
    raw_gyro_dict = {
        "samplingFrequency": 32.0,
        "timestampStart": 1609459200000000,
        "imuParams": {
            "physicalMin": -250,
            "physicalMax": 250,
            "digitalMin": 0,
            "digitalMax": 1024,
        },
        "x": [0, 512, 1024],
        "y": [0, 512, 1024],
        "z": [0, 512, 1024],
    }
    results_dict = {}
    key = "gyro"
    read_empatica_avro.get_gyroscope(raw_gyro_dict, results_dict, key)

    assert "gyro" in results_dict
    assert np.array_equal(results_dict["gyro"]["fs"], 32.0)
    assert np.array_equal(
        results_dict["gyro"]["time"],
        np.array([1609459200.0, 1609459200.03125, 1609459200.0625]),
    )
    assert np.array_equal(
        results_dict["gyro"]["values"],
        np.array([[-250.0, -250.0, -250.0], [0.0, 0.0, 0.0], [250.0, 250.0, 250.0]]),
    )


def test_get_values_1d():
    read_empatica_avro = ReadEmpaticaAvro()
    raw_dict = {
        "samplingFrequency": 4.0,
        "timestampStart": 1609459200000000,
        "values": [0, 1, 2, 3],
    }
    results_dict = {}
    key = "eda"
    read_empatica_avro.get_values_1d(raw_dict, results_dict, key)

    assert "eda" in results_dict
    assert np.array_equal(results_dict["eda"]["fs"], 4.0)
    assert np.array_equal(
        results_dict["eda"]["time"],
        np.array([1609459200.0, 1609459200.25, 1609459200.5, 1609459200.75]),
    )
    assert np.array_equal(results_dict["eda"]["values"], np.array([0, 1, 2, 3]))


def test_get_systolic_peaks():
    read_empatica_avro = ReadEmpaticaAvro()
    raw_dict = {
        "peaksTimeNanos": [
            1609459200000000000,
            1609459201000000000,
            1609459202000000000,
        ]
    }
    results_dict = {}
    key = "systolic_peaks"
    read_empatica_avro.get_systolic_peaks(raw_dict, results_dict, key)

    assert "systolic_peaks" in results_dict
    assert np.array_equal(
        results_dict["systolic_peaks"]["values"],
        np.array([1609459200.0, 1609459201.0, 1609459202.0]),
    )


def test_get_steps():
    read_empatica_avro = ReadEmpaticaAvro()
    raw_dict = {
        "samplingFrequency": 1.0,
        "timestampStart": 1609459200000000,
        "values": [0, 1, 2, 3],
    }
    results_dict = {}
    key = "steps"
    read_empatica_avro.get_steps(raw_dict, results_dict, key)

    assert "steps" in results_dict
    assert np.array_equal(results_dict["steps"]["fs"], 1.0)
    assert np.array_equal(
        results_dict["steps"]["time"],
        np.array([1609459200.0, 1609459201.0, 1609459202.0, 1609459203.0]),
    )
    assert np.array_equal(results_dict["steps"]["values"], np.array([0, 1, 2, 3]))


def test_handle_resampling():
    read_empatica_avro = ReadEmpaticaAvro()
    results_dict = {
        "accel": {
            "fs": 32.0,
            "time": np.array([1609459200.0, 1609459200.03125, 1609459200.0625]),
            "accel": np.array([[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        },
        "gyro": {
            "fs": 32.0,
            "time": np.array([1609459200.0, 1609459200.03125, 1609459200.0625]),
            "values": np.array(
                [[-250.0, -250.0, -250.0], [0.0, 0.0, 0.0], [250.0, 250.0, 250.0]]
            ),
        },
        "eda": {
            "fs": 4.0,
            "time": np.array(
                [1609459200.0, 1609459200.25, 1609459200.5, 1609459200.75]
            ),
            "values": np.array([0, 1, 2, 3]),
        },
        "systolic_peaks": {
            "values": np.array([1609459200.0, 1609459201.0, 1609459202.0])
        },
        "steps": {
            "fs": 1.0,
            "time": np.array([1609459200.0, 1609459201.0, 1609459202.0, 1609459203.0]),
            "values": np.array([0, 1, 2, 3]),
        },
    }
    fs = 32.0
    results_dict = read_empatica_avro.handle_resampling(results_dict)

    assert np.array_equal(results_dict["fs"], fs)
    assert np.array_equal(
        results_dict["time"],
        np.array([1609459200.0, 1609459200.03125, 1609459200.0625]),
    )
    assert np.array_equal(
        results_dict["accel"],
        np.array([[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
    )
    assert np.array_equal(
        results_dict["gyro"],
        np.array([[-250.0, -250.0, -250.0], [0.0, 0.0, 0.0], [250.0, 250.0, 250.0]]),
    )

    print(results_dict["eda"])

    assert np.array_equal(
        results_dict["eda"],
        np.array([0, 0.125, 0.25]),  # shortened to match accel timestamps array
    )


def test_get_datastreams():
    read_empatica_avro = ReadEmpaticaAvro(resample_to_accel=False)
    raw_dict = {
        "accelerometer": {
            "samplingFrequency": 32.0,
            "timestampStart": 1609459200000000,
            "imuParams": {
                "physicalMin": -2,
                "physicalMax": 2,
                "digitalMin": 0,
                "digitalMax": 1024,
            },
            "x": [0, 512, 1024],
            "y": [0, 512, 1024],
            "z": [0, 512, 1024],
        },
        "gyroscope": {"x": []},
        "bvp": {"values": []},
        "temperature": {"values": []},
        "eda": {
            "samplingFrequency": 4.0,
            "timestampStart": 1609459200000000,
            "values": [0, 1, 2, 3],
        },
        "systolicPeaks": {
            "peaksTimeNanos": [
                1609459200000000000,
                1609459201000000000,
                1609459202000000000,
            ]
        },
        "steps": {
            "samplingFrequency": 1.0,
            "timestampStart": 1609459200000000,
            "values": [0, 1, 2, 3],
        },
    }
    results_dict = read_empatica_avro.get_datastreams(raw_dict)

    assert "accel" in results_dict
    assert "gyro" not in results_dict
    assert "bvp" not in results_dict
    assert "temperature" not in results_dict
    assert "eda" in results_dict
    assert "systolic_peaks" in results_dict
    assert "steps" in results_dict
    assert np.isclose(results_dict["fs"], 32.0)
    assert np.isclose(results_dict["eda"]["fs"], 4.0)
    assert np.isclose(results_dict["steps"]["fs"], 1.0)
    assert np.array_equal(
        results_dict["systolic_peaks"]["values"],
        np.array([1609459200.0, 1609459201.0, 1609459202.0]),
    )
