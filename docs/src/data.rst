Data
====

Data Types
----------

SKDH uses the following standardization of input data:

+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------+
| Name                  | Parameter Name | Description                                                                                                                    |
+=======================+================+================================================================================================================================+
| Time                  | `time`         | Time in seconds since 1970-01-01. These timestamps can either be naive or passed along-side time-zone information (see below). |
+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------+
| Acceleration          | `accel`        | Acceleration normalized to gravity. Signal magnitude values at rest should be ~1.0                                             |
+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------+
| Angular Velocity      | `gyro`         | Angular velocity in radians per second.                                                                                        |
+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------+
| Sampling Frequency    | `fs`           | Sampling frequency, in samples per second.                                                                                     |
+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------+

Time specification
''''''''''''''''''

Time information can be specified in two ways:

* **naive/local timestamps**: these timestamps have no timezone information associated with them. 
  They are converted directly to datetimes (i.e. human readable) without any conversion. This will likely mean that
  if the recording duration crosses a time change (e.g. Daylight Savings Time) timestamps following the change will 
  be shifted/offset by an hour.
* **aware timestamps**: these timestamps are "proper" UTC timestamps (seconds since 1970-01-01 UTC), but the time-zone name 
  is also passed to any of the SKDH methods `process` (or SKDH pipeline `run` method) to provide the necessary information to 
  convert these to local time.

IO methods usually produce **naive/local timestamps** if no timezone information (`tz_name`) is provided, as this information
is not usually available from the files themselves. However, all IO methods should accept the `tz_name` parameter which allows
the user to specify the timezone of the data, and convert the timestamps to aware timestamps.

Methods which produce final results/endpoints (e.g. `GaitLumbar`, `Sleep`, etc) will also accept the `tz_name` parameter. When passing
`tz_name` to these (or any) methods, SKDH will assume that the timestamps passed in via `time` are UTC, and must be
converted to the local time using the provided `tz_name`.

Data Flow
---------

Data flow in SKDH pipelines is such that any parameters passed in are either used, potentially 
modified (e.g. `accel` in the `CalibrateAccelerometer` process), and then passed along to
following processes, or simply passed along unused. This means that if you setup a pipeline,
any additional parameters passed to the `Pipeline.run` method will be passed down to all 
processes in the pipeline.

For example, this is what allows a height value to be passed through a pipeline all the way
to the `GaitLumbar` process where it is required to compute spatial gait metrics:

.. code-block:: python

    import skdh

    pipeline = skdh.Pipeline()
    pipeline.add(skdh.io.ReadCSV())
    pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
    pipeline.add(skdh.gait.GaitLumbar())

    pipeline.run(file='data.csv', height=1.8)  # height is passed all the way to GaitLumbar


This also means that if your data contains UTC timestamps (e.g. ActiGraph CSV files), 
passing the local timezone name will follow through the entire pipeline and make sure that
the end results are in local time.