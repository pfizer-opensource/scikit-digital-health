Examples
========

Run a stand-alone process
-------------------------

.. code-block:: python

    import skdh

    # load your data
    time, accel = fn_to_get_data()

    # setup and run the gait process. No prediction of gait periods, the entire
    # recording/data is assumed to be gait data. Height is subject's height
    # in meters
    gait = skdh.gait.GaitLumbar()
    gait_res = gait.predict(time=time, accel=accel, height=1.8)


Run a process with a prediction of gait periods
-----------------------------------------------

.. code-block:: python

    import skdh

    # load your data
    time, accel = fn_to_get_data()

    # setup a pipeline to first predict gait periods and then estimate gait
    # metrics during those periods.
    pipeline = skdh.Pipeline()
    pipeline.add(skdh.context.PredictGaitLumbarLgbm())
    pipeline.add(skdh.gait.GaitLumbar())

    # get the results
    res = pipeline.run(time=time, accel=accel, height=1.8)
    # gait results are in res['GaitLumbar']


Run a process starting with data ingestion
------------------------------------------

.. code-block:: python

    import skdh

    # pipeline setup
    pipeline = skdh.Pipeline()
    pipeline.add(
        skdh.io.ReadCsv(
            time_col_name='Timestamp UTC',
            column_names={'accel': ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']},
            drop_duplicate_timestamps=False,
            fill_gaps=True,
            fill_value={'accel': [0.0, 1.0, 0.0]},  # y axis is vertical
            to_datetime_kwargs={'unit': 's'},  # timestamp column is unix timestamps in seconds
            raw_conversions={'accel': 9.81},   # convert from m/s^2 to g
            read_csv_kwargs={'skiprows': 6},  # skip the first 6 rows - header information
        )
    )
    # find days (midnight-midnight) so that get gait results already with an associated
    # day number & date
    pipeline.add(skdh.preprocessing.GetDayWindowIndices(bases=[0], periods=[24]))
    pipeline.add(skdh.preprocessing.CalibrateAccelerometer())  # calibrate the accelerometer
    pipeline.add(skdh.context.PredictGaitLumbarLgbm())
    pipeline.add(
        skdh.gait.GaitLumbar(),  # default parameters
        save_file="{file}_gait_results.csv"  # automatically save gait results to a file
        # this will use the CSV file name as the start of the output file name
    )

    # run the pipeline
    res = pipeline.run(file='your_data.csv', height=1.8)
