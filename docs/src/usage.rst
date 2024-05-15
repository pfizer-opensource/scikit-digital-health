Usage
=====

Basic examples of using the Scikit Digital Health libary. Refer to the *SKDH Reference* for more advanced usage.

.. tab-set::

    .. tab-item:: Preparing data

        ::

          unix_s = data['TIMESTAMP_COLUMN'].values # must be in unix time
          accels = data[['ACCX_COLUMN', 'ACCY_COLUMN', 'ACCZ_COLUMN']].values # accelerometer columns

    .. tab-item:: Import libraries
        :selected:

        ::

          import skdh # this imports the scikit-digital-health library
          from skdh.activity import ActivityLevelClassification # this imports the activity classification module
          from skdh.preprocessing import GetDayWindowIndices # this imports the day windowing module
          from skdh.preprocessing import CountWearDetection # this imports the wear detection module
          from skdh.gait import GaitLumbar # this imports the older gait classification and gait feature extraction module
          from skdh.sit2stand import Sit2Stand # this imports the sit to stand prediction and feature extraction module
          from skdh.sleep import Sleep # this imports the sleep classification and feature extraction module

    .. tab-item:: Defining classes

        ::

          ACLASS = ActivityLevelClassification()
          CWD = CountWearDetection()
          GAIT = GaitLumbar()
          S2S = Sit2Stand()
          SLEEP = Sleep()
    .. tab-item:: Defining classes

        ::
          
          # Wear Detection and Identifying Wear Indicies
          wear = CWD.predict(
              time = unix_s, 
              accel = accels, 
          )

          # Activity Level Classification and Activity-Based Feature Extraction
          act = ACLASS.predict(
              time = unix_s, 
              accel = accels,
              #wear=wear['wear'] #optional argument, can use the output from the wear prediction
          )

          # Gait-Based Feature Extraction
          gait = GAIT.predict(
              time = unix_s, 
              accel = accels,
              #wear=wear['wear'] #optional argument, can use the output from the wear prediction
          )

          # Sit to Stand Classification and Feature Extraction
          sit2stand = S2S.predict(
              time = unix_s, 
              accel = accels,
              #wear=wear['wear'] #optional argument, can use the output from the wear prediction
          )

          # Sleep Classification and Sleep-Based Feature Extraction
          sleep = SLEEP.predict(
              time = unix_s, 
              accel = accels,
              #wear=wear['wear'] #optional argument, can use the output from the wear prediction
          )
