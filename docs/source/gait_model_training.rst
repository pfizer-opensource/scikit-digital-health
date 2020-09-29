Training a Gait Model
---------------------

Gait model training is setup to use data in a specific format for loading.

Folder structure: 

* Study folder

  * subject_1.h5
  * subject_2.h5
  * ...

Each h5 file should then be setup in the following manner:

* subject_1.h5

  * [attr] Age (optional)
  * [attr] Gender (optional)
  * [attr] Height (optional)
  * [attr] ...
  * Activity-1

    * [attr] Gait label (bool, is considered gait)
    * Trial 1

      * [attr] Sampling rate
      * Accelerometer [g]
      * Gyroscope [rad/s]
    
    * Trial 2

      * ...
