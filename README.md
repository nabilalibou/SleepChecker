# SleepChecker

Wrapper of the yasa package to have a simple class allowing to:

Combine predictions from several EEG electrodes to return the Sleep Stages predicted

Method to get overall sleep percentage detected on the EEG data

Method to directly annotate the sleep sequences on the MNE Raw data

I use this class to detect whether the subjects have fallen asleep during the protocol, especially when it is resting state eyes closed (it is more frequent than we think).

Can deliver a sleep score (percentage of period detected as asleep in the data), and you can directly annotate these sleep period on your MNE raw data using ...
