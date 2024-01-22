# SleepChecker

Wrapper of the yasa package (https://github.com/raphaelvallat/yasa/tree/master) to have a simple class allowing to:

=> wrap only:
Automatic sleep staging function : automatic_staging: Automatic sleep staging of polysomnography data.


Combine predictions from several EEG electrodes to return the Sleep Stages predicted

Method to get overall sleep percentage detected on the EEG data

Method to directly annotate the sleep sequences on the MNE Raw data

I use this class to detect whether the subjects have fallen asleep during the protocol, especially when it is resting state eyes closed (it is more frequent than we think).

Can deliver a sleep score (percentage of period detected as asleep in the data), and you can directly annotate these sleep period on your MNE raw data using ...

Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092
