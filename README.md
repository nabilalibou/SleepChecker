# SleepChecker

Wrapper of the [yasa](https://raphaelvallat.com/yasa/build/html/index.html) [[1]](#1) 
SleepStaging module which is an automatic sleep staging algorithm. The idea behind **SleepChecker** is to have a simple 
package to robustly detect when the subject has fallen asleep, as this happens much more than we think (especially 
during Resting State paradigms).  
To achieve this, the **SleepChecker** class provides the methods to:
- Automatically combine the sleep stages predictions into one final robust prediction (~ majority voting).
- Get the total percentage of time asleep during an EEG recording session.
- Annotate sleeping time segments on the [MNE](https://mne.tools/stable/index.html) raw EEG data.

# Installation

```
git clone https://github.com/nabilalibou/SleepChecker.git
pip install -r requirements.txt
```

# Example

```python
import mne
from SleepChecker import SleepChecker

# Load an EDF file using MNE
raw_eeg = mne.io.read_raw_edf("myfile.edf", preload=True)  
sc = SleepChecker(raw_eeg, eeg_name=['C4', 'C3'], eog_name="HEOGR-HEOGL", ref_channel=["M1", "M2"])
# Return an array containing the predicted sleep phases among ['W', 'N1', 'N2', 'N3', 'R']
sleep_stages = sc.predict()  
# get the overall % of time asleep
sleep_percent = sc.get_tot_sleep_percentage(sleep_stages)  
# annotate the sleeping time spans directly on the raw data
raw_eeg = sc.annotate_data()  
```

# References

<a id="1">[1]</a>
Vallat, Raphael, and Matthew P. Walker. "An open-source, high-performance tool for automated sleep staging." Elife 10 
(2021). doi: https://doi.org/10.7554/eLife.70092
