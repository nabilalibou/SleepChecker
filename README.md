# SleepChecker: A Simplified Wrapper for Robust Sleep Stage Detection in EEG

SleepChecker simplifies the process of detecting sleep stages in EEG recordings. It wraps the powerful 
[yasa](https://raphaelvallat.com/yasa/build/html/index.html) [[1]](#1) SleepStaging module, providing a 
user-friendly interface for robust sleep detection.

Key Features:

- Automated Sleep Stage Classification: Combines predictions from the underlying algorithm for a single, reliable sleep stage label.
- Total Sleep Time: Calculates the total percentage of time spent asleep during the recording.
- Sleep Annotation: Annotates sleep segments directly onto your [MNE](https://mne.tools/stable/index.html) raw EEG data for easy visualization and analysis.

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
