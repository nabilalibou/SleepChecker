import numpy as np
import yasa
import mne
import re
from checks import is_list_of_strings


class SleepChecker:
    """
    Wrapper object for the yasa

    Function using the yasa package to detect sleep stages in raw eeg data. Each detection are done on 30-sec epochs.
    It returns a list of sleep stages detected (W, N1, N2, N3, R) and can optionally annotate the resulting stages on
    the input MNE raw.
    :param raw_eeg:
    :param eeg_name: (str of list of str) EEG channels used for the sleep detection. Can be a channel name or list
        of channel name.
    :param eog_name: (str) EOG channel used for the sleep detection (add it only if the channel quality is correct).
    :param keepN1: (bool) Whether to consider N1 or not as it is the most misclassified class.
    :param ref_channel: (str of list of str) can be a channel name or a list of channel name used to construct a
        reference or 'average' or 'REST'.
    :return:
    """
    def __init__(
        self, raw_eeg, eeg_name="C4", eog_name=None, ref_channel="M1", keepN1=False
    ):
        assert isinstance(raw_eeg, mne.io.BaseRaw)
        assert isinstance(eeg_name, str) or is_list_of_strings(eeg_name)
        assert isinstance(eog_name, (str, type(None)))
        assert isinstance(ref_channel, str) or is_list_of_strings(ref_channel)
        assert isinstance(keepN1, bool)

        if isinstance(eeg_name, str):
            eeg_name = [eeg_name]
        if isinstance(ref_channel, str) and ref_channel not in ['average', 'REST']:
            ref_channel = [ref_channel]

        ch_names = eeg_name.copy()
        if isinstance(ref_channel, list):
            ch_names += ref_channel
        if eog_name:
            ch_names.append(eog_name)
        for c in ch_names:
            assert c in raw_eeg.ch_names, f"chan {c} does not exist"

        self.data = raw_eeg
        self.eeg_name = eeg_name
        self.eog_name = eog_name
        self.ref_channel = ref_channel
        self.keepN1 = keepN1

        self._sleep_stages = None
        self._tot_sleep_percentage = None
        self._sleep_onset = []

    def _which_hemisphere(chan_names):
        """
        Return list of bool reflecting whether the channel name are from right hemisphere or right (10-20 system).
        :param chan_names:
        :return: is_right_hemisphere (list of bool) True if channel is from right hemisphere, False if channel is from left
            hemisphere.
        """
        try:
            chan_number = [int(re.search("[0-9]+", chan).group()) for chan in chan_names]
        except AttributeError as e:
            raise ValueError(
                f"{e}. input channel names should belong to one of the 2 hemisphere according to the 10-20 "
                "system")
        is_right_hemisphere = [not num & 1 for num in chan_number]
        return is_right_hemisphere

    def _combine_predictions(self, predictions):
        """
        Keep the resulting sleeping stages if predicted in every prediction array.
        It will check rows by index according to the 1st one then check the resulting boolean columns.
        :param predictions: (array)
        :return: res (array)
        """
        check = np.all(predictions == predictions[0, :], axis=0)
        res = np.zeros((len(check)),  dtype='U2')
        for i in range(len(check)):
            if check[i]:
                if predictions[0][i] == "N1" or (not self.keepN1 and predictions[0][i] == "N1"):
                    res[i] = "W"
                else:
                    res[i] = predictions[0][i]
            else:
                res[i] = "W"
        return res

    def predict(self):

        ref_channel = ["M1", "M2"]
        OneRefOneHemisphere = False
        predictions = None
        is_rh = []
        if len(ref_channel) == 2:
            is_rh = _which_hemisphere(ref_channel)
            if is_rh[0] != is_rh[1]:
                OneRefOneHemisphere = True
        if OneRefOneHemisphere:
            raw_rh = self.data.copy()
            raw_lh = self.data.copy()
            raw_rh.set_eeg_reference(
                ref_channels=[ref_channel[np.where(is_rh)[0][0]]], ch_type="eeg"
            )
            raw_lh.set_eeg_reference(
                ref_channels=[ref_channel[~np.where(is_rh)[0][0]]], ch_type="eeg"
            )
            is_rh = [int(val) for val in _which_hemisphere(self.eeg_name)]
            for i, eeg_ch in enumerate(self.eeg_name):
                if i:
                    predictions = np.vstack(
                        (
                            predictions,
                            yasa.SleepStaging(
                                [raw_rh, raw_lh][is_rh[i]], eeg_name=eeg_ch, eog_name=self.eog_name
                            ).predict(),
                        )
                    )
                else:
                    predictions = yasa.SleepStaging(
                        [raw_rh, raw_lh][is_rh[i]], eeg_name=eeg_ch, eog_name=self.eog_name
                    ).predict()
        else:
            self.data.set_eeg_reference(ref_channels=ref_channel, ch_type="eeg")
            for i, eeg_ch in enumerate(self.eeg_name):
                if i:
                    predictions = np.vstack(
                        (
                            predictions,
                            yasa.SleepStaging(self.data, eeg_name=eeg_ch, eog_name=self.eog_name).predict(),
                        )
                    )
                else:
                    predictions = yasa.SleepStaging(
                        self.data, eeg_name=eeg_ch, eog_name=self.eog_name
                    ).predict()
        if predictions.ndim == 2:
            self._sleep_stages = self._combine_predictions(predictions)
        return self._sleep_stages

    def _check_sleep_stages(self, sleep_stages):
        if sleep_stages is None and self._sleep_stages is None:
            raise ValueError("Must call .predict before this function")
        if sleep_stages is None:
            sleep_stages = self._sleep_stages.copy()
        else:
            assert isinstance(sleep_stages, np.ndarray), "sleep_stages must be an numpy array"
            if not all(ele in ['W', 'N1', 'N2', 'N3', 'R'] for ele in list(sleep_stages)):
                raise ValueError("input sleep stage contains values not in ['W', 'N1', 'N2', 'N3', 'R']")
        return sleep_stages

    def annotate_data(self, sleep_stages=None, SpecifyStage=False):
        """
        Annotate as 'bad' the time segment identified as sleep phases.
        :param sleep_stages:
        :param SpecifyStage: (bool) If True, annotations will specify sleep stages.
        :return:
        """
        sleep_stages = self._check_sleep_stages(sleep_stages)
        sleep_phases = []
        for i in range(len(sleep_stages)):
            if sleep_stages[i] != 'W':
                self._sleep_onset.append(i*30)
                sleep_phases.append(sleep_stages[i])

        description = ['bad'] * len(self._sleep_onset)
        if SpecifyStage:
            description = [': '.join(z) for z in zip(description, sleep_phases)]

        my_annot = mne.Annotations(
            onset=self._sleep_onset,
            duration=[30] * len(self._sleep_onset),
            description=description
        )
        return self.data.set_annotations(my_annot)

    def get_tot_sleep_percentage(self, sleep_stages=None):
        """
        Return the total percentage of time asleep.
        :return: self._tot_sleep_percentage (float) percentage of time asleep according to sleep stages and raw data.
        """
        sleep_stages = self._check_sleep_stages(sleep_stages)
        sleep_cnt = 0
        if sleep_stages[-1] != 'W':
            if len(sleep_stages) > len(self.data.times) / 512 // 30:
                sleep_cnt += len(self.data.times) / self.data.info['sfreq'] % 30 / 30
            else:
                sleep_cnt += 1
        for i in range(len(sleep_stages)):
            if sleep_stages[i] != 'W':
                sleep_cnt += 1
            self._tot_sleep_percentage = sleep_cnt / len(sleep_stages) * 100

        return self._tot_sleep_percentage
