import numpy as np
import yasa
import mne
import re


class SleepChecker:
    """
    Wrapper of the yasa SleepStaging module which is an automatic sleep staging algorithm.
    Parameters
    ----------
    raw_eeg: (MNE Raw object): An instance of MNE Raw.
    eeg_name: (str | list of str) EEG channels used for the sleep detection. Can be a channel name or list of channel
        name.
    eog_name: (str) EOG channel used for the sleep detection (add it only if the channel quality is correct).
    keepN1: (bool) Whether to consider N1 or not as it is the most misclassified class.
    ref_channel: (str | list of str) can be a channel name or a list of channel name used to construct a reference or
        'average' or 'REST'.
    """

    def __init__(self, raw_eeg, eeg_name="C4", eog_name=None, ref_channel="M1", keepN1=False):
        assert isinstance(raw_eeg, mne.io.BaseRaw)
        assert isinstance(eeg_name, str) or self.is_list_of_strings(eeg_name)
        assert isinstance(eog_name, (str, type(None)))
        assert isinstance(ref_channel, str) or self.is_list_of_strings(ref_channel)
        assert isinstance(keepN1, bool)

        if isinstance(eeg_name, str):
            eeg_name = [eeg_name]
        if isinstance(ref_channel, str) and ref_channel not in ["average", "REST"]:
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

    @staticmethod
    def which_hemisphere(chan_names):
        """
        Return a list of bool reflecting whether the channel names are from right or left hemisphere (10-20 system).
        Parameters
        ----------
        chan_names: (list of str) list of channel names.
        Returns
        -------
        is_right_hemisphere (list of bool) True if channel is from right hemisphere, False if channel is from left
            hemisphere.
        """
        try:
            chan_number = [int(re.search("[0-9]+", chan).group()) for chan in chan_names]
        except AttributeError as e:
            raise ValueError(
                f"{e}. Input channel names should belong to one of the 2 hemisphere according to the 10-20 "
                "system"
            )
        is_right_hemisphere = [not num & 1 for num in chan_number]
        return is_right_hemisphere

    @staticmethod
    def is_list_of_strings(lst):
        """
        Check if input object is a list of strings.
        Note: Should check for basestring instead of str since it's a common class from which both the str and unicode
        types inherit from. Checking only the str leaves out the unicode types.
        Parameters
        ----------
        lst: (object) input object to be checked.
        Returns
        -------
        (bool) Whether input variable is a list of strings or not.
        """
        if lst and isinstance(lst, list):
            return all(isinstance(elem, str) for elem in lst)
        else:
            return False

    def _combine_predictions(self, predictions):
        """
        Keep the resulting sleeping stages if predicted in every prediction array. It will check rows by index
        and compare to the first row then check the resulting boolean columns.
        Parameters
        ----------
        predictions: (array) All the predicted sleep stages.
        Returns
        -------
        res (array) The final array containing the sleep stages that have reached consensus among the various
            predictions.
        """
        check = np.all(predictions == predictions[0, :], axis=0)
        res = np.zeros((len(check)), dtype="U2")
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
        """
        Use the yasa LGBMClassifier to predict sleep stages for each 30-sec epoch of data using all the eeg, eog and
        reference channels provided. Return the final array of predicted sleep stage that have reached consensus among
        the various predictions.
        Returns
        -------
        sleep_stages (array) The predicted sleep stages.
        """
        ref_channel = ["M1", "M2"]
        OneRefOneHemisphere = False
        predictions = None
        is_rh = []
        if len(ref_channel) == 2:
            is_rh = self.which_hemisphere(ref_channel)
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
            is_rh = [int(val) for val in self.which_hemisphere(self.eeg_name)]
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
                            yasa.SleepStaging(
                                self.data, eeg_name=eeg_ch, eog_name=self.eog_name
                            ).predict(),
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
        """
        Verify the sleep stages provided.
        """
        if sleep_stages is None and self._sleep_stages is None:
            raise ValueError("Must call .predict before this function")
        if sleep_stages is None:
            sleep_stages = self._sleep_stages.copy()
        else:
            assert isinstance(sleep_stages, np.ndarray), "sleep_stages must be an numpy array"
            if not all(ele in ["W", "N1", "N2", "N3", "R"] for ele in list(sleep_stages)):
                raise ValueError(
                    "input sleep stage contains values not in ['W', 'N1', 'N2', 'N3', 'R']"
                )
        return sleep_stages

    def annotate_data(self, sleep_stages=None, SpecifyStage=False):
        """
        Annotate as 'bad' the time segments identified as sleep phases.
        Parameters
        ----------
        sleep_stages: (array) The predicted sleep stages.
        SpecifyStage: (bool) If True, annotations will specify sleep stages.
        Returns
        -------
        data (MNE Raw object): An instance of MNE Raw with the annotated sleep time segments.
        """
        sleep_stages = self._check_sleep_stages(sleep_stages)
        data_annot = self.data.annotations  # store current annotations to not erase them
        sleep_phases = []
        for i in range(len(sleep_stages)):
            if sleep_stages[i] != "W":
                self._sleep_onset.append(i * 30)
                sleep_phases.append(sleep_stages[i])

        description = ["bad"] * len(self._sleep_onset)
        if SpecifyStage:
            description = [": ".join(z) for z in zip(description, sleep_phases)]

        my_annot = mne.Annotations(
            onset=self._sleep_onset, duration=[30] * len(self._sleep_onset), description=description
        )
        return self.data.set_annotations(data_annot + my_annot)

    def get_tot_sleep_percentage(self, sleep_stages=None):
        """
        Return the total percentage of time asleep during an EEG recording session.
        Parameters
        ----------
        sleep_stages (array): Predicted sleep stages.
        Returns
        -------
        tot_sleep_percentage (float) percentage of time asleep.
        """
        sleep_stages = self._check_sleep_stages(sleep_stages)
        sleep_cnt = 0
        if sleep_stages[-1] != "W":
            if len(sleep_stages) > len(self.data.times) / 512 // 30:
                sleep_cnt += len(self.data.times) / self.data.info["sfreq"] % 30 / 30
            else:
                sleep_cnt += 1
        for i in range(len(sleep_stages)):
            if sleep_stages[i] != "W":
                sleep_cnt += 1
            self._tot_sleep_percentage = sleep_cnt / len(sleep_stages) * 100

        return self._tot_sleep_percentage
