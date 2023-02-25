from thinkdsp import Wave
from thinkdsp import Signal
import numpy as np 
from thinkdsp import read_wave
from copy import copy
import random
from random import sample
from thinkdsp import SinSignal, CosSignal, SquareSignal, SumSignal, TriangleSignal, SawtoothSignal
from math import ceil
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from itertools import chain

# function to display the resulting waves in a clear way
from IPython.display import display


# let's set the random module seed for reproducibility
np.random.seed(69)
random.seed(69)

# define the signals used in the sinusoidal modeling
SIGNALS = [SinSignal(), CosSignal(), SquareSignal(), TriangleSignal()]


# PREPROCESSING FUNCTIONS: FILTERING, SPLITTING WAVES, stretching and some statistics...

def filter_wave(wave: Wave):
    """
    The function filters the wave by removing any sequence of timestamps satisfying simultaneously these two conditions:
        1. the sequence's length is at least 5% the total wave's length
        2. The amplitude of each value in this sequence is less than 5% of the maximum amplitude
    Returns:
        Wave: the filtered wave
    """

    # first define the threshold
    min_value = max(abs(wave.ys)) * 0.05
    min_removable_length = int(len(wave.ys) * 0.05)

    # copy the amplitudes    
    new_ys = copy(wave.ys)

    # variables that set the boundaries of removable sequences
    start, end = 0, 0
    for i, y in enumerate(wave.ys):
        if np.abs(y) > min_value:  # which means this value should not be removed
            # first check if the length of the removable sequence is long enough
            if end - start >= min_removable_length:
                # set all the values in this range to 0
                new_ys[start: end] = 0
            # now time to update the start value
            start = i + 1
        else:  # which means the value is too low and can be removed
            end = i + 1

    # check if there is a removable sequence at the end
    if end - start >= min_removable_length:
        new_ys[start:end] = 0

    new_ys = np.asarray([y for y in new_ys if y != 0])

    # the new wave should be of the same framerate
    return Wave(ys=new_ys, framerate=wave.framerate)


def set_timestamps(waves: list[Wave]):
    # let's now modify the timestamps
    durations = [0]
    durations.extend([s.duration for s in waves[:-1]])
    # 
    new_waves = [copy(w) for w in waves]

    for i, d in zip(range(len(waves)), np.cumsum(durations)):
        # the idea is to first shift the timestamps back to start from zero
        # and then shift them back in the opposite direction by the durations' accumulative sum         
        new_waves[i].ts = new_waves[i].ts - min(new_waves[i].ts) + d
    return new_waves


def split_wave(wave: Wave, part_duration: float = 0.5) -> list[Wave]:
    """
    this function breaks down a given wave into several intermediate waves with duration of at most part_duration,
    filters each part and then modify the values of the timestamps such that the values of the timestamps are disjoint.

    Args:
        wave (Wave): The wave to break down
        part_duration (float, optional): The maximum length of . Defaults to 0.5.

    Returns:
        list[Wave]: a list of filtered waves that can be combined into a single wave  
    """

    # make sure the part duration is at least 10ms and at most 1s
    part_duration = max(part_duration, 0.05)
    part_duration = min(part_duration, 1)

    num_splits = ceil(wave.duration / part_duration)

    # the first part is to split the wave
    splits = [wave.segment(start=i * part_duration, duration=part_duration) for i in range(num_splits - 1)]
    # check the length of the last split
    last_split_duration = wave.duration - (num_splits - 1) * part_duration   

    # if the last split is too short, plug it to the last one
    if last_split_duration < part_duration:
        # this block will be executed only if the number of splits is at least 2. 
        try:
            splits.pop()
        except IndexError:
            pass
        splits.append(wave.segment(start=(num_splits - 2) * part_duration))
    else:
        splits.append(wave.segment(start=(num_splits - 1) * part_duration))



    # let's filter our splits
    splits = [filter_wave(s) for s in splits]

    # let's now modify the timestamps
    # durations = [0]
    # durations.extend([s.duration for s in splits[:-1]])
    # # 
    # for i, d in zip(range(len(splits)), np.cumsum(durations)):
    #     splits[i].ts += d

    splits = set_timestamps(splits)

    # make sure the timestamps are adjusted correctly
    for i in range(len(splits) - 1):
        assert max(splits[i].ts) <= min(splits[i + 1].ts)

    return splits


def modify(wave: Wave, factor: float) -> Wave:
    """This function modifies the ryth of wave by the given factor
    if factor > 1: the wave will be stretched. Otherwise, it will be speed up
        
    Args:
        wave (Wave): the initial factor
        factor (float): the modification's factor: 

    Returns:
        Wave: the modified wave
    """

    new_wave = wave.copy()
    new_wave.ts *= factor
    new_wave.framerate /= factor
    return new_wave


# given a set of waves (or audio files), define a high and low pass filters: find the most expressive frequencies 

def find_passes(waves, save:str, is_file:bool=True) -> tuple[float, float]:
    if is_file:
        waves = [read_wave(file_name_wave) for file_name_wave in waves]
    # create a counter: where each values will be associated with its amplitude
    # for each wave: make the spectrum, convert
    freqs_amp_dict = {}

    for w in waves:
        # first make the spectrum
        spec = w.make_spectrum()
        # now iterate through the frequencies, round each to its integer part and add its amplitude
        for freq, amp in zip(list(spec.fs), list(spec.hs)):
            f = int(freq)
            # add the new frequency to the dictionary
            if f not in freqs_amp_dict:
                freqs_amp_dict[f] = 0
            # add the associated amplitude either way
            freqs_amp_dict[f] += abs(amp)

    # don't forget to divide the value associated with each frequency by the number of waves
    # to obtain the average amplitude associated with each frequency by wave
    for freq, amp in freqs_amp_dict.items():
        freqs_amp_dict[freq] /= len(waves)

    # to convert the frequencies into a random variable, each frequency should be repeated [total_amplitude]
    freq_random_variable = [ [freq] * ceil(amp) for freq, amp in freqs_amp_dict.items()]
    # flatten the list
    freq_random_variable = list(chain(*freq_random_variable))

    # displaying the distribution wouldn't hurt
    sns.displot(data=pd.DataFrame(data=freq_random_variable, columns=['fs']), x='fs')
    plt.show()

    # let's determine the percentiles
    q1, q3 = np.percentile(freq_random_variable, [25, 75])
    iqr = q3 - q1

    upper_freq, lower_freq = q3 + 1.5 * iqr,  q1 - 1.5 * iqr

    # let's calculate the mean and standard deviation of the amplitudes
    # amp_mean = np.mean(freq_random_variable)
    # amp_std = np.std(freq_random_variable)
    # lower_freq, upper_freq = amp_mean - 2 * amp_std, amp_mean + amp_std * 2

    min_freq = None  # default values of low and high pass arguments in the approximation functions
    max_freq = None


    for f in freqs_amp_dict.keys():
        if lower_freq <= f <= upper_freq:
            # update the value of the minimal frequency
            min_freq = f if min_freq is None else min(f, min_freq)
            # update the values of the maximal frequency
            max_freq = f if max_freq is None else max(f, max_freq)
            

    # find the frequencies associated with minimum and maximum amplitudes within the accepted range
    # for f, amp in freqs_amp_dict.items():
        
    #     if lower_amp <= amp <= upper_amp:
    #         # update the value of the minimal frequency
    #         min_freq = f if min_freq is None else min(f, min_freq)
    #         # update the values of the maximal frequency
    #         max_freq = f if max_freq is None else max(f, max_freq)


    # make sure to save the results
    if save is not None:
        # save is expected to be a file location relative to the file
        # where these values will be saved
        # if the file doesn't have an extension add .txt
        try:
            if '.' not in save[-5:]:
                save += '.txt'
        except IndexError:
            save += '.txt'

        path = os.path.join(os.getcwd(), save)
        with open(path, 'w') as f:
            f.write(f"The low pass value: {min_freq}\n")
            f.write(f"The high pass value: {max_freq}\n")

    return min_freq, max_freq


# MODELING FUNCTIONS: APPROXIMATING AN INITIAL WAVE USING BASIC SIGNALS

def find_alpha_one_signal(wave: Wave, signal: Signal, k=1500, sample_portion=0.75, low_pass: float = None,
                          high_pass: float = None, peek_only=True):
    """_summary_

    Args: wave (Wave): The signal to model signal (Signal): the basic signal k (int, optional): the value of k and
    peek_only determine the number of frequencies (and thus linear coefficients) considered sample_portion (float,
    optional): the fraction of the timestamps to consider in the modeling process. Defaults to 0.75.: low_pass (
    float, optional): the lower threshold to apply on the original wave. Defaults to None. high_pass (float,
    optional): the lower threshold to apply on the original wave. Defaults to None. peek_only (bool, optional): if
    True: the number of frequencies is the value of the argument 'k'. Otherwise, the number of frequencies is: k + k
    / 2 + k / 4. Defaults to True

    Returns:
        tuple: frequencies and their corresponding coefficients
    """

    # make the spectrum of the wave
    spec = wave.make_spectrum()

    # apply the filter if the corresponding paramter is passed
    if low_pass is not None:
        spec.low_pass(low_pass)

    if high_pass is not None:
        spec.high_pass(high_pass)

    # extract the top k frequencies, with their amplitudes
    _, freqs = list(map(list, zip(*spec.peaks()[:k])))

    # if the peek_
    if not peek_only:
        # extract k / 2 frequencies in the middle range 
        mid_point = int(len(freqs) / 2)
        _, more_freqs = list(map(list, zip(*spec.peaks()[mid_point - int(k / 2): mid_point + int(k / 2)])))
        freqs.extend(more_freqs)

        # extract the k / 4 frequencies with the lowest amplitudes
        _, more_freqs = list(map(list, zip(*spec.peaks()[-int(k / 4):])))
        freqs.extend(more_freqs)
        # now the 'freqs' variable contain all the different frequencies 

    # before proceeding with sampling timestamps
    # let's map each t to the corresponding value
    time_value_map = dict(zip(wave.ts, wave.ys))

    # take a random sample out of the timestamps
    sample_size = int(sample_portion * len(wave.ts))
    # extract the timestamps and sort them
    ts = sorted(sample(list(wave.ts.reshape(-1, )), sample_size))

    # calculate T
    T = np.asarray([time_value_map[t] for t in ts]).reshape(-1, 1)

    # calculate F
    # create the empty array with the shape
    F = np.empty([len(freqs), sample_size])

    for index, f in enumerate(freqs):
        # set the frequency
        signal.freq = f
        # add the evaluation with the specific frequency
        F[index] = signal.evaluate(ts)
    # transpose F
    F = F.T

    # print(F.shape)
    # print(T.shape)

    # find alpha with linear regression    
    try:
        alpha = np.linalg.lstsq(F, T, rcond=None)[0]
    except Exception as e:
        print(e)
        alpha = None
    return freqs, alpha


def approximate_one_signal(wave: Wave, signal: Signal, n_trials: int = 5, k: int = 1500, sample_portion: float = 0.75,
                           low_pass: float = None, high_pass: float = None, peek_only=True, display_final=False):
    """This function given an initial wave, generates a new wave that approximates the initial one based on the
    passed basic signal

    Args:
        wave (Wave): the wave to model 
        signal (Signal): the basic signal used in modeling
        n_trials (int, optional): The final values of the linear coefficients is set as a mean of the coefficients
        found through "n_trail" approximations. Defaults to 5.
        k (int, optional): the value of k and peek_only determine the number of frequencies
        (and thus linear coefficients) considered
        sample_portion (float, optional): the fraction of the timestamps to consider in the modeling process.
         Defaults to 0.75.:
        low_pass (float, optional): the lower treshold to apply on the original wave. Defaults to None.
        high_pass (float, optional): the lower treshold to apply on the original wave. Defaults to None.
        peek_only (bool, optional): if True: the number of frequencies is the value of the argument 'k'.
        Otherwise, the number of frequencies is: k + k / 2 + k / 4. Defaults to True

    Returns:
        Wave: the approximation Wave
    """

    results = [
        find_alpha_one_signal(wave, signal, k=k, sample_portion=sample_portion, low_pass=low_pass, high_pass=high_pass,
                              peek_only=peek_only) for _ in range(n_trials)]
    # extract the frequencies
    freqs = results[0][0]
    # extract the alpha values from the results
    alphas = np.asarray([r[1] for r in results if r[1] is not None])  # filter the None values if any
    # final coefficients as the mean
    alpha = np.mean(alphas, axis=0).reshape(-1, )

    # time to create the resulting signal    
    approx_signal = SumSignal(*[SinSignal(freq=f, amp=a, offset=0) for f, a in zip(freqs, alpha)])
    approx_wave = approx_signal.make_wave(duration=wave.duration, start=wave.start, framerate=wave.framerate)
    
    # display the result if the corresponding argument is set to True
    if display_final:
        print("The original wave")
        display(wave.make_audio())
        print(f"The approximated wave with signal {signal}")
        display(approx_wave.make_audio())
        print()

    # return the wave and correlation score with the original wave
    corr = wave.corr(approx_wave)
    return approx_wave, corr


def approximate_multiple_signals(wave: Wave, signals: list[Signal], n_trials=2, k: int = 1500,
                                 sample_portion: float = 0.75, low_pass: int = None, high_pass: int = None,
                                 peek_only=True, min_correlation=0.4, display_final=True, display_all=False):
    """_summary_

    Args: 
    wave (Wave): the wave to model signals (list[Signal]): a list of basic signals used in modeling 
    
    n_trials (int, optional): The final values of the linear coefficients is set as a mean of the coefficients found through
    "n_trail" approximations. Defaults to 5. 
    
    k (int, optional): the value of k and peek_only determine the number of
    frequencies (and thus linear coefficients) considered 
    
    sample_portion (float, optional): the fraction of the
    timestamps to consider in the modeling process. Defaults to 0.75.
    
    : low_pass (float, optional): the lower treshold
    to apply on the original wave. Defaults to None.

    high_pass (float, optional): the lower treshold to apply on the
    original wave. Defaults to None. 
    
    peek_only (bool, optional): if True: the number of frequencies is the value of
    the argument 'k'. Otherwise, the number of frequencies is: k + k / 2 + k / 4. Defaults to True

    Returns:
        Wave: the approximation Wave
    """

    # for each of the passed signal we approximate the wave to that signals
    results = [approximate_one_signal(wave, s, n_trials=n_trials, k=k, sample_portion=sample_portion, low_pass=low_pass,
                                      high_pass=high_pass, peek_only=peek_only, display_final=display_all) for s in signals]

    # consider only the waves with at least a correlation of more than 0.6 (absolute value)
    approx_waves = [result[0] for result in results if np.abs(result[1]) >= min_correlation]

    # sometimes only one wave satisfies the correlation condition:  
    # thus the final approximation should be this wave
    # without further overhead

    if len(approx_waves) == 1:
        # find the wave and its correlation
        r = [r for r in results if np.abs(r[1]) >= min_correlation][0]
        return  r[0], r[1]

    try:
        assert approx_waves
    except AssertionError:
        print(f"MAKE SURE TO CHECK THE PARAMETERS. NONE OF THE APPROXIMATIONS HAVE A CORRELATION HIGHER THAT {min_correlation}")
        return None, None

    times = [aw.ts for aw in approx_waves]

    # check if the timestamps are the same accross different approximate waves
    for t in times:
        assert np.array_equal(t, times[0])

    # construct F
    F = np.squeeze(np.asarray([a_w.ys for a_w in approx_waves]).T)
    
    # construct T
    T = np.asarray(wave.ys).reshape(-1, 1)
    # find alpha: the coefficients
    alpha = np.asarray(np.linalg.lstsq(F, T, rcond=None)[0]).reshape(-1, )
    # the resulting wave 
    final_approx = copy(wave)
    # set its ys parameters to the matrix product of alpha and F
    final_approx.ys = np.dot(F, alpha)

    # display the final result if the corresponding argument is set accordingly.
    if display_final:
        print("The original wave")
        display(wave.make_audio())
        print("The approximated wave with multiple signals")
        display(final_approx.make_audio())
        print()

    corr = wave.corr(final_approx)
    return final_approx, corr


# COMBINING WAVES


def merge_2_waves(wave1: Wave, wave2: Wave, overlap_size: float = 0.25) -> Wave:
    """This function merges two waves into a single wave with the resulting timestamps and signal values  as the
    concatenation of both waves and signal values of the passed waves

    Args: wave1 (Wave): the first wave (in chronological order) wave2 (Wave): the second wave (in chronological
    order) overlap_size (float, optional): the size in fraction of the number timestamps to be considered as overlap.
    Defaults to 0.25.

    Returns:
        Wave: the merged wave
    """
    # work on copies of the passed waves 
    w1 = wave1.copy()
    w2 = wave2.copy()

    # preprocess the waves
    w1.normalize()
    w1.apodize()
    w2.normalize()
    w2.apodize()

    overlap = int(min(len(w1.ts), len(w2.ts)) * overlap_size)

    proportions = np.linspace(0, 0.5, overlap)

    # change the values of the last portion of the first wave to gradually integrate the 2nd wave
    w1.ys[-overlap:] = (1 - proportions) * (w1.ys[-overlap:]) + proportions * (w2.ys[:overlap])

    # change the values of the first overlap portion of the 2nd wave to gradually integrate the 1st wave
    w2.ys[:overlap] = proportions * w1.ys[-overlap:] + (1 - proportions) * (w2.ys[:overlap])

    # Concatenate the signals
    ys = np.concatenate([w1.ys, w2.ys], axis=None)
    # concatenate the timestamsp
    ts = np.concatenate([w1.ts, w2.ts], axis=None)

    # Create a new wave object
    merged_wave = Wave(ys=ys, ts=ts)
    return merged_wave


def merge_waves(waves: list[Wave], hamming=True):
    """This function merges a number of waves to produce a final wave satisfying the following conditions:
    1. the transition between any two consecutive waves is smooth
    2. the final array of signal values is the concatenation of each individual array
    3. the final array of timestamps is the concatenation of each individual array
    4. the duration of the merged wave is the sum of the durations of the respective waves 

    Args:
        waves (list[Wave]): the list of waves to merge
        hamming (bool, optional): whether to applying hamming to each individual wave. Defaults to True.

    Returns:
        _type_: _description_
    """
    # make sure the waves' timestamps are disjoint

    for i in range(0, len(waves) - 1):
        assert max(waves[i].ts) <= min(waves[i + 1].ts)

    # first determine the framerate used for the merged wave
    assert len(waves) >= 2
    result_wave = waves[0]

    if hamming:
        result_wave.hamming()

    for w in waves[1:]:
        if hamming:
            w.hamming()
        result_wave = merge_2_waves(result_wave, w)

    # set the correct frame rate: to make sure the resulting duration is as expected
    new_framerate = int(len(result_wave.ys) / (max(result_wave.ts) - min(result_wave.ts)))
    result_wave.framerate = new_framerate

    return result_wave


# APPROXIMATING A WAVE

def approximate_wave(wave: Wave, low_pass:int=None, high_pass:int=None, display_final=True, display_all=False, display_result=True) -> Wave:
    """This function approximates a given wave using the steps coded above:
    1. split the wave into smaller pieces
    2. approximate each piece using linear combination of basic signals
    3. merge the approximations into a final approximation of the initial wave

    Args:
        wave (Wave): the wave to approximate
        low_pass (int, optional): a low pass filter if any. Defaults to None.
        high_pass (int, optional): a high pass filter if any. Defaults to None.

    Returns:
        Wave: the resulting approximated wave
    """
    global SIGNALS
    # split the wave into smaller pieces
    intermediate_waves = split_wave(wave, part_duration=0.5)
    # approximate each piece
    results = [
        approximate_multiple_signals(iw, SIGNALS, n_trials=1, low_pass=low_pass, high_pass=high_pass, peek_only=True, 
        display_final=display_final, display_all=display_all)
        for iw in intermediate_waves]
    # combine the pieces to generate the final wave
    approx_waves = [r[0] for r in results if r[0] is not None]

    # before merging the approximation waves, let's set their timestamps
    approx_waves = set_timestamps(approx_waves)
    result =  merge_waves(approx_waves, hamming=True) if len(approx_waves) >= 2 else approx_waves[0]

    if display_result:
        print("The COMPLETE ORIGINAL WAVE ")
        display(wave.make_audio())
        print("The final result")
        display(result.make_audio())

    return result
