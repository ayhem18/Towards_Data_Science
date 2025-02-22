import os
import numpy as np

# we know for a fact that we have 160 samples belonging to 20 subjects each having

# let's calculate the auto correlation of each time sequence in each of scan
from typing import Union, List
from scipy.signal import correlate2d, correlate

def autocorrelation_stats(scan: np.ndarray, aggregate:bool=True) -> Union[List, float]:	
	assert len(scan) == 10
	auto_correlations =  [float(correlate(scan[i:], scan[:-i])) for i in range(1, 6)]
	if aggregate:
		return np.mean(auto_correlations)
	return auto_correlations

def build_ac_pairs(scans: List[np.ndarray]) -> set:
	auto_corrs = np.zeros(shape=(len(scans), len(scans)))

	for i1, element1 in enumerate(scans):
		for i2, element2 in enumerate(scans):
			auto_corrs[i1][i2] = correlate2d(element1, element2, "valid").item()	
	
	# for each row, row[0] represents the closest index to scan[i] in terms of auto correlation
	# row[1] represents the same index
	paired_scans_by_ac = np.argsort(auto_corrs, axis=-1)[:, -2:]

	pairs = set()

	for i in range(len(scans)):
		assert paired_scans_by_ac[i, 1] == i, "check the code"
		closest_scan_index = paired_scans_by_ac[i, 0]
		if paired_scans_by_ac[closest_scan_index, 0] == i and (closest_scan_index, i) not in pairs:
			pairs.add((i, closest_scan_index)) 

	return pairs


def unified_segment_rep(scans: List[np.ndarray], pairs_indices: set) -> List[np.ndarray]:
	avg_segments = []
	for i1, i2 in pairs_indices:
		s1, s2  = scans[i1], scans[i2]
		if s1.shape != s2.shape:
			raise ValueError("Make sure the code is correct. found pairs with different shapes")
		avg_segments.append((s1 + s2) / 2)
	return avg_segments


def find_best_next_segment(seg_index: int, segments: List[np.ndarray]):
	max_ac_corr = -float('inf')
	best_index = None
	best_order = None

	for i in range(len(segments)):
		if i == seg_index: 
			continue

		other_seg = segments[i]
		# build the bigger sequence
		compound_seg1 = np.concatenate([segments[seg_index], other_seg], axis=0)
		compound_seg2 = np.concatenate([other_seg, segments[seg_index]], axis=0)

		assert compound_seg1.shape[0] == other_seg.shape[0] * 2 and compound_seg1.shape[1] == other_seg.shape[1], "make sure the concatenation is done vertically"
		assert compound_seg2.shape[0] == other_seg.shape[0] * 2 and compound_seg2.shape[1] == other_seg.shape[1], "make sure the concatenation is done vertically"

		# calcuate the auto correlation between compound, other and the given segment
		c11 = correlate2d(compound_seg1, np.concatenate([segments[seg_index], segments[seg_index]], axis=0), "valid").item()
		c12 = correlate2d(compound_seg1, np.concatenate([other_seg, other_seg], axis=0), "valid").item()
		c1 = (c11 + c12) / 2

		c21 = correlate2d(compound_seg2, segments[seg_index], "valid").item()
		c22 = correlate2d(compound_seg2, other_seg, "valid").item()
		c2 = (c21 + c22) / 2

		corr = max(c1, c2)

		if corr > max_ac_corr:
			corr = max_ac_corr 
			best_index = i
			best_order = [seg_index, i] if c1 > c2 else [i, seg_index]

	return best_index, best_order 


def find_consecutive_time_stamps(corrs: np.ndarray, k_closest:int):
	closest_groups = np.argsort(corrs, axis=-1)[:, -k_closest:-1]	

	possible_pairs = set()

	for i in range(len(corrs)):
		possible_neighbors = closest_groups[i]
		for pn in possible_neighbors:
			if i in closest_groups[pn]:
				possible_pairs.add((i, pn))

	return possible_pairs



from sklearn.decomposition import PCA

def estimate_consecutive_samples(seg1: np.ndarray, seg2: np.ndarray, eval_method: str) -> float:
	if eval_method not in ['avg', 'count']:
		raise NotImplementedError("eval_method non-implemented")

	assert seg1.shape == seg2.shape, "both segments same shape"
	c_seg1 = np.concatenate([seg1, seg2], axis=0)
	c_seg2 = np.concatenate([seg2, seg1], axis=0)

	assert c_seg1.shape[0] == 2 * seg1.shape[0] and c_seg1.shape[1] == seg1.shape[1], "concatenation correct"
	assert c_seg2.shape[0] == 2 * seg2.shape[0] and c_seg2.shape[1] == seg2.shape[1], "concatenation correct"

	if eval_method == 'count':
		pca1 = PCA(n_components=0.9)
		pca2 = PCA(n_components=0.9)
	else:
		pca1 = PCA(n_components=10)
		pca2 = PCA(n_components=10)

	pca1.fit(seg1)
	pca2.fit(seg2)

	# extract the explained ratio
	var_ratio1 = pca1.explained_variance_ratio_
	var_ratio2 = pca2.explained_variance_ratio_

	if eval_method == 'count':
		return min(len(var_ratio1), len(var_ratio2))

	return max(np.mean(var_ratio1), np.mean(var_ratio2))

if __name__ == '__main__':
	current_dir = os.path.dirname(os.path.realpath(__file__))
	# let's see how it goes
	npy_file_path = os.path.join(current_dir, 'ihb.npy')
	all_data = np.load(npy_file_path)


	g1 = [all_data[i] for i in range(all_data.shape[0]) if np.isnan(all_data[i]).sum() == 0]
	g2 = [all_data[i] for i in range(all_data.shape[0]) if np.isnan(all_data[i]).sum() == 460]

	g2_nan_distribution = np.zeros(shape=(len(g2), len(g2)))

	for i1, element1 in enumerate(g2):
		for i2, element2 in enumerate(g2):
			g2_nan_distribution[i1][i2] = all(np.sum(np.isnan(element1), axis=1) == np.sum(np.isnan(element2), axis=1))
	g2 = [scan[:, :-46] for scan in g2]

	g1_pairs, g2_pairs = build_ac_pairs(g1), build_ac_pairs(g2)
	avg_g1, avg_g2 = unified_segment_rep(g1, g1_pairs), unified_segment_rep(g2, g2_pairs)


	# find the pairs using the mean and variance
	feats_mean = np.concatenate([np.mean(x, axis=0, keepdims=True) for x in avg_g1])
	feats_var = np.concatenate([np.var(x, axis=0, keepdims=True) for x in avg_g1])

	# # compute correlation 
	# corr_feat_var = np.corrcoef(feats_var)
	# corr_feat_mean = np.corrcoef(feats_mean)

	# assert corr_feat_var.shape == (80, 80)
	# assert corr_feat_mean.shape == (80, 80)

	# pairs_with_var = find_consecutive_time_stamps(corr_feat_var, 2)
	# pairs_with_mean = find_consecutive_time_stamps(corr_feat_mean, 2)	

	estimate = estimate_consecutive_samples(avg_g1[0], avg_g1[1], eval_method='count')	


    