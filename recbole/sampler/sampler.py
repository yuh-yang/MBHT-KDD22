# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

# UPDATE
# @Time   : 2021/7/23, 2020/8/31, 2020/10/6, 2020/9/18, 2021/3/19
# @Author : Xingyu Pan, Kaiyuan Li, Yupeng Hou, Yushuo Chen, Zhichao Feng
# @email  : xy_pan@foxmail.com, tsotfsk@outlook.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, fzcbupt@gmail.com

"""
recbole.sampler
########################
"""

import copy

import numpy as np
from numpy.random import sample
import torch
from collections import Counter


class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.
    Args:
        distribution (str): The string of distribution, which is used for subclass.
    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution):
        self.distribution = ''
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        """Set the distribution of sampler.
        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == 'popularity':
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.
        Args:
            sample_num (int): the number of samples.
        
        Returns:
            sample_list (np.array): a list of samples. 
        """
        raise NotImplementedError('Method [_uni_sampling] should be implemented')

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()
        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError('Method [_get_candidates_list] should be implemented')

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling.
        """
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        del self.prob[0]
        keys = list(self.prob.keys())
        self.keys = sorted(keys, key=(lambda x: self.prob[x]), reverse=True)[:200]

        if self.dataset_name == "ijcai":
            # ijcai
            self.keys = [
                73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355]
        elif self.dataset_name == "retail":
            # retail
            self.keys = [
                101, 11, 14, 493, 163, 593, 1464, 12, 297, 123, 754, 790, 243, 250, 508, 673, 1161, 523, 41, 561, 2126, 196, 1499, 1093, 1138, 1197, 745, 1431, 682, 1567, 440, 1604, 145, 1109, 2146, 209, 2360, 426, 1756, 46, 1906, 520, 3956, 447, 1593, 1119, 894, 2561, 381, 939, 213, 1343, 733, 554, 2389, 1191, 1330, 1264, 2466, 2072, 1024, 2015, 739, 144, 1004, 314, 1868, 3276, 1184, 866, 1020, 2940, 5966, 3805, 221, 11333, 5081, 685, 87, 2458, 415, 669, 1336, 3419, 2758, 2300, 1681, 2876, 2612, 2405, 585, 702, 3876, 1416, 466, 7628, 572, 3385, 220, 772]
        else:
            # tmall
            self.keys = [
                2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701]
        self.alias = self.prob.copy()

        # large_q = []
        # small_q = []

        # for i in self.prob:
        #     self.alias[i] = -1
        #     self.prob[i] = self.prob[i] / len(candidates_list) * len(self.prob)
        #     if self.prob[i] > 1:
        #         large_q.append(i)
        #     elif self.prob[i] < 1:
        #         small_q.append(i)

        # while len(large_q) != 0 and len(small_q) != 0:
        #     l = large_q.pop(0)
        #     s = small_q.pop(0)
        #     self.alias[s] = l
        #     self.prob[l] = self.prob[l] - (1 - self.prob[s])
        #     if self.prob[l] < 1:
        #         small_q.append(l)
        #     elif self.prob[l] > 1:
        #         large_q.append(l)

    def _pop_sampling_static(self, sample_num):
        final_random_list = self.keys[:sample_num]
        return np.array(final_random_list)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.
        Args:
            sample_num (int): the number of samples.
        
        Returns:
            sample_list (np.array): a list of samples. 
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num, uni=False):
        """Sampling [sample_num] item_ids.
        
        Args:
            sample_num (int): the number of samples.
        
        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if uni:
            return self._uni_sampling(sample_num)
        if self.distribution == 'uniform':
            return self._uni_sampling(sample_num)
        elif self.distribution == 'popularity':
            return self._pop_sampling_static(sample_num)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError('Method [get_used_ids] should be implemented')

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.
        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.
        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list), uni=True)
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list), uni=True)
                check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        return torch.tensor(value_ids)


class Sampler(AbstractSampler):
    """:class:`Sampler` is used to sample negative items for each input user. In order to avoid positive items
    in train-phase to be sampled in valid-phase, and positive items in train-phase or valid-phase to be sampled
    in test-phase, we need to input the datasets of all phases for pre-processing. And, before using this sampler,
    it is needed to call :meth:`set_phase` to get the sampler of corresponding phase.
    Args:
        phases (str or list of str): All the phases of input.
        datasets (Dataset or list of Dataset): All the dataset for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, datasets, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError(f'Phases {phases} and datasets {datasets} should have the same length.')

        self.phases = phases
        self.datasets = datasets
        self.dataset_name = datasets[0].dataset_name

        self.uid_field = datasets[0].uid_field
        self.iid_field = datasets[0].iid_field

        self.user_num = datasets[0].user_num
        self.item_num = datasets[0].item_num

        super().__init__(distribution=distribution)

    def _get_candidates_list(self):
        candidates_list = []
        for dataset in self.datasets:
            candidates_list.extend(dataset.inter_feat['item_id_list'].numpy().flatten())
        return candidates_list

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.
        Args:
            phase (str): The phase of new sampler.
        Returns:
            Sampler: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
            is set to the value of corresponding phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.
        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.
        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')


class KGSampler(AbstractSampler):
    """:class:`KGSampler` is used to sample negative entities in a knowledge graph.
    Args:
        dataset (Dataset): The knowledge graph dataset, which contains triplets in a knowledge graph.
        distribution (str, optional): Distribution of the negative entities. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.entity_num, sample_num)

    def _get_candidates_list(self):
        return list(self.hid_list) + list(self.tid_list)

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used entity_ids is the same as tail_entity_ids in knowledge graph.
            Index is head_entity_id, and element is a set of tail_entity_ids.
        """
        used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)

        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                raise ValueError(
                    'Some head entities have relation with all entities, '
                    'which we can not sample negative entities for them.'
                )
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        """Sampling by head_entity_ids.
        Args:
            head_entity_ids (numpy.ndarray or list): Input head_entity_ids.
            num (int, optional): Number of sampled entity_ids for each head_entity_id. Defaults to ``1``.
        Returns:
            torch.tensor: Sampled entity_ids.
            entity_ids[0], entity_ids[len(head_entity_ids)], entity_ids[len(head_entity_ids) * 2], ...,
            entity_id[len(head_entity_ids) * (num - 1)] is sampled for head_entity_ids[0];
            entity_ids[1], entity_ids[len(head_entity_ids) + 1], entity_ids[len(head_entity_ids) * 2 + 1], ...,
            entity_id[len(head_entity_ids) * (num - 1) + 1] is sampled for head_entity_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(head_entity_ids, num)
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError(f'head_entity_id [{head_entity_id}] not exist.')


class RepeatableSampler(AbstractSampler):
    """:class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
    :class:`Sampler` is it can only sampling the items that have not appeared at all phases.
    Args:
        phases (str or list of str): All the phases of input.
        dataset (Dataset): The union of all datasets for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, dataset, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return np.array([set() for _ in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.
        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.
        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            self.used_ids = np.array([{i} for i in item_ids])
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.
        Args:
            phase (str): The phase of new sampler.
        Returns:
            Sampler: the copy of this sampler, and :attr:`phase` is set the same as input phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler


class SeqSampler(AbstractSampler):
    """:class:`SeqSampler` is used to sample negative item sequence.
        Args:
            datasets (Dataset or list of Dataset): All the dataset for each phase.
            distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):
        pass

    def sample_neg_sequence(self, pos_sequence):
        """For each moment, sampling one item from all the items except the one the user clicked on at that moment.
        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.
        Returns:
            torch.tensor : all users' negative item history sequence.
        """
        total_num = len(pos_sequence)
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list] = self.sampling(len(check_list))
            check_index = np.where(value_ids[check_list] == pos_sequence[check_list])
            check_list = check_list[check_index]

        return torch.tensor(value_ids)