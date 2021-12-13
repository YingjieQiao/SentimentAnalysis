import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from datautils import build_vocab, encode_data, get_test_data, get_train_data


class SecondOrderHiddenMarkovModel:
    def __init__(self, train_filepath: str = None):
        if train_filepath is not None:
            self._init_module(train_filepath)

    def _init_module(self, train_filepath: str):
        self.train_filepath = Path(train_filepath)
        self.tokens, self.labels = get_train_data(
            self.train_filepath,
            add_start_stop_states=False,
        )
        self.token_to_idx: Dict[str, int] = build_vocab(self.tokens, use_unk=True)
        self.label_to_idx: Dict[str, int] = build_vocab(self.labels, use_unk=False)
        self.idx_to_token: Dict[int, str] = {v: k for k, v in self.token_to_idx.items()}
        self.idx_to_label: Dict[int, str] = {v: k for k, v in self.label_to_idx.items()}
        self.token_num = len(self.token_to_idx)
        self.label_num = len(self.label_to_idx)

        self.encoded_tokens = encode_data(self.tokens, self.token_to_idx)
        self.encoded_labels = encode_data(self.labels, self.label_to_idx)

        # transition probabilities: P(u->v) = transition_probs[u][v]
        self.transition_probs: List[List[float]] = None
        # emission probabilities: P(u->o) = emission_probs[u][o]
        self.emission_probs: List[List[float]] = None

    def update_emission_probs(
        self,
        k: int = 1,
        label_smoothing_factor: float = 1.0,
        use_log_likelihood: bool = True,
    ) -> None:
        counter = [[0] * self.token_num for _ in range(self.label_num)]
        tokens_flattened = [token for sample in self.encoded_tokens for token in sample]
        labels_flattened = [label for sample in self.encoded_labels for label in sample]
        assert len(tokens_flattened) == len(labels_flattened)
        for label, token in zip(labels_flattened, tokens_flattened):
            counter[label][token] += 1

        u_sums = [sum(u) + k for u in counter]
        if use_log_likelihood:
            prob_func = lambda x: math.log(x)
        else:
            prob_func = lambda x: x

        self.emission_probs = [
            [
                prob_func(
                    (o_count + label_smoothing_factor)
                    / (u_sum + label_smoothing_factor * self.token_num)
                )
                for o_count in u
            ]
            for u, u_sum in zip(counter, u_sums)
        ]

        # give <UNK> a very low probability for each label
        if k > 0:
            for u_idx, u_sum in enumerate(u_sums):
                self.emission_probs[u_idx][0] = prob_func(
                    (k + label_smoothing_factor)
                    / (u_sum + label_smoothing_factor * self.token_num)
                )

        return

    def train(
        self,
        train_filepath: str = None,
        smoothing_factor: int = 1,
        use_log_likelihood: bool = True,
    ) -> None:
        if train_filepath is not None:
            self._init_module(train_filepath)
        self.update_emission_probs(
            k=1,
            label_smoothing_factor=0.3,
            use_log_likelihood=use_log_likelihood,
        )
        self.update_transition_probs(
            label_smoothing_factor=1,
            use_log_likelihood=use_log_likelihood,
        )
        return

    def get_transition(self, u: int, v: int) -> float:
        if u == "START":
            u = -1
        if v == "STOP":
            v = -1
        return self.transition_probs[u][v]

    def get_emission(self, u: int, o: int) -> float:
        return self.emission_probs[u][o]

    def a(self, u: int, v: int) -> float:
        """ Get transition probability from u to v """
        return self.get_transition(u, v)

    def b(self, u: int, o: int) -> float:
        """ Get emission probability from u to o """
        return self.get_emission(u, o)

    def index_max(self, values: List[float]) -> int:
        """Return index of maximum value in list"""
        return max(range(len(values)), key=values.__getitem__)

    def custom_argmax(self, values: list, top_k: int = 3) -> Tuple[list, list]:
        """
        Custom argmax function. Returns the top-k max values and their indices.

        Args:
            values: list of floats or integers
            top_k: number of top-k values to return

        Returns:
            Tuple[k_max_vals, k_max_indices]
                k_max_vals: list of top-k max values
                k_max_indices: list of indices of top-k max values
        """
        sorted_values = sorted(values, reverse=True)
        k_max_vals = sorted_values[:top_k]
        k_max_idxs = []
        for val in k_max_vals:
            copy = values[:]
            offset = 0
            while True:
                _raw_idx = copy.index(val)
                _idx = offset + _raw_idx
                if _idx not in k_max_idxs:
                    k_max_idxs.append(_idx)
                    break
                else:
                    offset = offset + _raw_idx + 1
                    copy = values[offset:]
        return k_max_vals, k_max_idxs

    def update_transition_probs(
        self,
        label_smoothing_factor: float = 1.0,
        use_log_likelihood: bool = True,
    ) -> None:
        """
        transition_probs

        Last row: START -> v
        Last column: v -> STOP
        """
        # (N + 2) * (N+1) * (N+1)
        counter = []
        for i in range(self.label_num + 2):
            tmp = []
            for j in range(self.label_num + 1):
                tmp.append([0] * (self.label_num + 1))
            counter.append(tmp)

        for sample in self.encoded_labels:
            for i in range(len(sample) - 2):
                # state: START -> u
                # counter: counter[-1][u]
                if i == 0:
                    counter[-2][-1][sample[i]] += 1
                    counter[-1][sample[i]][sample[i + 1]] += 1
                # state: u -> v
                # counter: counter[u][v]

                counter[sample[i]][sample[i + 1]][sample[i + 2]] += 1
                # state: v -> STOP
                # counter: counter[v][-1]
                if i == len(sample) - 3:
                    counter[sample[i + 1]][sample[i + 2]][-1] += 1

        sums = []
        for i in range(self.label_num + 2):
            tmp = []
            for j in range(self.label_num + 1):
                tmp.append(sum(counter[i][j]))
            sums.append(tmp)

        if use_log_likelihood:
            prob_func = lambda x: math.log(x)
        else:
            prob_func = lambda x: x

        self.transition_probs = []
        for i in range(self.label_num + 2):
            tmp = []
            for j in range(self.label_num + 1):
                tmp.append(
                    [
                        prob_func(
                            (u + label_smoothing_factor)
                            / (sums[i][j] + label_smoothing_factor * self.label_num)
                        )
                        for u in counter[i][j]
                    ]
                )
            self.transition_probs.append(tmp)
        return

    def second_order_viterbi(self, X: List[int]) -> List[List[int]]:
        # (T) * (num_labels) * (num_labels)
        T = len(X)
        mem = []
        for i in range(T):
            tmp = []
            for j in range(self.label_num):
                tmp.append([0] * (self.label_num))
            mem.append(tmp)
        for i in range(self.label_num):
            mem[0][-1][i] = (
                self.transition_probs[-2][-1][i] + self.emission_probs[i][X[0]]
            )
        if T > 1:
            for i in range(self.label_num):
                for j in range(self.label_num):
                    mem[1][i][j] = (
                        mem[0][-1][i]
                        + self.transition_probs[-1][i][j]
                        + self.emission_probs[j][X[1]]
                    )
            for t in range(2, T):
                for current_label in range(self.label_num):
                    for prev_label in range(self.label_num):
                        mem[t][prev_label][current_label] = max(
                            [
                                mem[t - 1][i][prev_label]
                                + self.transition_probs[i][prev_label][current_label]
                                + self.emission_probs[current_label][X[t]]
                                for i in range(self.label_num)
                            ]
                        )
        last = []
        for j in range(self.label_num):
            last.append(
                max(
                    [
                        mem[-1][i][j] + self.transition_probs[i][j][-1]
                        for i in range(self.label_num)
                    ]
                )
            )
        ret = [self.arg_max(last), self.label_num]
        for t in range(T - 1, 0, -1):
            max_id = self.arg_max(
                [
                    mem[t][i][ret[0]] + self.transition_probs[i][ret[0]][ret[1]]
                    for i in range(self.label_num)
                ]
            )
            ret.insert(0, max_id)
        return ret[:-1]

    def arg_max(self, ls):
        max_ = max(ls)
        for i in range(len(ls)):
            if ls[i] == max_:
                return i

    def predict(self, test_filepath: str, top_k: int = 5) -> None:
        export_dir = Path(test_filepath).parent
        stem = Path(test_filepath).stem
        test_tokens = get_test_data(test_filepath)
        encoded_test_tokens = encode_data(test_tokens, self.token_to_idx)
        predictions = []
        for test_sample in encoded_test_tokens:
            top_k_paths = self.second_order_viterbi(test_sample)
            predictions.append(top_k_paths)

        export_path = export_dir / f"{stem}.p4.out"
        with export_path.open("w") as f:
            content = []
            assert len(predictions) == len(test_tokens)
            for pred, tokens in zip(predictions, test_tokens):
                assert len(pred) == len(tokens)
                pred_idx = pred
                pred_label = [self.idx_to_label[idx] for idx in pred_idx]
                new_lines = [
                    f"{token} {label}" for token, label in zip(tokens, pred_label)
                ]
                new_lines.append("")
                content.extend(new_lines)
            f.write("\n".join(content))
            print(f"Result saved: '{export_path}'")

        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="ES/train")
    parser.add_argument("--test", type=str, default="ES/dev.in")
    args = parser.parse_args()

    model = SecondOrderHiddenMarkovModel()

    model.train(train_filepath=args.train)
    model.predict(test_filepath=args.test)
    return


if __name__ == "__main__":
    main()
