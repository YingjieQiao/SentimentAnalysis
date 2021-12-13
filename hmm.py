import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from datautils import build_vocab, encode_data, get_test_data, get_train_data


class HiddenMarkovModel:
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
        k: int = 0,
        smoothing_factor: float = 0.000001,
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
            prob_func = lambda x: math.log(x) if x > 0 else math.log(smoothing_factor)
        else:
            prob_func = lambda x: x

        self.emission_probs = [
            [prob_func(o_count / u_sum) if u_sum > 0 else 0 for o_count in u]
            for u, u_sum in zip(counter, u_sums)
        ]

        # give <UNK> a very low probability for each label
        if k > 0:
            for u_idx, u_sum in enumerate(u_sums):
                self.emission_probs[u_idx][self.token_to_idx["<UNK>"]] = prob_func(
                    k / u_sum
                )

        return

    def update_transition_probs(
        self,
        smoothing_factor: float = 0.000001,
        use_log_likelihood: bool = True,
    ) -> None:
        """
        transition_probs

        Last row: START -> v
        Last column: v -> STOP
        """
        counter = [[0] * (self.label_num + 1) for _ in range(self.label_num + 1)]

        for sample in self.encoded_labels:
            for i in range(len(sample) - 1):
                u = sample[i]
                v = sample[i + 1]
                # state: START -> u
                # counter: counter[-1][u]
                if i == 0:
                    counter[-1][u] += 1
                # state: u -> v
                # counter: counter[u][v]
                counter[sample[i]][v] += 1
                # state: v -> STOP
                # counter: counter[v][-1]
                if i == len(sample) - 2:
                    counter[v][-1] += 1

        u_sums = [sum(u) for u in counter]
        if use_log_likelihood:
            prob_func = lambda x: math.log(x) if x > 0 else math.log(smoothing_factor)
        else:
            prob_func = lambda x: x

        self.transition_probs = [
            [prob_func(v_count / u_sum) if u_sum > 0 else 0 for v_count in u]
            for u, u_sum in zip(counter, u_sums)
        ]
        return

    def train(
        self,
        train_filepath: str = None,
        smoothing_factor: float = 0.000001,
        use_log_likelihood: bool = True,
    ) -> None:
        if train_filepath is not None:
            self._init_module(train_filepath)
        self.update_emission_probs(
            smoothing_factor=smoothing_factor,
            use_log_likelihood=use_log_likelihood,
        )
        self.update_transition_probs(
            smoothing_factor=smoothing_factor,
            use_log_likelihood=use_log_likelihood,
        )
        return

    def native_train(self, train_filepath: str = None) -> None:
        if train_filepath is not None:
            self._init_module(train_filepath)
        self.update_emission_probs(k=1, use_log_likelihood=False)
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

    def viterbi(self, X: List[int], top_k: int = 5) -> List[List[int]]:
        """
        Implementation of Top-K Viterbi Algorithm

        Args:
            X: List[int] - encoded sequence
            top_k: int - number of top-k results to return

        Returns:
            List[List[int]] - top-k best path represented by lists of encoded labels
        """
        N = len(X)

        ### Initialization
        # save top_k max log likelihood pi values for each node
        # pi_vals has size (top_k, N, label_num)
        pi_vals = [[[0] * (self.label_num) for _ in range(N)] for _ in range(top_k)]

        # save top-k best predecessor node for each node
        # memo has size (top_k, N, label_num)
        memo = [[[0] * (self.label_num) for _ in range(N)] for _ in range(top_k)]
        top_k_paths = [[] for _ in range(top_k)]

        ### Forward Computation of pi scores
        ## j = 0
        pi_0_start = 1
        ## j >= 1, j ranges from [1, N + 1] inclusive
        for j in range(1, N + 1 + 1):
            # current layer index
            j_idx: int = j - 1
            # previous layer index
            prev_j_idx: int = j - 2
            # x: observed emission value at this step
            x: int = X[j_idx] if j_idx < N else None

            if j == 1:
                for u in self.idx_to_label:
                    _pi = pi_0_start + self.a("START", u) + self.b(u, x)
                    for k in range(top_k):
                        pi_vals[k][j_idx][u] = _pi
            elif 2 <= j <= N:
                for u in self.idx_to_label:
                    v_s = []
                    for v in self.idx_to_label:
                        for k in range(top_k):
                            v_s.append(
                                pi_vals[k][prev_j_idx][v] + self.a(v, u) + self.b(u, x)
                            )
                    # take top k max and argmax
                    k_max, k_max_idx = self.custom_argmax(v_s, top_k=top_k)
                    # save top k max
                    for k in range(top_k):
                        pi_vals[k][j_idx][u] = k_max[k]
                    # save top k argmax
                    for k in range(top_k):
                        memo[k][j_idx][u] = k_max_idx[k]
            else:  # j == N + 1
                v_s = []
                for v in self.idx_to_label:
                    v_s.append(pi_vals[0][prev_j_idx][v] + self.a(v, "STOP"))
                last_k_max, last_k_max_idx = self.custom_argmax(v_s, top_k=top_k)
                for k in range(top_k):
                    top_k_paths[k].append(last_k_max_idx[k])

        ### Recover top-k best paths from memo
        for j in range(1, N):
            for k in range(top_k):
                curr = top_k_paths[k][-1]
                argmax = memo[0][-j][curr]
                argmax = argmax // top_k
                top_k_paths[k].append(argmax)

        ### Reverse path
        top_k_paths = [list(reversed(i)) for i in top_k_paths]
        return top_k_paths

    def predict(self, test_filepath: str, top_k: int = 5) -> None:
        export_dir = Path(test_filepath).parent
        stem = Path(test_filepath).stem
        test_tokens = get_test_data(test_filepath)
        encoded_test_tokens = encode_data(test_tokens, self.token_to_idx)
        predictions = []
        for test_sample in encoded_test_tokens:
            top_k_paths = self.viterbi(test_sample, top_k=top_k)
            predictions.append(top_k_paths)

        for i in range(top_k):
            export_path = export_dir / f"{stem}.top{i+1}.tmp"
            with export_path.open("w") as f:
                content = []
                for pred, tokens in zip(predictions, test_tokens):
                    pred_idx = pred[i]
                    pred_label = [self.idx_to_label[idx] for idx in pred_idx]
                    new_lines = [
                        f"{token} {label}" for token, label in zip(tokens, pred_label)
                    ]
                    new_lines.append("")
                    content.extend(new_lines)
                f.write("\n".join(content))
                print(f"Result saved: '{export_path}'")

        p2_path_tmp = export_dir / f"{stem}.top1.tmp"
        p2_path = export_dir / f"{stem}.p2.out"
        p3_path_tmp = export_dir / f"{stem}.top5.tmp"
        p3_path = export_dir / f"{stem}.p3.out"
        if p2_path_tmp.is_file():
            shutil.copy(p2_path_tmp, p2_path)
        if p3_path_tmp.is_file():
            shutil.copy(p3_path_tmp, p3_path)

        return

    def naive_predict_single(self, X: List[int]) -> List[int]:
        vals = [[self.b(u, o) for u in self.idx_to_label] for o in X]
        return [self.index_max(i) for i in vals]

    def naive_predict(self, test_filepath: str) -> None:
        export_dir = Path(test_filepath).parent
        stem = Path(test_filepath).stem
        test_tokens = get_test_data(test_filepath)
        encoded_test_tokens = encode_data(test_tokens, self.token_to_idx)
        predictions = []
        for test_sample in encoded_test_tokens:
            predictions.append(self.naive_predict_single(test_sample))

        export_path = export_dir / f"{stem}.p1.out"
        with export_path.open("w") as f:
            content = []
            for pred, tokens in zip(predictions, test_tokens):
                pred = [self.idx_to_label[idx] for idx in pred]
                new_lines = [f"{token} {label}" for token, label in zip(tokens, pred)]
                new_lines.append("")
                content.extend(new_lines)
            f.write("\n".join(content))
            print(f"Result saved: '{export_path}'")
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="ES/train")
    parser.add_argument("--test", type=str, default="ES/dev.in")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--naive", action="store_true")
    args = parser.parse_args()

    model = HiddenMarkovModel()
    if args.naive:
        model.native_train(train_filepath=args.train)
        model.naive_predict(test_filepath=args.test)
    else:
        model.train(train_filepath=args.train)
        model.predict(test_filepath=args.test, top_k=args.top_k)
    return


if __name__ == "__main__":
    main()
