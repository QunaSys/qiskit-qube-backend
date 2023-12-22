import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from e7awgsw import AWG, CaptureParam, CaptureUnit, WaveSequence

QubeSetup: TypeAlias = list[tuple[AWG, WaveSequence] | tuple[CaptureUnit, CaptureParam]]
QubeResult: TypeAlias = dict[CaptureUnit, list[np.ndarray[complex]]]


def plot_setup(setup: QubeSetup, path_prefix: str = "setup") -> None:
    for i, (dev, param) in enumerate(setup):
        if isinstance(param, WaveSequence):
            ss = np.array(param.all_samples())
            i_data, q_data = ss[:, 0], ss[:, 1]
            fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
            for ax, title, data in zip(axs, ["I data", "Q data"], [i_data, q_data]):
                ax.plot(range(len(data)), data)
                ax.set_xlabel("samples")
                ax.set_title(title)
            fig.savefig(f"{path_prefix}_{i}.png")
        elif isinstance(param, CaptureParam):
            ...
        else:
            raise ValueError(f"Unknown instruction: {dev}, {param}")


def plot_out_arr(out_arr: np.ndarray, path_prefix: str = "out_arr") -> None:
    for i in range(out_arr.shape[1]):
        i0, q0 = out_arr[:, i, 0], out_arr[:, i, 1]
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
        for ax, title, data in zip(axs, ["I data", "Q data"], [i0, q0]):
            ax.plot(range(len(data)), data)
            ax.set_xlabel("samples")
            ax.set_title(title)
        fig.savefig(f"{path_prefix}_{i}.png")


def plot_iq_plane(out_arr: np.ndarray, path_prefix: str = "iq_plane") -> None:
    for i in range(out_arr.shape[1]):
        i0, q0 = out_arr[:, i, 0], out_arr[:, i, 1]
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.scatter(i0, q0, s=2, alpha=0.5)
        fig.savefig(f"{path_prefix}_{i}.png")


def setup_to_array(setup: QubeSetup) -> np.ndarray:
    arr = np.array([], dtype=np.int64)
    for dev, param in setup:
        arr = np.append(arr, int(dev))
        if isinstance(param, WaveSequence):
            samples = np.array(param.all_samples()).flatten()
            arr = np.concatenate([arr, samples])
        elif isinstance(param, CaptureParam):
            m = hashlib.sha256()
            m.update(str(param).encode())
            d = int.from_bytes(m.digest()) % (2**64)
            # d = int.from_bytes(m.digest())
            arr = np.append(arr, d)
        else:
            raise ValueError(f"Unknown instruction: {dev}, {param}")
    return arr


def setup_to_uid(setup: QubeSetup, bits: int = 64) -> int:
    arr = setup_to_array(setup)
    m = hashlib.sha256()
    m.update(arr.tobytes())
    d = int.from_bytes(m.digest()) % (2**bits)
    return d


K = TypeVar("K", np.ndarray, QubeSetup)
V = TypeVar("V", np.ndarray, QubeResult)


class Recorder(Generic[K, V], ABC):
    def __init__(self, path: str | None = None):
        self._key_record = {}
        self._value_record = {}
        self._history = []
        if path is not None:
            self.load(path)

    @abstractmethod
    def key_to_uid(self, key: K) -> int:
        ...

    def record(self, key: K, value: V) -> None:
        uid = self.key_to_uid(key)
        self._key_record[uid] = key
        self._value_record[uid] = value
        self._history.append(uid)

    def history(self) -> list[tuple[K, V]]:
        return [(self._key_record[i], self._value_record[i]) for i in self._history]

    def play(self, key: K) -> V:
        uid = self.key_to_uid(key)
        return self._value_record[uid]

    def save(self, path: str) -> None:
        with open(path, mode="wb") as f:
            pickle.dump((self._key_record, self._value_record, self._history), f)

    def load(self, path: str) -> None:
        with open(path, mode="rb") as f:
            self._key_record, self._value_record, self._history = pickle.load(f)


class QubeRecorder(Recorder[QubeSetup, QubeResult]):
    def key_to_uid(self, key: QubeSetup) -> int:
        return setup_to_uid(key)


class SignalRecorder(Recorder[QubeSetup, np.ndarray]):
    def key_to_uid(self, key: QubeSetup) -> int:
        return setup_to_uid(key)


class ArrayRecorder(Recorder[np.ndarray, np.ndarray]):
    def key_to_uid(self, key: np.ndarray) -> int:
        m = hashlib.sha256()
        m.update(key.tobytes())
        return m.digest()


def main() -> None:
    rec = ArrayRecorder()

    xss = np.random.rand(5, 5)
    yss = np.random.rand(5, 5)

    for xs, ys in zip(xss, yss):
        rec.record(xs, ys)
    for xs, ys in zip(xss, yss):
        print(np.array_equal(ys, rec.play(xs)))

    filename = "record.pkl"
    rec.save(filename)
    rec2 = ArrayRecorder(filename)
    for xs, ys in zip(xss, yss):
        print(np.array_equal(ys, rec2.play(xs)))


if __name__ == "__main__":
    main()
