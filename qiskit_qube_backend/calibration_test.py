import itertools as it
from typing import Callable

import calibrations as cs
import circuits as cc
import numpy as np
from backends import gen_backend, init_jax
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, pulse
from qiskit import schedule as build_schedule
from qiskit import transpile
from qiskit.providers import Backend, BackendV2
from qiskit.providers.fake_provider import FakeAlmaden, FakeBelemV2
from qiskit.pulse import Schedule
from qube_backend import Mode, Qube
from recorder import SignalRecorder, plot_iq_plane, plot_out_arr, plot_setup


def plot_data(data: np.ndarray, path: str) -> None:
    i_data, q_data = np.real(data), np.imag(data)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, title, data in zip(axs, ["I data", "Q data"], [i_data, q_data]):
        ax.plot(range(len(data)), data)
        ax.set_xlabel("samples")
        ax.set_title(title)
    fig.savefig(path)


def load_calibrations(
    circ: QuantumCircuit, qubits: list[int], backend: BackendV2
) -> QuantumCircuit:
    imap = backend.instruction_schedule_map
    print(imap)
    for name in imap.instructions:
        if name == "cx":
            if len(qubits) > 1:
                sched = imap.get(name, [0, 1])
                for qs in it.combinations(qubits, 2):
                    circ.add_calibration(name, qubits=qs, schedule=sched)
        else:
            for q in qubits:
                sched = imap.get(name, [q])
                circ.add_calibration(name, qubits=[q], schedule=sched)


def add_measure(circ: QuantumCircuit) -> QuantumCircuit:
    for qubit in [q.index for q in circ.qubits]:
        with pulse.build() as sched:
            pulse.acquire(
                duration=1, qubit_or_channel=qubit, register=pulse.MemorySlot(qubit)
            )
        circ.add_calibration("measure", qubits=[qubit], schedule=sched)
    return circ


def simulate_circuit(circ: QuantumCircuit) -> None:
    backend = gen_backend()
    backend.mode = Mode.SIM
    job = backend.run(circ)
    result = job.result()
    # print(result.get_counts(0))
    # print(result.get_memory(0))
    # print(result.data(0))
    data = result.get_memory(0)
    plot_data(data, "result.png")


def record_cals(calsf: Callable[[Backend], None], path: str) -> None:
    recorder = SignalRecorder()
    backend = gen_backend()
    backend.qube = Qube()
    backend.recorder = recorder
    backend.mode = Mode.RECORD

    calsf(backend)
    setup, out_arr = recorder.history()[0]
    plot_setup(setup)
    plot_out_arr(out_arr)
    plot_iq_plane(out_arr)

    recorder.save(path)


def play_cals(calsf: Callable[[Backend], None], path: str) -> None:
    recorder = SignalRecorder(path)
    backend = gen_backend()
    backend.qube = Qube()
    backend.recorder = recorder
    backend.mode = Mode.PLAY

    calsf(backend)


def record_circuit(circ: QuantumCircuit, path: str) -> None:
    recorder = SignalRecorder()
    backend = gen_backend()
    backend.qube = Qube()
    backend.recorder = recorder
    backend.mode = Mode.RECORD

    data = backend.run(circ).result().get_memory(0)
    plot_data(data, "result.png")

    recorder.save(path)


def play_circuit(circ: QuantumCircuit, path: str) -> None:
    recorder = SignalRecorder(path)
    backend = gen_backend()
    backend.qube = Qube()
    backend.recorder = recorder
    backend.mode = Mode.PLAY

    data = backend.run(circ).result().get_memory(0)
    plot_data(data, "result.png")

    setup, out_arr = recorder.history()[0]
    plot_setup(setup)
    plot_out_arr(out_arr)


def plot_circuit(circ: QuantumCircuit, path: str) -> None:
    fig = circ.draw("mpl")
    fig.savefig(path)


def plot_schedule(schedule: Schedule, path: str) -> None:
    fig = schedule.draw()
    fig.savefig(path)


def plot_schedule_for_backend(
    circ: QuantumCircuit, backend: Backend, path: str
) -> None:
    circ = transpile(circ, backend)
    schedule = build_schedule(circ, backend)
    fig = schedule.draw()
    fig.savefig(path)


def schedule_for_backend(circ: QuantumCircuit, backend: Backend) -> Schedule:
    circ = transpile(circ, backend)
    schedule = build_schedule(circ, backend)
    return schedule


def run_default_pulse() -> None:
    backend = FakeBelemV2()

    # circ = cc.circ_h()
    # circ = cc.circ_x()
    circ = cc.circ_cx()
    # circ = cc.circ_xcx()
    circ = add_measure(circ)
    print(circ)
    plot_circuit(circ, "circuit.png")

    sched = schedule_for_backend(circ, backend)

    print("Scheduled instructions:")
    for _, instr in sched.instructions:
        print(f"    {instr}")
    print("Channels:")
    for channel in sched.channels:
        print(f"    {channel}")

    plot_schedule(sched, "schedule.png")


def run_custom_pulse() -> None:
    init_jax()

    # circ = cc.circ_h_custom()
    circ = cc.circ_1_qubit_custom()
    # circ = cc.circ_2_qubits_custom()
    # circ = cc.circ_cx_custom()
    # circ = cc.circ_measure_custom()
    print(circ)
    plot_circuit(circ, "circuit.png")
    plot_schedule(build_schedule(circ, gen_backend()), "schedule.png")

    path = "record.pkl"
    # simulate_circuit(circ)
    record_circuit(circ, path)
    play_circuit(circ, path)


def run_calibrations() -> None:
    init_jax()

    path = "record.pkl"

    # record_cals(cs.calibrate_frequency_with_spectroscopy, path)
    # play_cals(cs.calibrate_frequency_with_spectroscopy, path)

    record_cals(cs.calibrate_amplitude_with_rabi_experiment, path)
    play_cals(cs.calibrate_amplitude_with_rabi_experiment, path)

    # record_cals(cs.calibrate_sx_pulse, path)
    # play_cals(cs.calibrate_sx_pulse, path)


def main() -> None:
    # run_default_pulse()
    # run_custom_pulse()
    run_calibrations()


if __name__ == "__main__":
    main()
