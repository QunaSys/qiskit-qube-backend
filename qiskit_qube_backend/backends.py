# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import numpy as np
from matplotlib import pyplot as plt
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate, RZGate, SXGate, XGate
from qiskit.providers import Backend
from qiskit.providers.backend import QubitProperties
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.transpiler import InstructionProperties
from qiskit_dynamics.array import Array
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.solvers import Solver
from qiskit_experiments.calibration_management.basis_gate_library import (
    FixedFrequencyTransmon,
)
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qube_backend import DynamicsBackend


def init_jax() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    Array.set_default_backend("jax")


def gen_solver(dim: int, dt: float, v0: float, v1: float) -> Solver:
    anharm0 = -0.32e9
    r0 = 0.22e9

    anharm1 = -0.32e9
    r1 = 0.26e9

    J = 0.002e9

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))

    ident = np.eye(dim, dtype=complex)
    full_ident = np.eye(dim**2, dtype=complex)

    N0 = np.kron(ident, N)
    N1 = np.kron(N, ident)

    a0 = np.kron(ident, a)
    a1 = np.kron(a, ident)

    a0dag = np.kron(ident, adag)
    a1dag = np.kron(adag, ident)

    static_ham0 = 2.0 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
    static_ham1 = 2.0 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

    static_ham_full = (
        static_ham0 + static_ham1 + 2.0 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
    )

    drive_op0 = 2.0 * np.pi * r0 * (a0 + a0dag)
    drive_op1 = 2.0 * np.pi * r1 * (a1 + a1dag)

    solver = Solver(
        static_hamiltonian=static_ham_full,
        hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1],
        rotating_frame=static_ham_full,
        hamiltonian_channels=["d0", "d1", "u0", "u1"],
        channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1, "u1": v0},
        dt=dt,
    )
    return solver


def gen_backend() -> Backend:
    dim = 3
    dt = 1.0 / 4.5e9
    v0 = 4.86e9
    v1 = 4.97e9

    solver = gen_solver(dim=dim, dt=dt, v0=v0, v1=v1)
    solver_options = {
        "method": "jax_odeint",
        "atol": 1.0e-6,
        "rtol": 1.0e-8,
        "hmax": dt,
    }

    backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=[dim, dim],
        solver_options=solver_options,
        meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.SINGLE,
    )

    target = backend.target
    target.qubit_properties = [
        QubitProperties(frequency=v0),
        QubitProperties(frequency=v1),
    ]
    target.add_instruction(XGate(), properties={(0,): None, (1,): None})
    target.add_instruction(SXGate(), properties={(0,): None, (1,): None})
    target.add_instruction(CXGate(), properties={(0, 1): None, (1, 0): None})

    phi = Parameter("phi")
    with pulse.build() as rz0:
        pulse.shift_phase(phi, pulse.DriveChannel(0))
        pulse.shift_phase(phi, pulse.ControlChannel(1))

    with pulse.build() as rz1:
        pulse.shift_phase(phi, pulse.DriveChannel(1))
        pulse.shift_phase(phi, pulse.ControlChannel(0))

    target.add_instruction(
        RZGate(phi),
        {
            (0,): InstructionProperties(calibration=rz0),
            (1,): InstructionProperties(calibration=rz1),
        },
    )

    return backend


def gen_calibrations() -> Calibrations:
    library = FixedFrequencyTransmon(basis_gates=["x", "sx"])
    cals = Calibrations(libraries=[library])
    return cals


def schedule_to_signal() -> None:
    # 1. create schedule
    # Strength of the Rabi-rate in GHz.
    r = 0.1

    # Frequency of the qubit transition in GHz.
    w = 5.0

    # Sample rate of the backend in ns.
    dt = 1 / 4.5

    # Define gaussian envelope function to approximately implement an sx gate.
    amp = 1.0 / 1.75
    sig = 0.6985 / r / amp
    T = 4 * sig
    duration = int(T / dt)
    beta = 2.0

    with pulse.build(name="sx-sy schedule") as sxp:
        pulse.play(pulse.Drag(duration, amp, sig / dt, beta), pulse.DriveChannel(0))
        pulse.shift_phase(np.pi / 2.0, pulse.DriveChannel(0))
        pulse.play(pulse.Drag(duration, amp, sig / dt, beta), pulse.DriveChannel(0))

    fig = sxp.draw()
    fig.savefig("schedule.png")

    # 2. convert schedule to signal
    plt.rcParams["font.size"] = 16

    converter = InstructionToSignals(dt, carriers={"d0": w})

    signals = converter.get_signals(sxp)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, title in zip(axs, ["envelope", "signal"]):
        signals[0].draw(0, 2 * T, 2000, title, axis=ax)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.vlines(T, ax.get_ylim()[0], ax.get_ylim()[1], "k", linestyle="dashed")

    fig.savefig("signal.png")


def simulate_schedule() -> None:
    sigma = 128
    num_samples = 256

    schedules = []

    for amp in np.linspace(0.0, 1.0, 10):
        gauss = pulse.library.Gaussian(num_samples, amp, sigma, name="Parametric Gauss")
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.play(gauss, d0)
                pulse.shift_phase(0.5, d0)
                pulse.shift_frequency(0.1, d0)
                pulse.play(gauss, d0)
                pulse.acquire(
                    duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0)
                )

        schedules.append(schedule)

    backend = gen_backend()
    job = backend.run(schedules, shots=100)
    result = job.result()
    print(result.get_counts(3))
