# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from qiskit import QuantumCircuit, pulse


def circ_h_custom() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.h(0)
    circ.measure([0], [0])

    # Moch
    with pulse.build() as h_q0:
        pulse.play(
            pulse.library.Gaussian(duration=256, amp=0.2, sigma=50, name="h_q0"),
            pulse.DriveChannel(0),
        )
    circ.add_calibration("h", qubits=[0], schedule=h_q0)

    return circ


def circ_1_qubit_custom() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.h(0)
    circ.x(0)
    circ.measure([0], [0])

    # Moch
    with pulse.build() as h_q0:
        pulse.play(
            pulse.library.Gaussian(duration=256, amp=0.2, sigma=50, name="h_q0"),
            pulse.DriveChannel(0),
        )
    circ.add_calibration("h", qubits=[0], schedule=h_q0)

    with pulse.build() as x_q1:
        pulse.play(
            pulse.library.Drag(
                duration=160,
                sigma=40,
                beta=-2.4305800297101414,
                amp=0.24045125169257026,
                angle=0.0,
                name="x_q1",
            ),
            pulse.DriveChannel(0),
        )
    circ.add_calibration("x", qubits=[0], schedule=x_q1)

    return circ


def circ_2_qubits_custom() -> QuantumCircuit:
    circ = QuantumCircuit(2, 2)
    circ.h(0)
    circ.x(1)
    circ.measure([0, 1], [0, 1])

    # Moch
    with pulse.build() as h_q0:
        pulse.play(
            pulse.library.Gaussian(duration=256, amp=0.2, sigma=50, name="h_q0"),
            pulse.DriveChannel(0),
        )
    circ.add_calibration("h", qubits=[0], schedule=h_q0)

    with pulse.build() as x_q1:
        pulse.play(
            pulse.library.Drag(
                duration=160,
                sigma=40,
                beta=-2.4305800297101414,
                amp=0.24045125169257026,
                angle=0.0,
                name="x_q1",
            ),
            pulse.DriveChannel(1),
        )
    circ.add_calibration("x", qubits=[1], schedule=x_q1)

    return circ


def circ_cx_custom() -> QuantumCircuit:
    circ = QuantumCircuit(2, 2)
    circ.cx(0, 1)
    circ.measure([0, 1], [0, 1])

    # Moch
    with pulse.build() as cx_q01:
        pulse.shift_phase(np.pi / 2.0, pulse.ControlChannel(0))
        pulse.play(
            pulse.library.Drag(
                duration=160,
                sigma=40,
                beta=-2.4030014266125312,
                amp=0.11622814090041741,
                angle=1.6155738267853115,
                name="cx_d0",
            ),
            pulse.DriveChannel(0),
        )
        pulse.play(
            pulse.library.Drag(
                duration=160,
                sigma=40,
                beta=0.6889687213780946,
                amp=0.11976666126366188,
                angle=0.02175043021781017,
                name="cx_d1",
            ),
            pulse.DriveChannel(1),
        )
        pulse.play(
            pulse.library.GaussianSquare(
                duration=1584,
                sigma=64,
                width=1328,
                amp=0.14343084605450945,
                angle=-2.9352017171062608,
                name="cx_c1",
            ),
            pulse.ControlChannel(1),
        )
    circ.add_calibration("cx", qubits=[0, 1], schedule=cx_q01)

    return circ


def circ_measure_custom() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.h(0)
    circ.measure([0], [0])

    # Moch
    with pulse.build() as h_q0:
        pulse.play(
            pulse.library.Gaussian(duration=256, amp=0.2, sigma=50, name="h_q0"),
            pulse.DriveChannel(0),
        )
    circ.add_calibration("h", qubits=[0], schedule=h_q0)

    with pulse.build() as m_q0:
        pulse.play(
            pulse.library.GaussianSquare(
                duration=1584,
                sigma=64,
                width=1328,
                amp=0.14343084605450945,
                angle=0.20639093648353235,
                name="m_q0",
            ),
            pulse.DriveChannel(0),
        )
    circ.add_calibration("measure", qubits=[0], schedule=m_q0)

    return circ


def circ_h() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.h(0)
    circ.measure([0], [0])
    return circ


def circ_x() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.x(0)
    circ.measure([0], [0])
    # circ.measure_all()
    return circ


def circ_1_qubit() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.x(0)
    circ.sx(0)
    circ.measure([0], [0])
    # circ.measure_all()
    return circ


def circ_2_qubits() -> QuantumCircuit:
    circ = QuantumCircuit(2, 2)
    circ.x(0)
    circ.sx(1)
    circ.measure([0, 1], [0, 1])
    # circ.measure_all()
    return circ


def circ_cx() -> QuantumCircuit:
    circ = QuantumCircuit(2, 2)
    circ.cx(0, 1)
    circ.measure([0, 1], [0, 1])
    # circ.measure_all()
    return circ


def circ_xcx() -> QuantumCircuit:
    circ = QuantumCircuit(2, 2)
    circ.x(0)
    circ.cx(0, 1)
    circ.measure([0, 1], [0, 1])
    # circ.measure_all()
    return circ
