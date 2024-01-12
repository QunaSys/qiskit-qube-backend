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
from qiskit.providers import Backend
from qiskit_experiments.calibration_management.basis_gate_library import (
    FixedFrequencyTransmon,
)
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.library.calibration import (
    FineSXAmplitudeCal,
    RoughFrequencyCal,
    RoughXSXAmplitudeCal,
)


def gen_cals(backend: Backend) -> Calibrations:
    library = FixedFrequencyTransmon(default_values={"duration": 320})
    cals = Calibrations.from_backend(backend, libraries=[library])
    return cals


def calibrate_frequency_with_spectroscopy(backend: Backend) -> None:
    cals = gen_cals(backend)
    qubit = 0

    # freq_est = backend.defaults().qubit_freq_est[qubit]
    freq_est = 4.86e9
    frequencies = np.linspace(freq_est - 15e6, freq_est + 15e6, 51)
    spec = RoughFrequencyCal([qubit], cals, frequencies, backend=backend)
    spec.set_experiment_options(amp=0.005)

    spec_data = spec.run().block_for_results()
    fig = spec_data.figure(0).figure
    fig.savefig("spec.png")


def calibrate_amplitude_with_rabi_experiment(backend: Backend) -> None:
    cals = gen_cals(backend)
    qubit = 0

    rabi = RoughXSXAmplitudeCal(
        (qubit,), cals, backend=backend, amplitudes=np.linspace(-0.1, 0.1, 51)
    )

    rabi_data = rabi.run().block_for_results()
    fig = rabi_data.figure(0).figure
    fig.savefig("rabi.png")


def calibrate_sx_pulse(backend: Backend) -> None:
    cals = gen_cals(backend)
    qubit = 0
    sx_cal = FineSXAmplitudeCal([qubit], cals, backend=backend, schedule_name="sx")

    sx_data = sx_cal.run().block_for_results()
    fig = sx_data.figure(0).figure
    fig.savefig("sx.png")
