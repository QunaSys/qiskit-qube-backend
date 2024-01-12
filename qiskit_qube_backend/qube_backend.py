# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import datetime
import hashlib
import math
import uuid
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeAlias

import numpy as np
import qiskit_experiments.data_processing as dp
from qiskit import QuantumCircuit, pulse
from qiskit import schedule as build_schedule
from qiskit.providers.backend import BackendV2
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.channels import (
    AcquireChannel,
    Channel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
    MemorySlot,
)
from qiskit.pulse.instructions import Acquire
from qiskit.pulse.transforms import block_to_schedule
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.result import Counts, Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_dynamics.backend import DynamicsBackend
from qiskit_dynamics.backend.dynamics_job import DynamicsJob
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.signals import DiscreteSignal, Signal
from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.test.pulse_backend import (
    PulseBackend,
    SingleTransmonTestBackend,
)
from qiskit_experiments.test.utils import FakeJob
from recorder import QubeResult, QubeSetup, SignalRecorder

from e7awgsw import (
    AWG,
    AwgCtrl,
    CaptureCtrl,
    CaptureModule,
    CaptureParam,
    CaptureUnit,
    WaveSequence,
)

# from qubecalib import backendqube
# from qubecalib.qube import Qube, QubeBase, QubeTypeA, QubeTypeB


# QubeSetup: TypeAlias = list[tuple[AWG, WaveSequence] | tuple[CaptureUnit, CaptureParam]]
# QubeResult: TypeAlias = dict[CaptureUnit, list[np.ndarray[complex]]]


def signals_to_wavesequence(
    signals: list[DiscreteSignal],
    converter: InstructionToSignals,
) -> WaveSequence:
    iq_samples = []
    current_time = 0

    # 同一チャネルに対する時間ソート済みのsignalsであることを仮定
    for signal in signals:
        start_time = signal.start_time

        if start_time > current_time:
            iq_samples.extend([(0, 0)] * (start_time - current_time))
        if_modulation = 0.0  # TODO acquire if modulation
        in_phase, quadrature = converter.get_awg_signals([signal], if_modulation)
        iq_data = list(
            zip(
                (np.real(np.array(in_phase.samples)) * 32767.0).astype(int),
                (np.imag(np.array(quadrature.samples)) * 32767.0).astype(int),
            )
        )
        iq_samples.extend(iq_data)

        current_time = start_time + len(iq_data)

    # 波形チャンク内の波形パートの最小単位64に合わせて0データをパディング
    pudding = 64 - (len(iq_samples) % 64)
    iq_samples.extend([(0, 0)] * pudding)

    waveseq = WaveSequence(num_wait_words=0, num_repeats=1)
    waveseq.add_chunk(iq_samples, num_blank_words=0, num_repeats=1)

    # 波形チャンクの波形パートの最小単位が64、ポストプランクの最小単位が4であることから
    # 波形チャンクを並べる形では任意のstart_timeに対応できない可能性がある
    # for i, signal in enumerate(signals):
    #     start_time = signal.start_time

    #     if start_time > current_time:
    #         waveseq.add_chunk(
    #             [(0, 0)] * (start_time - current_time), num_blank_words=0, num_repeats=1
    #         )

    #     in_phase, quadrature = converter.get_awg_signals([signal], if_modulation)
    #     iq_samples = list(
    #         zip(
    #             (np.array(in_phase.samples) * 32767.0).astype(int),
    #             (np.array(quadrature.samples) * 32767.0).astype(int),
    #         )
    #     )
    #     waveseq.add_chunk(iq_samples, num_blank_words=0, num_repeats=1)

    #     current_time = start_time + len(iq_samples)

    return waveseq


def signal_to_wavesequence(signal: Signal, dt: float, n_samples: int) -> WaveSequence:
    return signal_to_wavesequence([signal], dt, n_samples)


def awg_dt() -> float:
    return 1.0 / AwgCtrl.SAMPLING_RATE


# def awg_dt(freq: float = 2.5e6) -> float:
#     return 2.0 * pi * freq * AwgCtrl.SAMPLING_RATE


# Moch
@dataclass
class Qube:
    awg0 = 0
    awg1 = 1
    unit0 = 0
    unit1 = 1
    mhz = 2000.0

    def __getattr__(self, name) -> "Qube":
        return self


class ChannelDeviceMap:
    def __init__(
        self, channels: list[Channel], qube: Qube
    ):  # TODO consider qubit connectivity?
        self._qube = qube

        channel_device_map = {}
        capture_channel_map = {}
        index_capture_map = {}

        # TODO support multiple awg cap pairs
        # awgs = list(qube.port0.awgs)
        awgs = [qube.port0.awg0, qube.port0.awg1]
        caps = [qube.port1.capt.unit0, qube.port1.capt.unit1]

        # TODO check if we can use channel index for mapping
        for channel in channels:
            if isinstance(channel, DriveChannel):  # send 1 qubit gate signal : Awg
                channel_device_map[channel] = awgs[channel.index]
            elif isinstance(
                channel, ControlChannel
            ):  # send 2 qubit gate control signal : Awg
                raise NotImplementedError()
            elif isinstance(channel, MeasureChannel):  # send measurement signal : Awg
                channel_device_map[channel] = awgs[channel.index]
            elif isinstance(
                channel, AcquireChannel
            ):  # capture measurement data : CaptureUnit
                channel_device_map[channel] = caps[channel.index]
                capture_channel_map[caps[channel.index]] = channel
                index_capture_map[channel.index] = caps[channel.index]

        self._channel_device_map = channel_device_map
        self._capture_channel_map = capture_channel_map
        self._index_capture_map = index_capture_map

    def channel_to_device(self, channel: Channel) -> AWG | CaptureUnit:
        return self._channel_device_map[channel]

    def channel_index_to_capture_device(self, index: int) -> CaptureUnit:
        return self._index_capture_map[index]

    def capture_device_to_channel(self, device: CaptureUnit) -> Channel:
        return self._capture_channel_map[device]

    def analog_carrier_frequency(self, channel: Channel) -> float:
        # TODO check meaning of analog carrier frequency
        return self._qube.port0.lo.mhz * 1.0e6


def schedule_to_setup(
    schedule: Schedule, channel_device_map: ChannelDeviceMap
) -> QubeSetup:
    setup = []
    # channel_index_to_waveseq = {}
    print("Scheduled instructions:")
    for _, instr in schedule.instructions:
        print(f"    {instr}")

    print("Channels:")
    for channel in schedule.channels:
        print(f"    {channel}")

        if isinstance(channel, AcquireChannel):
            continue
            # cap = map_channel(channel)
            # setup.append((cap, capparam))
        elif isinstance(channel, MemorySlot):
            continue
        else:
            awg = channel_device_map.channel_to_device(channel)
            converter = InstructionToSignals(
                dt=awg_dt(),  # TODO check acquire dt from device
                carriers={
                    channel.name: channel_device_map.analog_carrier_frequency(channel)
                },
                channels=[channel.name],
            )
            signals = converter.get_signals(schedule)
            waveseq = signals_to_wavesequence(signals, converter)
            setup.append((awg, waveseq))
            # channel_index_to_waveseq[channel.index] = waveseq

    for start_time, inst in schedule.instructions:
        if isinstance(inst, Acquire):
            # TODO construct CaptureParam depending on meas level
            # capture_delayの最小単位が16wordであることによるstart_timeとのずれについて要確認
            delay = math.ceil(math.ceil(start_time / 4.0) / 16.0) * 16

            p = CaptureParam()
            p.num_integ_sections = 1
            p.add_sum_section(num_words=32, num_post_blank_words=1)
            p.capture_delay = delay

            # TODO check awg cap pair specs - pairing by inst order or by device instance
            # setup.append(channel_index_to_waveseq[inst.channels[0].index])
            cap = channel_device_map.channel_to_device(inst.channels[0])
            setup.append((cap, p))

    return setup


# qubecalibを利用するか、e7awgswを直接利用するかここで選択
def send_receive(
    setup: list[tuple[AWG, WaveSequence] | tuple[CaptureUnit, CaptureParam]]
) -> QubeResult:
    # return backendqube.send_recv(*setup)
    # TODO switch moch / real device here in the future
    ...


def experiment_result_to_quel_result(
    experiment_result: ExperimentResult,
    channel_device_map,
    measurement_subsystems,
    memory_slot_indices,
    num_memory_slots,
) -> QubeResult:
    quel_result = {}
    mem_slot_iq = experiment_result.data.memory

    for mem_idx, channel_idx in zip(memory_slot_indices, measurement_subsystems):
        cap = channel_device_map.channel_index_to_capture_device(channel_idx)
        data = mem_slot_iq[mem_idx].T / 32767.0
        quel_result[cap] = data[0] + 1j * data[1]

    return quel_result


def quel_result_to_experimentresult(
    quel_result: QubeResult,
    channel_device_map,
    measurement_subsystems,
    memory_slot_indices,
    num_memory_slots,
    name,
    metadata,
    backend,
) -> ExperimentResult:
    # Acquire IQ data
    mem_slot_iq = np.zeros(num_memory_slots, 2)
    channel_slot_iq = np.zeros(num_memory_slots, 2)
    for cap, data in quel_result.items():
        iq_data = np.array([np.real(data[0]), np.imag(data[0])]).T * 32767.0
        i = channel_device_map.capture_device_to_channel(cap).index
        channel_slot_iq[i] = iq_data

    for mem_idx, channel_idx in zip(memory_slot_indices, measurement_subsystems):
        mem_slot_iq[mem_idx] = channel_slot_iq[channel_idx]

    match backend.options.meas_level:
        case MeasLevel.CLASSIFIED:
            raise NotImplementedError()
            # TODO IQ data to counts
            # counts = ...
            # exp_data = ExperimentResultData(
            #     counts=counts, memory=mem_slot_iq if backend.options.memory else None
            # )
            # return ExperimentResult(
            #     shots=backend.options.shots,
            #     success=True,
            #     data=exp_data,
            #     meas_level=MeasLevel.CLASSIFIED,
            #     # seed=seed,
            #     header=QobjHeader(name=name, metadata=metadata),
            # )
        case MeasLevel.KERNELED:
            exp_data = ExperimentResultData(memory=mem_slot_iq)
            return ExperimentResult(
                shots=backend.options.shots,
                success=True,
                data=exp_data,
                meas_level=MeasLevel.KERNELED,
                # seed=seed,
                header=QobjHeader(name=name, metadata=metadata),
            )
        case _:
            raise QiskitError(
                f"meas_level=={backend.options.meas_level} not implemented."
            )


def out_arr_to_counts(data: np.ndarray) -> dict[int, int]:
    # シミュレータのiq_centersに合わせてIQ平面上の回転角で簡易的に分類
    t0, t1 = np.pi / 3.0, 4.0 * np.pi / 3.0

    i, q = data.squeeze().T
    t = np.angle(i + 1j * q) % (2.0 * np.pi)
    c0 = len(t[(t >= t1) | (t < t0)])
    c1 = len(t) - c0

    return {0: c0, 1: c1}


# def memory_to_counts(data: dict[str, Any]) -> dict[int, int]:
#     # nodes = [dp.nodes.Probability(outcome="1")]
#     # nodes = [dp.nodes.SVD(), dp.nodes.AverageData(axis=1), dp.nodes.MinMaxNormalize()]
#     nodes = [dp.nodes.SVD(), dp.nodes.MemoryToCounts()]
#     processor = dp.DataProcessor("memory", nodes)
#     processor.train(data)
#     return processor(data)


# from qiskit dynamics
def _get_acquire_instruction_timings(
    schedules: list[Schedule], subsystem_dims: list[int], dt: float
) -> tuple[list[list[float]], list[list[int]], list[list[int]]]:
    """Get the required data from the acquire commands in each schedule.

    Additionally validates that each schedule has Acquire instructions occurring at one time, at
    least one memory slot is being listed, and all measured subsystem indices are less than
    ``len(subsystem_dims)``. Additionally, a warning is raised if a 'trivial' subsystem is measured,
    i.e. one with dimension 1.

    Args:
        schedules: A list of schedules.
        subsystem_dims: List of subsystem dimensions.
        dt: The sample size.
    Returns:
        A tuple of lists containing, for each schedule: the list of integration intervals required
        for each schedule (in absolute time, from 0.0 to the beginning of the acquire instructions),
        a list of the subsystems being measured, and a list of the memory slots indices in which to
        store the results of each subsystem measurement.
    Raises:
        QiskitError: If a schedule contains no measurement, if a schedule contains measurements at
            different times, or if a measurement has an invalid subsystem label.
    """
    t_span_list = []
    measurement_subsystems_list = []
    memory_slot_indices_list = []
    for schedule in schedules:
        schedule_acquires = []
        schedule_acquire_times = []
        for start_time, inst in schedule.instructions:
            # only track acquires saving in a memory slot
            if isinstance(inst, pulse.Acquire) and inst.mem_slot is not None:
                schedule_acquires.append(inst)
                schedule_acquire_times.append(start_time)

        # validate
        if len(schedule_acquire_times) == 0:
            raise QiskitError(
                "At least one measurement saving a a result in a MemorySlot "
                "must be present in each schedule."
            )

        for acquire_time in schedule_acquire_times[1:]:
            if acquire_time != schedule_acquire_times[0]:
                raise QiskitError(
                    "DynamicsBackend.run only supports measurements at one time."
                )

        # use dt to convert acquire start time from sample index to the integration interval
        t_span_list.append([0.0, dt * schedule_acquire_times[0]])
        measurement_subsystems = []
        memory_slot_indices = []
        for inst in schedule_acquires:
            if not inst.channel.index < len(subsystem_dims):
                raise QiskitError(
                    f"Attempted to measure out of bounds subsystem {inst.channel.index}."
                )

            if subsystem_dims[inst.channel.index] == 1:
                warnings.warn(
                    f"Measuring trivial subsystem {inst.channel.index} with dimension 1."
                )

            measurement_subsystems.append(inst.channel.index)

            memory_slot_indices.append(inst.mem_slot.index)

        measurement_subsystems_list.append(measurement_subsystems)
        memory_slot_indices_list.append(memory_slot_indices)

    return t_span_list, measurement_subsystems_list, memory_slot_indices_list


# from qiskit dynamics
def _to_schedule_list(
    run_input: list[QuantumCircuit | Schedule | ScheduleBlock], backend: BackendV2
):
    """Convert all inputs to schedules, and store the number of classical
    registers present in any circuits."""
    if not isinstance(run_input, list):
        run_input = [run_input]

    schedules = []
    num_memslots = []
    for sched in run_input:
        num_memslots.append(None)
        if isinstance(sched, ScheduleBlock):
            schedules.append(block_to_schedule(sched))
        elif isinstance(sched, Schedule):
            schedules.append(sched)
        elif isinstance(sched, QuantumCircuit):
            num_memslots[-1] = sum(creg.size for creg in sched.cregs)
            schedules.append(
                build_schedule(sched, backend, dt=backend.options.solver._dt)
            )
        else:
            raise QiskitError(f"Type {type(sched)} cannot be converted to Schedule.")
    return schedules, num_memslots


class Mode(Enum):
    SIM = auto()
    RECORD = auto()
    PLAY = auto()
    QUBE = auto()


# class QuelDynamicsBackend(DynamicsBackend):
class DynamicsBackend(DynamicsBackend):
    @property
    def recorder(self) -> SignalRecorder:
        return self._recorder

    @recorder.setter
    def recorder(self, recorder: SignalRecorder) -> None:
        self._recorder = recorder

    @property
    def mode(self) -> Mode:
        return self._mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        self._mode = mode

    @property
    def qube(self) -> Qube:
        return self._qube

    @qube.setter
    def qube(self, qube: Qube) -> None:
        self._qube = qube

    def run(
        self,
        run_input: list[QuantumCircuit | Schedule | ScheduleBlock],
        validate: bool | None = True,
        **options,
    ) -> DynamicsJob:
        backend = self

        schedules, num_memory_slots_list = _to_schedule_list(run_input, backend=backend)

        (
            t_span,
            measurement_subsystems_list,
            memory_slot_indices_list,
        ) = _get_acquire_instruction_timings(
            schedules,
            backend.options.subsystem_dims,
            awg_dt(),  # TODO check acquire dt from device
        )

        match self.mode:
            case Mode.SIM:
                runf = self._run
            case Mode.RECORD:
                runf = self._run_record
            case Mode.PLAY:
                runf = self._run_play
            case Mode.QUBE:
                runf = self._run_qube
            case _:
                raise ValueError()

        job_id = str(uuid.uuid4())
        dynamics_job = DynamicsJob(
            backend=backend,
            job_id=job_id,
            # fn=backend._run,
            fn=runf,
            fn_kwargs={
                "t_span": t_span,
                "schedules": schedules,
                "measurement_subsystems_list": measurement_subsystems_list,
                "memory_slot_indices_list": memory_slot_indices_list,
                "num_memory_slots_list": num_memory_slots_list,
            },
        )
        dynamics_job.submit()

        return dynamics_job

    def _run_record(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )

        experiment_results = []
        # rng = np.random.default_rng(self.options.seed_simulator)
        # seed=rng.integers(low=0, high=9223372036854775807),
        seed = 0
        for schedule, meas_subs, memory_slots, num_memory_slots, result in zip(
            schedules,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
            solver_results,
        ):
            # self.options.meas_level = MeasLevel.CLASSIFIED

            exp_result = self.options.experiment_result_function(
                schedule.name,
                result,
                meas_subs,
                memory_slots,
                num_memory_slots,
                self,
                seed=seed,
                metadata=schedule.metadata,
            )
            experiment_results.append(exp_result)

            channel_device_map = ChannelDeviceMap(schedule.channels, qube=self.qube)
            setup = schedule_to_setup(schedule, channel_device_map)
            out_arr = exp_result.data.memory
            self.recorder.record(setup, out_arr)

            # print(exp_result.data.counts)

            # print("--------")
            # print(memory_to_counts({"memory": out_arr}))
            counts = out_arr_to_counts(out_arr)
            print(f"Counts: {counts}")

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    def _run_play(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        experiment_results = []
        seed = 0
        for schedule, meas_subs, memory_slots, num_memory_slots in zip(
            schedules,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
        ):
            channel_device_map = ChannelDeviceMap(schedule.channels, qube=self.qube)
            setup = schedule_to_setup(schedule, channel_device_map)
            out_arr = self.recorder.play(setup)

            counts = out_arr_to_counts(out_arr)
            print(f"Counts: {counts}")

            match self.options.meas_level:
                case MeasLevel.CLASSIFIED:
                    raise NotImplementedError(
                        "MeasLevel.CLASSIFIED is not yet supported."
                    )
                case MeasLevel.KERNELED:
                    exp_result = ExperimentResult(
                        shots=self.options.shots,
                        success=True,
                        data=ExperimentResultData(memory=out_arr),
                        meas_level=MeasLevel.KERNELED,
                        seed=seed,
                        header=QobjHeader(
                            name=schedule.name, metadata=schedule.metadata
                        ),
                    )
                case _:
                    raise ValueError(
                        f"Unsupported meas_level: {self.options.meas_level}"
                    )

            experiment_results.append(exp_result)

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    def _run_qube(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        # converter = InstructionToSignals(dt, carriers, channels)

        experiment_results = []
        for schedule, meas_subs, memory_slots, num_memory_slots in zip(
            schedules,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
        ):
            # TODO consider qubit connectivity?
            channel_device_map = ChannelDeviceMap(schedule.channels, qube=self.qube)
            setup = schedule_to_setup(schedule, channel_device_map)

            quel_result = send_receive(*setup)

            exp_result = quel_result_to_experimentresult(
                quel_result,
                channel_device_map,
                meas_subs,
                memory_slots,
                num_memory_slots,
                schedule.name,
                schedule.metadata,
                self,
            )
            experiment_results.append(exp_result)

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    def _run(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        """Simulate a list of schedules."""

        # simulate all schedules
        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )

        # compute results for each experiment
        experiment_names = [schedule.name for schedule in schedules]
        experiment_metadatas = [schedule.metadata for schedule in schedules]
        rng = np.random.default_rng(self.options.seed_simulator)
        experiment_results = []
        for (
            experiment_name,
            solver_result,
            measurement_subsystems,
            memory_slot_indices,
            num_memory_slots,
            experiment_metadata,
        ) in zip(
            experiment_names,
            solver_results,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
            experiment_metadatas,
        ):
            experiment_results.append(
                self.options.experiment_result_function(
                    experiment_name,
                    solver_result,
                    measurement_subsystems,
                    memory_slot_indices,
                    num_memory_slots,
                    self,
                    seed=rng.integers(low=0, high=9223372036854775807),
                    metadata=experiment_metadata,
                )
            )

        # Construct full result object
        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )
