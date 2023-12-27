# qiskit-qube-backend
Qiskit Backend implementation for QuBE devices

# 目的と機能

Qiskit Experimentsに用意されているキャリブレーション実験コードをQuBEデバイス上で動かすために、ソフトウェア的な接続が可能かどうかを検証するための、試作コードです。

- QuBEデバイスを表現するQiskitの`Backend`オブジェクトを用意
- `Backend`オブジェクト内で、Qiskit Pulse APIで表現されたパルススケジュール (`Schedule`) をe7awgswの`WaveSequence` / `CaptureParam`等の形式に変換
    - 実機で動作させるにはポートの対応関係や時刻の同期等についてデバッグが必要と予想される
- Qiskit Dynamicsのシミュレータを利用して、生成されるはずの測定結果を再現
- Qiskit Experimentsの各キャリブレーション実験について、QuBEデバイスに送受信されるはずのデータ (`send_recv()`の引数と戻り値) を記録、プロット、再生

# ファイル構成

|ファイル名|機能|
|:--|:--|
|backends.py|Qiskit Dynamicsのために`Backend`のセットアップを行うユーティリティ|
|calibrations.py|Qiskit Experimentsのキャリブレーション実験のコレクション|
|calibration\_test.py|実際にキャリブレーション実験を走らせてデータの生成や記録、プロット等をテストする実行可能コード|
|circuits.py|テスト用の回路のコレクション|
|qube\_backend.py|QuBEデバイスを表現するQiskitの`Backend`|
|recorder.py|送受信データの記録と再生のためのユーティリティ|

# 実行例

## 環境準備

リポジトリをクローンし、poetryで必要なpythonパッケージをインストールします。

```shell
git clone git@github.com:QunaSys/qiskit-qube-backend.git
cd qiskit-qube-backend
poetry shell
poetry install
```

## キャリブレーション実験のテスト実行

Qiskit Experimentsに用意されているキャリブレーション実験 (Calibrations) を幾つか実行し、実験中に実行されるQiskitのパルススケジュール (Schedule) の内容の確認や、送受信されるはずのデータの記録、プロットなどを行います。

```shell
cd qiskit_pulse_backend
python calibration_test.py
```

[以下のコード](https://github.com/QunaSys/qiskit-qube-backend/blob/264657418517da97caad5d0d9b8ec22fb0f487b5/qiskit_qube_backend/calibration_test.py#L188-L200)でどの実験を走らせるかを選択します。

```python
def run_calibrations() -> None:
    init_jax()

    path = "record.pkl"

    record_cals(cs.calibrate_frequency_with_spectroscopy, path)
    play_cals(cs.calibrate_frequency_with_spectroscopy, path)

    # record_cals(cs.calibrate_amplitude_with_rabi_experiment, path)
    # play_cals(cs.calibrate_amplitude_with_rabi_experiment, path)

    # record_cals(cs.calibrate_sx_pulse, path)
    # play_cals(cs.calibrate_sx_pulse, path)
```

**Qubit Spectroscopy**

以下の行を有効にしてスクリプトを実行します。

```python
    record_cals(cs.calibrate_frequency_with_spectroscopy, path)
    play_cals(cs.calibrate_frequency_with_spectroscopy, path)
```

**ラビ振動実験**

以下の行を有効にしてスクリプトを実行します。

```python
    record_cals(cs.calibrate_amplitude_with_rabi_experiment, path)
    play_cals(cs.calibrate_amplitude_with_rabi_experiment, path)
```

**ゲートファインチューニング**

以下の行を有効にしてスクリプトを実行します。

```python
    record_cals(cs.calibrate_sx_pulse, path)
    play_cals(cs.calibrate_sx_pulse, path)
```

## 出力結果の変更

各キャリブレーション実験では複数のパルススケジュールが実行されます。デフォルトではQiskitのパルススケジュールの概要がすべて標準出力に出力され、最初のパルススケジュールの結果が画像ファイルにプロットされます。[以下のコード](https://github.com/QunaSys/qiskit-qube-backend/blob/730e0bf443d3d95b414aee2072bff9afd55b7acd/qiskit_qube_backend/calibration_test.py#L68-L81)で、何番目の結果をプロットするかをカスタマイズできます。

```python
def record_cals(calsf: Callable[[Backend], None], path: str) -> None:
    recorder = SignalRecorder()
    backend = gen_backend()
    backend.qube = Qube()
    backend.recorder = recorder
    backend.mode = Mode.RECORD

    calsf(backend)
    setup, out_arr = recorder.history()[0]  # 記録データから最初の結果を取得
    plot_setup(setup)
    plot_out_arr(out_arr)
    plot_iq_plane(out_arr)

    recorder.save(path)
```
