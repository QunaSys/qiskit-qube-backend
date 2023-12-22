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

*TODO*

