

export default class Sweeper {
  constructor() {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.audioContext.destination.channelCount = this.audioContext.destination.maxChannelCount >= 8 ? 8 : 2;
    this.sweepGain = this.audioContext.createGain();
    this.sweepGain.gain.value = 0.707;
    this.sweepGain.connect(this.audioContext.destination);
    this.sweepData = null;
    this.sampleRate = null;
    this.activeSweep = null;
    this.recorder = null;
    this.channels = ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR']
    this.silence = 1.0;
    this.initSweepArray = this.initSweepArray.bind(this);
    this.initSweepArray();
    this.playSequence = this.playSequence.bind(this);
  };

  async initSweepArray() {
    const buf = await fetch('sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav')
      .then(r => r.arrayBuffer())
      .then(buf => this.audioContext.decodeAudioData(buf))
    this.sampleRate = buf.sampleRate;
    const data = new Float32Array(buf.length);
    buf.copyFromChannel(data, 0, 0);
    const leadingZeros = 0;
    this.sweepData = new Float32Array(data.length + leadingZeros);
    this.sweepData.fill(0, 0, leadingZeros);
    this.sweepData.set(data, leadingZeros);
  };

  playSequence(channelSequence) {
    this.activeSweep = this.audioContext.createBufferSource();
    const length = this.silence * this.sampleRate + channelSequence.length * (this.silence * this.sampleRate + this.sweepData.length);
    const buffer = this.audioContext.createBuffer(this.channels.length, length, this.sampleRate);
    for (let i = 0; i < channelSequence.length; ++i) {
      const offset = this.silence * this.sampleRate + i * (this.silence * this.sampleRate + this.sweepData.length);
      buffer.copyToChannel(this.sweepData, this.channels.indexOf(channelSequence[i]), offset);
    }
    this.activeSweep.buffer = buffer;
    this.activeSweep.connect(this.sweepGain);
    return new Promise(resolve => {
      this.activeSweep.addEventListener('ended', () => {
        this.activeSweep = null;
        resolve();
      })
      this.activeSweep.start();
    })
  };

  stop() {
    if (this.activeSweep) {
      this.activeSweep.stop();
      this.activeSweep = null;
    }
    if (this.recorder) {
      this.recorder.stop();
    }
  }

  async record() {
    if (navigator.mediaDevices) {
      const stream = await navigator.mediaDevices.getUserMedia({ 'audio': true }).catch((err) => {
        alert('Please allow microphone access and try again');
      });;

      this.recorder = new MediaRecorder(stream);
      this.recorder.start();

      const chunks = [];
      this.recorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      return new Promise(resolve => {
        this.recorder.onstop = async () => {
          this.recorder = null;
          const blob = new Blob(chunks)
          const buffer = await blob.arrayBuffer();
          const audioData = this.audioContext.decodeAudioData(buffer);
          resolve(audioData);
        };
      });

    } else {
      alert('Browser doesn\'t support microphone input');
    }

  }
}
