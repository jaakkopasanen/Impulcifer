

export default class Sweeper {
  constructor() {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.sweepGain = this.audioContext.createGain();
    this.sweepGain.gain.value = 0.1;
    this.sweepGain.connect(this.audioContext.destination);
    this.sweepData = null;
    this.sampleRate = null;
    this.activeSweep = null;
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
    const length = 2 * this.sampleRate + channelSequence.length * (2 * this.sampleRate + this.sweepData.length);
    const buffer = this.audioContext.createBuffer(channelSequence.length, length, this.sampleRate);
    for (let i = 0; i < channelSequence.length; ++i) {
      const offset = 2 * this.sampleRate + i * (2 * this.sampleRate + this.sweepData.length);
      buffer.copyToChannel(this.sweepData, i, offset);
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
  }
}
