import React from "react";
import { withStyles, Container, Typography, Fab, Box } from "@material-ui/core";
import { Pause as PauseIcon, PlayArrow as PlayArrowIcon, Stop as StopIcon } from '@material-ui/icons';
import Plot from 'react-plotly.js';
import Sweeper from './Sweeper';
import styles from './styles';

class BRIR extends React.Component {
  constructor(props) {
    super(props);

    this.sweeper = new Sweeper();

    const defaultChannels = {};
    for (const ch of this.sweeper.channels) {
      defaultChannels[ch] = {
        name: ch,
        left: {
          waveform: [],
          impulseResponse: [],
        },
        right: {
          waveform: [],
          impulseResponse: [],
        }
      };
    }

    this.state = {
      playing: false,
      channels: defaultChannels,
    };

    this.play = this.play.bind(this);
    this.saveRecording = this.saveRecording.bind(this);
    this.renderSpeaker = this.renderSpeaker.bind(this);
  };

  saveRecording(recording, speakers) {
    const newChannels = { ...this.state.channels };
    for (const [i, ch] of speakers.entries()) {
      // Test signal length plus N seconds of silence between the speakers
      const sweepLen = this.sweeper.sweepData.length + this.sweeper.silence * this.sweeper.sampleRate;
      // N seconds of silence in the beginning plus as many sweeps as has already passed
      const startIx = this.sweeper.silence * this.sweeper.sampleRate + i * sweepLen;
      // One more sweep
      const endIx = startIx + sweepLen;
      newChannels[ch].left.waveform = Array.from(recording.getChannelData(0).slice(startIx, endIx));
      newChannels[ch].right.waveform = Array.from(recording.getChannelData(1).slice(startIx, endIx));
    }
    this.setState({ channels: newChannels });
  };

  async play(speaker) {
    if (this.state.playing) {
      this.sweeper.stop();
    } else {
      this.setState({ playing: true });
      this.sweeper.record().then((recording) => {
        this.saveRecording(recording, [speaker]);
      });
      await this.sweeper.playSequence([speaker]);
      this.setState({ playing: false });
      setTimeout(() => { this.sweeper.stop(); }, this.sweeper.silence * 1000);
    }
  };

  formatRecordingData(data, channel) {
    return [{
      x: Array.from(Array(data.length), (_, i) => i / this.sweeper.sampleRate),
      y: data,
      type: 'scatter',
      color: channel === 'left'
        ? 'rgb(55, 128, 191)'
        : 'rgb(219, 64, 82)'
    }];
  };

  renderSpeaker(channel) {
    const leftPlotData = this.formatRecordingData(channel.left.waveform, 'left');
    const rightPlotData = this.formatRecordingData(channel.right.waveform, 'right');
    return (
      <Box key={channel.name} display="flex">
        <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" mr={1}>
          <Fab color="primary" onClick={ () => { this.play(channel.name) } }>
            {!this.state.playing && (<PlayArrowIcon />)}
            {this.state.playing && (<StopIcon />)}
          </Fab>
          <Typography varian="h4">{channel.name}</Typography>
        </Box>
        <Box>
          <Box>
            <Plot data={leftPlotData} layout={ { width: 800, height: 200, margin: { t: 16, b: 16 } } } />
          </Box>
          <Box>
            <Plot data={rightPlotData} layout={ { width: 800, height: 200, margin: { t: 16, b: 16 } } } />
          </Box>
        </Box>
      </Box>
    );
  };

  render() {
    return (
      <Container>
        <Typography variant='h1'>BRIR</Typography>
        {Object.keys(this.state.channels).map((ch) => {
          if (ch !== 'FL' && ch !== 'FR') {
            return null;
          }
          return this.renderSpeaker(this.state.channels[ch]);
        })}
      </Container>
    )
  };

}

export default withStyles(styles)(BRIR);
