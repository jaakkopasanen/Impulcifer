import React from "react";
import {withStyles, Container, Typography, Fab} from "@material-ui/core";
import {Pause as PauseIcon, PlayArrow as PlayArrowIcon, Stop as StopIcon} from '@material-ui/icons';
import Sweeper from './Sweeper';
import styles from './styles';

class BRIR extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      playing: false
    };
    this.sweeper = new Sweeper();
    this.sweeps = [];
    this.play = this.play.bind(this);
  };

  async play() {
    if (this.state.playing) {
      this.sweeper.stop();
    } else {
      this.setState({ playing: true });
      this.sweeper.record().then(recording => {
        console.log('recording length', recording.getChannelData(0).length / this.sweeper.sampleRate, 's');
      });
      await this.sweeper.playSequence(['FL', 'FR']);
      this.setState({ playing: false });
      setTimeout(() => { this.sweeper.stop(); }, 2000);
    }
  };

  render() {
    return (
      <Container>
        <Typography variant='h1'>BRIR</Typography>
        <Fab color="primary" onClick={this.play}>
          {!this.state.playing && (<PlayArrowIcon />)}
          {this.state.playing && (<StopIcon />)}
        </Fab>
      </Container>
    )
  };

}

export default withStyles(styles)(BRIR);
