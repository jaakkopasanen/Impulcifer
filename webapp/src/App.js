import React from "react";
import { withStyles, Grid, Button, Box } from "@material-ui/core";
import { Info as InfoIcon, Headset as HeadsetIcon, Speaker as SpeakerIcon, Home as HomeIcon, Equalizer as EqualizerIcon } from '@material-ui/icons';
import styles from './styles';
import Info from './Info';
import Levels from './Levels';
import RoomCorrection from "./RoomCorrection";
import BRIR from "./BRIR";
import HeadphoneCompensation from "./HeadphoneCompensation";

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      stepName: 'brir'
    };

    this.steps = [
      {name: 'info', label: 'Info', icon: <InfoIcon />},
      {name: 'levels', label: 'Levels', icon: <EqualizerIcon />},
      {name: 'roomCorrection', label: 'Room Correction', icon: <HomeIcon />},
      {name: 'brir', label: 'BRIR', icon: <SpeakerIcon />},
      {name: 'headphoneCompensation', label: 'Headphone Compensation', icon: <HeadsetIcon />},
    ];
  };

  selectStep(stepName) {
    this.setState({stepName});
  };

  render() {
    const {classes} = this.props;
    const {stepName} = this.state;
    return (
      <Box display='flex' flexDirection='row'>
        <Box display='flex' flexDirection='column' justifyContent='flex-start' alignItems='stretch'>
          {
            this.steps.map((step) => {
              const clss = [classes.navigationButton];
              if (stepName === step.name) clss.push(classes.active);
              return (
                <Grid item key={step.label}>
                  <Button className={clss.join(' ')} onClick={() => {
                    this.selectStep(step.name);
                  }}>
                    {step.icon} {step.label}
                  </Button>
                </Grid>
              )
            })
          }
        </Box>
        {stepName === 'info' && (<Box display="flex"><Info /></Box>)}
        {stepName === 'levels' && (<Box display="flex"><Levels /></Box>)}
        {stepName === 'roomCorrection' && (<Box display="flex"><RoomCorrection /></Box>)}
        {stepName === 'brir' && (<Box display="flex"><BRIR playSweep={this.playSweep}/></Box>)}
        {stepName === 'headphoneCompensation' && (<Box display="flex"><HeadphoneCompensation /></Box>)}
      </Box>
    )
  };
}

export default withStyles(styles)(App);
