import React from "react";
import {withStyles, Container, Box, Typography} from "@material-ui/core";
import styles from './styles';

class HeadphoneCompensation extends React.Component {
  constructor(props) {
    super(props);
  };

  render() {
    return (
      <Container>
        <Typography variant='h1'>Headphone Compensation</Typography>
      </Container>
    )
  };

}

export default withStyles(styles)(HeadphoneCompensation);
