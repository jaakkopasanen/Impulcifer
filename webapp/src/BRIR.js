import React from "react";
import {withStyles, Container, Box, Typography} from "@material-ui/core";
import styles from './styles';

class BRIR extends React.Component {
  constructor(props) {
    super(props);
  };

  render() {
    return (
      <Container>
        <Typography variant='h1'>BRIR</Typography>
      </Container>
    )
  };

}

export default withStyles(styles)(BRIR);
