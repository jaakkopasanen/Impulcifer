import React from "react";
import {withStyles, Container, Box, Typography} from "@material-ui/core";
import styles from './styles';

class Levels extends React.Component {
  constructor(props) {
    super(props);
  };

  render() {
    return (
      <Container>
        <Typography variant='h1'>Levels</Typography>
      </Container>
    )
  };

}

export default withStyles(styles)(Levels);
