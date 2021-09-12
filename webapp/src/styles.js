const styles = (theme) => ({
    root: {
        backgroundColor: '#eee',
    },
    navigation: {

    },
    navigationButton: {
        width: '100%',
        justifyContent: 'flex-start',
        fontSize: '1rem',
        padding: `${theme.spacing(2)}px ${theme.spacing(1)}px`,
        textTransform: 'none',
        '& svg': {
            marginRight: theme.spacing(2),
        }
    },
    active: {
        color: theme.palette.primary.main,
    }
});

export default styles;
