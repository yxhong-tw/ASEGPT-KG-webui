import React, { Component } from 'react';
import styles from '@/styles/Blocker.module.css';


interface BlockerProps {
    display?: boolean;
    clickable?: boolean;
    style?: any;
}

interface BlockerState {
    display: boolean;
}

export default class Blocker extends Component<BlockerProps, BlockerState> {
    static defaultProps = {
        display: false,
        clickable: true,
        style: {}
    };

    constructor(props: BlockerProps) {
        super(props);

        this.state = { display: props.display } as BlockerState;
    }

    _disabled = (event: React.MouseEvent<HTMLDivElement>) => {
        this.setState({ display: false });
    };

    componentDidUpdate(prevProps: BlockerProps, prevState: BlockerState) {
        if (prevProps.display !== this.props.display) {
            this.setState({ display: this.props.display } as BlockerState);
        }
    }

    render() {
        const { display } = this.state;

        return (
            <div className={styles.blocker}
                style={{ ...this.props.style, ...{ display: (display ? '  block' : ' hidden') } }}
                onClick={this.props.clickable ? this._disabled : undefined}>
                {this.props.children}
            </div>
        );
    }
}