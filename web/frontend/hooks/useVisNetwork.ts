import { useState, useRef, useEffect } from 'react';
import {
    Network,
    Options,
    Data,
    Edge,
    Node,
} from 'vis-network/standalone/esm/vis-network';

export interface UseVisNetworkOptions {
    options: Options;
    nodes: Node[];
    edges: Edge[];
}

export default (props: UseVisNetworkOptions) => {
    const { edges, nodes, options } = props;

    const [network, addNetwork] = useState<Network | null>(null);
    const ref = useRef<HTMLDivElement>(null);

    const data: Data = { nodes, edges };

    useEffect(() => {
        if (ref.current) {
            const instance = new Network(ref.current, data, options);
            addNetwork(instance);
        }
        return () => network?.destroy();
    }, []);

    return {
        network,
        ref,
    };
};
