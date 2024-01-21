import { useState } from 'react';
import { Options, Edge, Node } from 'vis-network/standalone/esm/vis-network';
import useVisNetwork from '@/hooks/useVisNetwork';

interface GraphProps {
    nodes: Node[];
    edges: Edge[];
    options: Options;
    handleSelectNode?: (node: Node) => void;
}

export { type Options, type Edge, type Node };

export default function Graph(props: GraphProps) {
    const { nodes, edges, options, handleSelectNode } = props;
    const { ref, network } = useVisNetwork({
        options,
        edges,
        nodes,
    });

    const handleClick = () => {
        if (!network) return;
        network.fit();
    };

    const [isCollapsed, setIsCollapsed] = useState(false);
    // const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    network?.on('selectNode', (params) => {
        const node = params.nodes[0];
        const nodeOptions = network.body.nodes[node].options;
        const nodeData = nodeOptions.data;

        handleSelectNode && handleSelectNode(nodeOptions);
        // setSelectedNode(nodeOptions);
    });

    return (
        <div className="h-full flex flex-wrap flex-row">
            <div className="w-full relative top-0 h-20 p-5 px-10 border-b-2 border-black border-opacity-40 flex flex-wrap flex-row gap-6">
                <button className="btn btn-primary" onClick={handleClick}>
                    Resize
                </button>
                <button className="btn btn-primary" onClick={handleClick}>
                    Resize
                </button>
            </div>
            <div className={'w-full h-full relative top-0 p-3 '} ref={ref} />
            <div
                className="w-full p-8 absolute left-0 bottom-0 bg-gray-200 shadow-sm cursor-pointer rounded-2xl transition-[height] duration-700"
                style={{
                    height: isCollapsed ? '1%' : '33%',
                }}
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                test
            </div>
        </div>
    );
}
