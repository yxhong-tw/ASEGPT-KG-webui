import { useState } from 'react';
import Layout from '@/components/Layout';
import Graph, { Options, Edge, Node } from '@/components/Graph';
import { graphTestData } from '@/constants/graphTestData';

export default function Home() {
    const nodes: Node[] = graphTestData.nodes;
    const edges: Edge[] = graphTestData.edges;
    const options: Options = {
        locale: 'tw',
        nodes: {
            shape: 'dot',
            scaling: {
                customScalingFunction: function (min, max, total, value) {
                    if (value && total) return (value / total) * 0.5;
                    return min || 5;
                },
                min: 5,
                max: 300,
            },
            interaction: { hover: true },
        },
        edges: {
            scaling: {
                customScalingFunction: function (min, max, total, value) {
                    if (value && total) return value / total;
                    return min || 1;
                },
                min: 1,
                max: 10,
            },
            font: {
                // Set to the default colors as per the documentation
                color: '#343434',
                strokeColor: '#ffffff',
            },
        },
    };

    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    return (
        <Layout selectedIdx={0}>
            <div className="h-full flex flex-wrap p-7">
                <div
                    className="p-8 bg-white rounded-2xl shadow "
                    style={{ width: '45%' }}
                >
                    <div>{selectedNode?.label}</div>
                </div>
                <div className="h-full" style={{ width: '1.5%' }}></div>
                <div
                    className="h-full bg-white rounded-2xl shadow relative "
                    style={{ width: '52.5%' }}
                >
                    <Graph
                        nodes={nodes}
                        edges={edges}
                        options={options}
                        handleSelectNode={setSelectedNode}
                    />
                </div>
            </div>
        </Layout>
    );
}
