import { useState } from 'react';
import { useQuery } from 'react-query';
import { useSearchParams } from 'next/navigation';
import Layout from '@/components/Layout';
import Graph, { Options, Edge, Node } from '@/components/Graph';
import { postQuery } from '@/core/api';
import { KnowledgeGraphQueryItem } from '@/core/types';



export default function KG() {
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    const searchParams = useSearchParams();
    const query = searchParams.get('query');

    const response = useQuery('query', async () => postQuery({ 'query': String(query) } as KnowledgeGraphQueryItem))

    if (response.status === 'loading') {
        return <div>Loading...</div>;
    } else if (response.status === 'error') {
        return <div>Error: {response.error}</div>;
    }

    // console.log(response.data.data);
    const relevantNodes = (response as any).data.data.relevant_nodes;
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const existingNodes: string[] = [];
    Object.keys(relevantNodes).forEach((nodeID: string) => {
        if (relevantNodes[nodeID].hasOwnProperty('triplets')) {
            const triplets = new Set<String>(relevantNodes[nodeID]['triplets']);
            for (let triplet_string of triplets) {
                const triplet = triplet_string.split(', ');
                if (triplet.length !== 3) continue;
                const [headEntity, relation, tailEntity] = triplet;

                edges.push({
                    from: `${nodeID}-head-${headEntity}`,
                    to: `${nodeID}-tail-${tailEntity}`,
                    id: `${nodeID}-relation-${headEntity}-${relation}-${tailEntity}`,
                    label: relation,
                    value: 1
                });

                [headEntity, tailEntity].forEach((entity: string, index: Number) => {
                    let entityID = '';
                    if (!index)
                        entityID = `${nodeID}-head-${headEntity}`;
                    else
                        entityID = `${nodeID}-tail-${tailEntity}`;

                    if (!existingNodes.includes(entityID)) {
                        const node: Node = {
                            id: entityID,
                            label: entity,
                            title: entity,
                            group: nodeID,
                            value: 1,
                        };
                        nodes.push(node);
                        existingNodes.push(entityID);
                    }
                });
            }
        }
    });

    edges.forEach((edge: Edge) => {
        (nodes.find((node: Node) => node.id === edge.from) as any).value += 1;
        (nodes.find((node: Node) => node.id === edge.to) as any).value += 1;
    });

    const options: Options = {
        locale: 'tw',
        nodes: {
            shape: 'dot',
            scaling: {
                customScalingFunction: function (min, max, total, value) {
                    if (value && total) return (value / total);
                    return min || 20;
                },
                min: 20,
                max: 200,
            }
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
            }
        }
    };



    return (
        <Layout selectedIdx={0}>
            <div className="h-full flex flex-wrap p-7">
                <div
                    className="p-8 bg-white rounded-2xl shadow overflow-y-scroll h-full"
                    style={{ width: '45%' }}
                >
                    <div className="text-xl font-semibold">Query: {query}</div>
                    {
                        selectedNode ? (
                            <div>
                                <hr className="my-5 border-2 border-black" />
                                <div className="text-lg font-semibold my-3">{selectedNode?.label}</div>
                            </div>
                        ) : null
                    }
                    <div className="w-full">
                        {
                            edges.filter((edge: Edge) => edge.from === selectedNode?.id || edge.to === selectedNode?.id)
                                .map((edge: Edge) => {
                                    const { from, to, label } = edge;
                                    const fromNode = nodes.find((node: Node) => node.id === from);
                                    const toNode = nodes.find((node: Node) => node.id === to);
                                    const headEntity = fromNode?.label;
                                    const tailEntity = toNode?.label;
                                    return (
                                        <div key={`triplet-${headEntity}-${label}${tailEntity}`} className="mt-3">
                                            ({headEntity}, {label}, {tailEntity})
                                        </div>
                                    );

                                })
                        }
                        <hr className="my-5 border-2 border-black" />
                        <div className="text-xl font-semibold">相關新聞</div>
                        {
                            Object.keys(relevantNodes).map((nodeID: string, i: number) => {
                                const node = relevantNodes[nodeID];
                                if (node.hasOwnProperty('content')) {
                                    return (
                                        <div key={`relevant-article-${i}`} className="mt-3 leading-loose">
                                            <div className="flex flex-row flex-wrap gap-2 my-3">
                                                <span className="btn btn-danger">Relevance Score: {Math.round(Number(node.score) * 1000) / 1000}</span>
                                                <span className="btn btn-primary">{node.metadata['article_title']}</span>
                                                <span className="btn btn-success">{node.metadata['source_name']}</span>
                                                {
                                                    node.metadata['article_creation_date'] && <span className="btn btn-warning">{node.metadata['article_creation_date'].match(/^\d{4}-\d{2}-\d{2}/)[0]}</span>
                                                }
                                            </div>
                                            {node['content']}
                                            <hr className="my-5 border-1 border-black" />
                                        </div>
                                    );
                                }
                            })
                        }
                    </div>
                </div>
                <div className="h-full" style={{ width: '1.5%' }}></div>
                <div
                    className="h-full bg-white rounded-2xl shadow relative "
                    style={{ width: '52.5%' }}
                >
                    <Graph
                        rowData={(response as any).data.data}
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


export async function getServerSideProps({ query }: any) {
    if (!query.query) {
        return {
            notFound: true,
        };
    }

    return {
        props: {},
    };
}