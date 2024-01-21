export const graphTestData = {
    nodes: [
        {
            id: 1,
            label: '測試1',
            title: '測試1',
            // level: 1,
            group: 'struct',
            value: 7,
            data: {
                topic: '測試1',
            }
        },
        {
            id: 2,
            label: '測試2',
            title: '測試2',
            // level: 2,
            group: 'struct',
            value: 6,
            data: {
                topic: '測試2',
            }
        },
        {
            id: 3,
            label: '測試3',
            title: '測試3',
            // level: 3,
            group: 'object',
            value: 5,
            data: {
                topic: '測試3',
            }
        },
        {
            id: 4,
            label: '測試4',
            title: '測試4',
            // level: 4,
            group: 'market',
            value: 5,
            data: {
                topic: '測試4',
            }
        },
        {
            id: 5,
            label: '測試5',
            title: '測試5',
            // level: 5,
            group: 'object',
            value: 2,
            data: {
                topic: '測試5',
            }
        },
        {
            id: 6,
            label: '測試6',
            title: '測試6',
            // level: 4,
            group: 'market',
            value: 1,
            data: {
                topic: '測試6',
            }
        },
        {
            id: 7,
            label: '測試7',
            title: '測試7',
            // level: 3,
            group: 'object',
            value: 1,
            data: {
                topic: '測試7',
            }
        },
    ],
    edges: [
        { from: 1, to: 2, id: 1, label: 'test', value: 6 },
        { from: 1, to: 3, id: 6, value: 3 },
        { from: 2, to: 3, id: 2, value: 5 },
        { from: 3, to: 5, id: 3, value: 7 },
        { from: 3, to: 4, id: 4, value: 4 },
        { from: 4, to: 5, id: 5, value: 3 },
        { from: 3, to: 6, id: 7, value: 2 },
        { from: 1, to: 7, id: 8, value: 8 },
        { from: 1, to: 7, id: 10, value: 1 },
        { from: 2, to: 7, id: 9, value: 1 },
    ],
};
