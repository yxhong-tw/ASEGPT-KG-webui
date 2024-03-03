import axios from 'axios';
import { KnowledgeGraphQueryItem } from './types';

const instance = axios.create({
    baseURL: '/api',
});

export const postQuery = async (query: KnowledgeGraphQueryItem) => {
    return await instance.post('/query', query);
};
