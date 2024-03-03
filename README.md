# ASEGPT-KG webui

|  | Service name |
| ---- | ---- |
|  **Frontend**    |  Next.js + TailwindCSS    |
|  **Backend**    |   Next.js (SSR) + FastAPI   |
| **RAG vector database** | NebulaGraph |
| **RAG generator** | Mistral |

## Screenshots

![Case Study 1](./screenshots/frontend-case-study-1.png)

![Case Study 2](./screenshots/frontend-case-study-2.png)

![Case Study 3](./screenshots/frontend-case-study-3.png)

## Getting started

First, install the NebulaGraph and start the service.

```bash
sh start-nebula.sh
```

Local Development (allows for updated changes on page refresh)

```bash
docker compose -f docker-compose-dev.yml up
```

Go to http://localhost:8080

Production Deployment

```bash
docker compose up
```
