FROM python:3.10.9-slim

WORKDIR /backend

COPY ./requirements.txt /backend/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

COPY ./src /backend/app

# TODO: upgrade this to a production command
# https://www.uvicorn.org/deployment/#gunicorn
# CMD [ "/bin/bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 4000 WORKERS_PER_CORE=2"]
CMD [ "/bin/bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 4"]