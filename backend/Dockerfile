FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y python3.9 python3-pip python-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9
RUN apt-get install -y python3.9-distutils

WORKDIR /backend

COPY ./requirements.txt /backend/requirements.txt

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt
RUN pip install accelerate
RUN pip install -i https://pypi.org/simple/ bitsandbytes
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu116/torch_stable.html

COPY ./src /backend/app

CMD [ "/bin/bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 4"]