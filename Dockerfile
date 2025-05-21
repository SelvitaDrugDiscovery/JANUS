FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG USER_NAME=ivan.smaliakou
ARG MY_UID=219600084
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 72
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 17
RUN apt-get update && \
      apt-get -y install sudo

RUN useradd --no-log-init -u ${MY_UID} -d /home/${USER_NAME} ${USER_NAME}

RUN usermod -aG sudo ${USER_NAME}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


# Its for python's USERBASE variable - all packages installed with --user flag will be located there
RUN sudo mkdir /home/${USER_NAME}
RUN sudo chmod og+w /home/${USER_NAME}


#WORKDIR /homes/${USER_NAME}
USER ${USER_NAME}
RUN sudo apt-get update -y
RUN sudo apt install software-properties-common -y
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN sudo apt update -y
RUN sudo apt install -y python3.10
RUN sudo apt-get install -y wget curl
RUN sudo curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN sudo python3.10 get-pip.py
RUN sudo rm get-pip.py
RUN sudo apt-get install libxrender1 -y
RUN sudo apt-get install libxrender1 libxtst6 libxi6 -y
RUN sudo apt-get install virtualenv -y

RUN echo 'alias python="python3.10"' >> ~/.bashrc
RUN echo 'alias python3="python3.10"' >> ~/.bashrc

RUN sudo apt-get install virtualenv -y
RUN sudo pip install --upgrade "numpy<2.0"
RUN sudo python3.10 -m pip install rdkit-pypi
RUN sudo python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN sudo python3.10 -m pip install selfies
RUN sudo python3.10 -m pip install joblib
RUN sudo python3.10 -m pip install scikit-learn
RUN sudo python3.10 -m pip install numpy
RUN sudo python3.10 -m pip install pandas


WORKDIR /app

COPY --chmod=777 ./src/ /app/src
COPY ./tests/ /app/tests
COPY janus_run.py /app
COPY setup.cfg /app
COPY pyproject.toml /app
COPY svr_model.pkl /app
COPY alex_rings_filters/ /app/alex_rings_filters

RUN pip install -e .

# ENTRYPOINT ["python", "janus_run.py"]
