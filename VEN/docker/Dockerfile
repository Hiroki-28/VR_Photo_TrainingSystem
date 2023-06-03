FROM pytorch/pytorch:latest
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install libgl1-mesa-dev -y
RUN apt-get install libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install opencv_contrib_python==4.4.0.46
RUN pip install matplotlib
RUN pip install pandas
RUN pip install progressbar2
WORKDIR /home/VEN/src

