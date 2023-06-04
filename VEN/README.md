# VEN (View Evaluation Net)
This project uses code from ["Good View Hunting: Learning Photo Composition from Dense View Pairs"](https://github.com/zijunwei/ViewEvaluationNet/tree/master) by [**Zijun Wei**]

The original project did not provide files for training the model, so I created them. 

Just like the original project, it is implemented in `Pytorch`.

## Running

(if you use Docker)
1. Navigate to the docker directory where the Dockerfile is located, and build your Docker image with the following command:
```
docker build -t pytorch:ver1 .
```
2. Next, run your Docker container using the following command:
```
docker run --name VEN_ver1 --gpus all -it -v /path/to/your/ViewEvaluationNet/src:/home/VEN/src pytorch:ver1
```
In this command, `/path/to/your/ViewEvaluationNet/src` should be replaced with the path to the project source code on your local system.
`VEN_ver1` is the name of the Docker container you are creating, and `pytorch:ver1` is the name of the Docker image you built earlier. The `--gpus all` option allows the Docker container to access all GPUs, if you are using GPU(s).
