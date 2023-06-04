# VPN (View Proposal Net)
This project uses code from ["Good View Hunting: Learning Photo Composition from Dense View Pairs"](https://github.com/zijunwei/ViewProposalNet) by [**Zijun Wei**]

The original project was implemented in `Tensorflow 1.3`. Here, we have implemented it in `PyTorch`.

## Running

(if you use Docker)
1. Navigate to the docker directory where the Dockerfile is located, and build your Docker image with the following command:
```
docker build -t pytorch:ver1 .
```
2. Next, run your Docker container using the following command:
```
docker run --name VPN_ver1 --gpus all -it -v /path/to/your/ViewProposalNet/src:/home/VPN/src pytorch:ver1
```
In this command, `/path/to/your/ViewProposalNet/src` should be replaced with the path to the project source code on your local system.
`VPN_ver1` is the name of the Docker container you are creating, and `pytorch:ver1` is the name of the Docker image you built earlier. The `--gpus all` option allows the Docker container to access all GPUs, if you are using GPU(s).
