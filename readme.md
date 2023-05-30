## How to get environment set up on apple silicon?

Have a look [here](https://developer.apple.com/metal/tensorflow-plugin/)

> Or you can optionally install [Miniforge3](https://github.com/conda-forge/miniforge/)\
> More hands on elaboration can be found [here](https://github.com/mrdbourke/m1-machine-learning-test)\
> My recommendation is to use **miniconda3**

## More on installing conda, creating conda environment, tensorflow and other libs

1. ``conda create --prefix ./env --name sign-language-detection``
2. ``conda activate ./env``
## Time to install tensorflow on M1
3. ``conda install -c apple tensorflow-deps``
4. ``pip install tensorflow-metal``
5. ``pip install tensorflow-macos``

## Now it's time for opencv
6. Installing naively opencv for m1 will fail terribly. But there is a workaround. Building from the source.
Follow [this](https://caffeinedev.medium.com/building-and-installing-opencv-on-m1-macbook-c4654b10c188)

## mediapipe
7. Use protobuf==3.20.1\
``pip install protobuf==3.20.1``

Other regular packages can be installed through **requirements.txt** using pip's recursive flag
> ``pip install -r requirements.txt``

# Possible issues I encountered while dealing with this project
1. RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd [#4281](https://github.com/freqtrade/freqtrade/issues/4281)
---
Potential solution
> Upgrade numpy via
> ``pip install numpy --upgrade``

---
# Final notes for evaluation
~~Please switch to [`submission`](https://github.com/parthokr/asl-recognition/tree/submission) branch for the final submission.~~

You may open kaggle [`notebook`](https://www.kaggle.com/code/nayeemrocks22/asl-recognition/notebook) and run the code there.


