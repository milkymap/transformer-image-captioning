# transformer-image-captioning
Implementation of the paper CPTR : FULL TRANSFORMER NETWORK FOR IMAGE CAPTIONING

<p align="center"> 
  <img src="./static/cptr_architecture.jpg">
  <h3 align="center">
    architecture of the CPTR model for image captioning 
  </h3>  
</p>

---
---



# contents
* [structure](#structure)
* [prerequisites](#prerequisites)
* [installation](#installation)

# structure

this project is structured in a modular way, it is linked to several libraries such as **[pytorch, clip(openai)]** 
It contains the :
* following directories and files:
    * core
        * this is the main file of the project
        * it contains the definition of the transformer
        * it is based on the paper Attention Is All You Need 
        * i added some modification for handling multiple outpt of the decoder
    * dataset
        * this file contains two classes :
        * DatasetForFeaturesExtraction 
        * DatasetForTraining 
    * models
        * this file conains the definition of the CPTR model 
        * it uses the transformer defined on the core module 
        * it has some additional modules like : token_embedding, prediction_head       
    * libraries
        * contains usefull function such as : 
        * log handler 
        * tokenization 
        * features extraction 
        * model loading
        * beam and greedy search for caption generation   
    * static
        * contains image and font for the readme
    * main
        * this is the entrypoint of the program
        * it define three subcommand 
        * processing : for features extraction and tokenization 
        * learning   : training loop of the CPTR 
        * describe   : generate caption by taking an image path 
    * .gitignore
    * .dockerignore
    * Dockerfile.gpu
    * LICENCE
    * README.md 

# prerequisites
* git
* python3
* python3-venv
* docker  

# clone the repo 
```bash
    git clone https://github.com/Milkymap/transformer-image-captioning
    cd transformer-image-captioning 
    mkdir models images source target 
```

# docker build and run 
```bash
    docker build -t capformer:0.0 -f Dockerfile.gpu
```

# docker run processing step 
```bash
    docker run 
        --rm 
        --tty 
        --name capformer 
        --gpus all 
        -v $(pwd)/source:/home/solver/source 
        -v $(pwd)/models:/home/solver/models 
        -v $(pwd)/target:/home/solver/target
        -v $(pwd)/images:/home/solver/images  
        -e TERM=xterm-256color 
        capformer:0.0 processing 
            --path2images /home/solver/source/images 
            --path2captions /home/solver/source/captions.json 
            --path2vectorizer /home/solver/models/resnet152.th 
            --extension jpg 
            --path2features /home/solver/target/map_img2features.pkl 
            --path2tokenids /home/solver/target/zip_img2tokenids.pkl 
            --path2vocabulary /home/solver/target/vocabulary.pkl
```


# docker run learning step
```bash
    docker run 
        --rm 
        --tty 
        --name capformer 
        --gpus all 
        -v $(pwd)/source:/home/solver/source 
        -v $(pwd)/models:/home/solver/models 
        -v $(pwd)/target:/home/solver/target
        -v $(pwd)/images:/home/solver/images  
        -e TERM=xterm-256color 
        capformer:0.0
        learning 
            --path2features /home/solver/target/map_img2features.pkl 
            --path2tokenids /home/solver/target/zip_img2tokenids.pkl 
            --path2vocabulary /home/solver/target/vocabulary.pkl 
            --nb_epochs 92 
            --bt_size 128 
            --path2checkpoint /home/solver/models/checkpoint_128.th 
            --checkpoint 16 
            --start 0
``` 

# docker run describe step
```bash
    docker run 
        --rm 
        --tty 
        --name capformer 
        --gpus all 
        -v $(pwd)/source:/home/solver/source 
        -v $(pwd)/models:/home/solver/models 
        -v $(pwd)/target:/home/solver/target 
        -v $(pwd)/images:/home/solver/images 
        -e TERM=xterm-256color 
        capformer:0.0 
        describe 
            --path2vectorizer /home/solver/models/resnet152.th 
            --path2ranker /home/solver/models/ranker.pkl 
            --path2vocabulary /home/solver/target/vocabulary.pkl 
            --path2checkpoint /home/solver/models/checkpoint_128.th 
            --beam_width 17 
            --path2image /home/solver/images/bob.jpg
```