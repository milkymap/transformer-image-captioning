from pickletools import optimize
import click 

from torch.utils.data import DataLoader

from os import getenv, path 
from time import sleep 
from rich.progress import track
from dataset import DatasetForFeaturesExtraction, DatasetForTraining
from libraries.log import logger
from libraries.strategies import * 

from model import CaptionTransformer

"""
    docker run --rm --tty --name nlogn --gpus all -v $(pwd)/source:/home/solver/source -v $(pwd)/models:/home/solver/models -v $(pwd)/target:/home/solver/target -e TERM=xterm-256color nlogn-cap:0.0 processing --path2images /home/solver/source/images --path2captions /home/solver/source/captions.json --path2vectorizer /home/solver/models/resnet152.th --extension jpg --path2features /home/solver/target/map_img2features.pkl --path2tokenids /home/solver/target/zip_img2tokenids.pkl --path2vocabulary /home/solver/target/vocabulary.pkl
    docker run --rm --tty --name nlogn --gpus all -v $(pwd)/source:/home/solver/source -v $(pwd)/models:/home/solver/models -v $(pwd)/target:/home/solver/target -e TERM=xterm-256color nlogn-cap:0.0 learning --path2features /home/solver/target/map_img2features.pkl --path2tokenids /home/solver/target/zip_img2tokenids.pkl --path2vocabulary /home/solver/target/vocabulary.pkl --nb_epochs 92 --bt_size 128 --path2checkpoint /home/solver/models/checkpoint_###.th --checkpoint 16 --start 0
    docker run --rm --tty --name nlogn --gpus all -v $(pwd)/source:/home/solver/source -v $(pwd)/models:/home/solver/models -v $(pwd)/target:/home/solver/target -v $(pwd)/images:/home/solver/images -e TERM=xterm-256color nlogn-cap:0.0 describe --path2vectorizer /home/solver/models/resnet152.th --path2ranker /home/solver/models/ranker.pkl --path2vocabulary /home/solver/target/vocabulary.pkl --path2checkpoint /home/solver/models/checkpoint_###.th --beam_width 16 --path2image /home/solver/images/007.jpg
"""

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=True)
@click.pass_context
def router_command(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug 
    command = ctx.invoked_subcommand 
    if command is None:
        logger.debug('no command was called, add --help option to see the avaiables command')
    else:
        logger.debug(f'{command} was called')

@router_command.command()
@click.option('--path2vectorizer', help='path to models for features extraction', type=click.Path(False))
@click.option('--path2images', help='path to images directory', type=click.Path(True))
@click.option('--path2captions', help='path to captions json file', type=click.Path(True))
@click.option('--extension', help='image file extension', type=click.Choice(['jpg', 'jpeg']))
@click.option('--path2features', help='path to features dump location', type=click.Path(False))
@click.option('--path2tokenids', help='path to tokenids dump lication', type=click.Path(False))
@click.option('--path2vocabulary', help='path to vacabulary dump location', type=click.Path(False))
def processing(path2vectorizer, path2images, path2captions, extension, path2features, path2tokenids, path2vocabulary):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    with open(file=path2captions, mode='r') as fp:
        img2captions = json.load(fp)
    
    captions = list(img2captions.values())
    captions = list(it.chain(*captions))

    tokenizer = build_tokenizer(tok_name='spacy', lang='en_core_web_sm')
    vocabulary = make_vocab(captions, tokenizer, SPECIALS2IDX)
    logger.success('vocaulary was built')
    
    serialize(path2vocabulary, vocabulary)

    bos = th.tensor([SPECIALS2IDX['<bos>']])
    eos = th.tensor([SPECIALS2IDX['<eos>']])
        
    zip_img2tokenids = []
    logger.debug('caption tokenization')
    for key, val in track(img2captions.items(), 'build map_img2tokenids'):
        for cap in val:
            tok = tokenizer(cap.strip().lower())
            idx = th.tensor(vocabulary(tok))
            seq = th.cat([bos, idx, eos]).numpy()  # more effective for storage 
            zip_img2tokenids.append((key, seq))
    
    serialize(path2tokenids, zip_img2tokenids)

    logger.debug('features extraction loading')
    vectorizer = load_vectorizer(path2vectorizer)
    vectorizer.eval()
    vectorizer.to(device)

    dataset = DatasetForFeaturesExtraction(path2images, f'*.{extension}')

    logger.debug('extraction will start')
    accumulator = []
    for sections in track(dataset, 'features extraction'):
        embedding = extract_features(vectorizer, sections[None, ...].to(device)).squeeze(0) # (2048, 7, 7)
        embedding = th.flatten(embedding, start_dim=1).T.cpu().numpy()  # 49, 2048
        accumulator.append(embedding)
    
    image_names = dataset.image_names
    accumulator = np.stack(accumulator)  # stack over batch axis ==> (nb_images, 49, 512)
    logger.debug(f'accumulated features shape : {accumulator.shape}')
    assert len(image_names) == len(accumulator)
    map_img2features = dict(zip(image_names, accumulator)) 

    serialize(path2features, map_img2features)

    logger.success('features, tokenids and vocabulary were saved')

@router_command.command()
@click.option('--path2vocabulary', help='path to vacabulary dump location', type=click.Path(True))
@click.option('--path2features', help='path to features dump location', type=click.Path(True))
@click.option('--path2tokenids', help='path to tokenids dump lication', type=click.Path(True))
@click.option('--nb_epochs', help='number of epochs', type=int, default=128)
@click.option('--bt_size', help='batch size', type=int, default=32)
@click.option('--path2checkpoint', help='path to checkpoint model', type=click.Path(False))
@click.option('--checkpoint', help='checkpoint period(save model)', type=int, default=16)
@click.option('--start', help='start epoch index', type=int, default=0)
def learning(path2vocabulary, path2features, path2tokenids, nb_epochs, bt_size, path2checkpoint, checkpoint, start):
    basepath2models = getenv('MODELS')

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    logger.debug('load vocabulary')
    vocabulary = deserialize(path2vocabulary)
    nb_tokens = len(vocabulary)

    logger.debug('build dataset')
    dataset = DatasetForTraining(path2tokenids, path2features)
    logger.debug(f'size of the dataset : {len(dataset):05d}')
    dataloader = DataLoader(dataset, batch_size=bt_size, shuffle=True, collate_fn=custom_fn)
    nb_data = len(dataset)

    logger.debug('define network')
    if path.isfile(path2checkpoint):
        net = th.load(path2checkpoint)
    else:
        net = CaptionTransformer(
            in_dim=2048,
            hd_dim=256,
            ff_dim=512,
            nb_heads=8,
            num_encoders=5,
            num_decoders=5,
            pre_norm=False,
            seq_length=64,
            nb_tokens=nb_tokens,
            padding_idx=SPECIALS2IDX['<pad>'] 
        )
    
    net.to(device)
    net.train()
    
    print(net)

    optimizer = th.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIALS2IDX['<pad>'])
    logger.debug('training  will begin ...!')
    sleep(1)

    nb_epochs += start 
    for epoch in range(start, nb_epochs):
        counter = 0
        for src, tgt in dataloader:
            counter += len(tgt)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = build_mask(tgt_input).to(device)
            tgt_key_padding_mask = build_key_padding_mask(tgt_input, SPECIALS2IDX['<pad>']).to(device)
            
            memory = net.encode(src=src.to(device))
            output = net.decode(
                tgt=tgt_input.to(device), 
                memory=memory, 
                tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            logits = [net.generator(out) for out in output ]
            logits = [ th.flatten(prb, start_dim=0, end_dim=1) for prb in logits ]
            tgt_output = th.flatten(tgt_output)

            optimizer.zero_grad() 
            errors = [ criterion(prb, tgt_output.to(device)) for prb in logits ]
            error = sum(errors)
            error.backward()
            optimizer.step()

            message = []
            for err in errors:
                msg = f'{err.cpu().item():07.3f}'
                message.append(msg)
            message = ' | '.join(message)
            logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{counter:05d}/{nb_data:05d}] | Loss : {error.cpu().item():07.3f} >> {message}')
        # end for loop over batchs 
        
        if epoch % checkpoint == 0:
            path2network = path.join(basepath2models, f'checkpoint_{epoch:03d}.th')
            th.save(net.cpu(), path2network)
            net.to(device)
            logger.success(f'a snapshot was saved {path2network}')

    # end for loop over epochs 
    
    path2network = path.join(basepath2models, f'checkpoint_###.th')
    th.save(net.cpu(), path2network)
    logger.success(f'a snapshot was saved {path2network}')
    logger.success('end of training')



@router_command.command()
@click.option('--path2vectorizer', help='name of the stored model(features extractor)', type=str)
@click.option('--path2checkpoint', help='model snapshot filename', type=str)
@click.option('--path2image', help='image to describe', type=str)
@click.option('--path2vocabulary', help='vocabulary object', type=str)
@click.option('--beam_width', help='size of beam', type=int, default=7)
@click.option('--path2ranker', help='name of the ranker model', type=str)
def describe(path2vectorizer, path2checkpoint, path2image, path2vocabulary, beam_width, path2ranker):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    logger.debug('env variables loading')
    logger.debug('features, vocab and token_ids loading')
    
    if path.isfile(path2checkpoint):
        logger.debug('model(snapshot) will be loaded')
        net = th.load(path2checkpoint)
        net.to(device)
        net.eval()

    vocab = deserialize(path2vocabulary)
    logger.debug(f'vocab was loaded | len => {len(vocab)}')
    
    logger.debug(f'load features extractor')

    vectorizer = load_vectorizer(path2vectorizer)
    vectorizer.eval()
    vectorizer.to(device)

    logger.debug('load ranker clip VIT model')
    ranker, processor = load_ranker(path2ranker, device)

    logger.debug('features extraction by resnet152')

    cv_image = read_image(path2image)
    th_image = cv2th(cv_image)
    th_image = prepare_image(th_image)

    embedding = extract_features(vectorizer, th_image[None, ...].to(device)).squeeze(0)
    output_batch = th.flatten(embedding, start_dim=1).T  # 49, 2048  
    
    response = beam_search(
        model=net, 
        source=output_batch[None, ...], 
        BOS=SPECIALS2IDX['<bos>'], 
        EOS=SPECIALS2IDX['<eos>'],
        max_len=64, 
        beam_width=beam_width,
        device=device, 
        alpha=0.7
    )
    
    logger.debug(f'nb generated : {len(response)}')
    sentences = []
    for sequence, _ in response:
        caption = vocab.lookup_tokens(sequence[1:-1])  # ignore <bos> and <eos>
        joined_caption = ' '.join(caption)
        sentences.append(joined_caption)
        
    logger.debug('ranking will begin...!')
    pil_image = cv2pil(cv_image)
    ranked_scores = rank_solutions(pil_image, sentences, ranker, processor, device)
    ranked_response = list(zip(sentences, ranked_scores))
    ranked_response = sorted(ranked_response, key=op.itemgetter(1), reverse=True)

    for caption, score in ranked_response:
        score = int(score * 100)
        logger.debug(f'caption : {caption} | score : {score:03d}')

if __name__ == '__main__':
    router_command(obj={})
