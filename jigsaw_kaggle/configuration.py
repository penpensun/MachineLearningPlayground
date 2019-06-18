import configparser;

if 'config' not in globals():
    # global config 
    config = configparser.ConfigParser();
    config.read('config.ini')
    config = config['Default'];
    #print(config);
    # get the int parameters
    # global feature_size;
    # global batch_size;
    # global hidden_size;
    # global embedding_size;
    # global epoch_size;
    # global toxic_comment_input_path;
    # global short_toxic_comment_input_path;
    # global bcolz_embedding_path;
    # global word2idx_path;
    # global target_col_name;
    # global result_file;
    # global train_loss_file;
    # global val_loss_file;
    # global train_accuracy_file;
    # global val_accuracy_file;
    # global model_file;

    feature_size = int(config['feature_size'], 10);
    batch_size = int(config['batch_size'], 10);
    hidden_size = int(config['hidden_size'], 10);
    embedding_size = int(config['embedding_size'], 10);
    epoch_size = int(config['epoch_size'], 10);
    toxic_comment_input_path = config['toxic_comment_input_path']
    short_toxic_comment_input_path = config['short_toxic_comment_input_path'];
    bcolz_embedding_path = config['bcolz_embedding_path'];
    word2idx_path = config['word2idx_path'];
    target_col_name = config['target_col_name'];
    result_file = config['result_file'];
    train_loss_file = config['train_loss_file'];
    val_loss_file = config['val_loss_file'];
    train_accuracy_file = config['train_accuracy_file'];
    val_accuracy_file = config['val_accuracy_file'];
    model_file = config['model_file'];
    processed_toxic_comment_input_path = config['processed_toxic_comment_input_path']
    pretrained_embeds_path = config['pretrained_embeds_path']


#print(toxic_comment_input_path);
#print(short_toxic_comment_input_path);
#print(model_file);