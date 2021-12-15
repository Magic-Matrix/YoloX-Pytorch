

def print_line():
    print('-' * 73)

def print_one_data(key, value, temp = "|{:^30s}|{:^40s}|"):
    print(temp.format(key, value))
    print_line()


def train_data_text(opt):
    print()
    print_line()
    print('|{:^71s}|'.format("Model List"))
    print_line()
    print_one_data("Model name", "YoloX_" + opt.model_type)
    print_one_data("GPU", str(opt.GPU))
    print_one_data("Class number", str(opt.class_names))
    print_one_data("Image shape", str(opt.input_shape) + 'x' + str(opt.input_shape))
    print_one_data("Batch size", str(opt.batch_size))
    print_one_data("Epoch", str(opt.epoch))
    print_one_data("Learning rate", str(opt.lr))
    print_one_data("Optimizer", str(opt.optimizer))
    print_one_data("Freeze", str(opt.freeze))
    if opt.freeze:
        print_one_data("Freeze batch size", str(opt.freeze_batch_size))
        print_one_data("Freeze epoch", '0 ~ ' + str(opt.freeze_epoch))
        print_one_data("Freeze learning rate", str(opt.freeze_lr))
    print_one_data("Cosine scheduler", str(opt.cosine))
    print_one_data("Mosaic", str(opt.mosaic))
    print_one_data("Number workers", str(opt.num_workers))
    print_one_data("Train data path", opt.train_annotation_path)
    print_one_data("Val data path", opt.val_annotation_path)

