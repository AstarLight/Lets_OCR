import ConfigParser


if __name__ == '__main__':
    cp = ConfigParser.ConfigParser()
    cp.add_section('global')
    cp.set('global', 'using_cuda', 'True')
    cp.set('global', 'epoch', '30')
    cp.set('global', 'gpu_id', '6')
    cp.set('global', 'display_file_name', 'False')
    cp.set('global', 'display_iter', '1')
    cp.set('global', 'val_iter', '30')
    cp.set('global', 'save_iter', '100')
    cp.add_section('parameter')
    cp.set('parameter', 'lr_front', '0.001')
    cp.set('parameter', 'lr_behind', '0.0001')
    cp.set('parameter', 'change_epoch', '9')
    with open('../config', 'w') as fp:
        cp.write(fp)
