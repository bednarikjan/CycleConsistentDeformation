import argparse
import os
import datetime

# Project files.
import cyccon.auxiliary.my_utils as my_utils
import jblib.file_sys as jbfs


def Args2String(opt):
    my_str = ""
    for i in opt.__dict__.keys():
        if i == "model":
            if opt.__dict__[i] is None:
                my_str = my_str + str(0) + "_"
            else:
                my_str = my_str + str(1) + "_"
        else:
            my_str = my_str + str(opt.__dict__[i]) + "_"
    my_str = my_str.replace('/', '-')
    return my_str
    return my_str


def parser():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--n_train_iters', type=int, default=200000, help='Number of oters to train for.')
    parser.add_argument('--env', type=str, default="dfaust_", help='visdom environment')
    # parser.add_argument('--lr_decay_1', type=int, default=400, help='first learning rate decay')
    # parser.add_argument('--lr_decay_2', type=int, default=450, help='second learning rate decay')
    parser.add_argument('--lr_decay_1', type=int, default=160000, help='first learning rate decay')
    parser.add_argument('--lr_decay_2', type=int, default=180000, help='second learning rate decay')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='weight decay')
    parser.add_argument('--port', type=int, default=8890, help='visdom port')
    parser.add_argument(
        '--writer_period', type=int, default=50,
        help='Tensorboard reporting period.')
    # Losses
    parser.add_argument('--lambda_chamfer', type=float, default=1.0, help='lambda parameter for reconstruction')
    parser.add_argument('--chamfer_loss_type', type=str, default="ASSYM_2",
                        choices=['SYM', 'ASSYM_1', 'ASSYM_2'],
                        help='chamfer symmetric or assymetric, as discussed in the paper')
    parser.add_argument('--part_supervision', type=int, default=0, choices=[0, 1], help='Use supervision from parts')
    parser.add_argument('--lambda_cycle_2', type=float, default=1, help='lambda parameter for cycle loss of lenght 2')
    parser.add_argument('--lambda_cycle_3', type=float, default=1, help='lambda parameter for cycle loss of lenght 3')
    parser.add_argument('--lambda_reconstruct', type=float, default=0.0,
                        help='lambda parameter for reconstruction from known transfor')
    parser.add_argument('--epoch_reconstruct', type=int, default=30,
                        help='intialize with strong self-reconstruction for X epoch to start from identity')
    parser.add_argument('--iter_reconstruct', type=int, default=12000,
                        help='intialize with strong self-reconstruction for X iters to start from identity')
    parser.add_argument(
        "--accelerated_chamfer",
        type=int,
        default=1,
        help="use custom build accelarated chamfer",
    )

    # Data
    parser.add_argument(
        '--ds', type=str, default='dfaust', choices=['shapenet', 'dfaust'],
        help='Datasets choice.')

    # Data ShapeNet
    parser.add_argument('--cat', type=str, default="Car", help='Shapenet Category')
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object')
    parser.add_argument('--knn', type=int, default=0, choices=[0, 1],
                        help='Use knn or not to sample a triplet training sample')
    parser.add_argument('--num_neighbors', type=int, default=20, help='Number of neighbors computed for knn')
    parser.add_argument('--anisotropic_scaling', type=int, default=1, )
    parser.add_argument('--normalization', type=str, default="BoundingBox",
                        choices=['UnitBall', 'BoundingBox_2', 'BoundingBox'], help='Type of normalization used')
    parser.add_argument('--mode', type=str, default="TRAIN", choices=['TRAIN', 'ALLDATA', 'Fine_tune_test'], )

    # Data DFAUST
    parser.add_argument(
        '--dfaust_subject', type=str, help='Id of the subject.')
    parser.add_argument(
        '--dfaust_sequence', type=str, help='Name of the sequence.')
    parser.add_argument(
        '--dfaust_mode', type=str, choices=['random', 'within_seq', 'neighbors'],
        default='neighbors', help='Loading mode.')
    parser.add_argument(
        '--dfaust_max_frames', type=int, default=6,
        help='Number of neighboring frames to sample from. Only applicable if '
             '"dfaust_mode" = "neighbors"')

    # Save dirs and reload
    parser.add_argument(
        '--logdir_base', type=str,
        default='/cvlabdata2/home/jan/projects/metcon/trruns/02',
        help='Path to base dir to save the results')
    parser.add_argument(
        '--logdir', type=str, help='Name of the experiment folder.')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training at X epoch')
    parser.add_argument('--model', type=str, default='', help='optional reload model path')

    # Network
    parser.add_argument('--skip_connections', type=int, default=0, choices=[0, 1], help='Use skip connections')
    parser.add_argument('--hidden_sizes', default=[64, 64, 64, 64, 64, 64], nargs='+', type=int)
    parser.add_argument('--resnet_layers', default=False, type=int)
    parser.add_argument('--display', default=1, type=int)
    parser.add_argument('--encoder_type', type=str, default="Pointnet", )
    parser.add_argument('--reload', type=int, default=0, choices=[0, 1], )
    parser.add_argument('--batchnorm', default=True, type=int)

    # Eval parameters
    parser.add_argument('--number_points_eval', type=int, default=5000, help='Number of point sampled on the object')
    parser.add_argument('--num_shots_eval', type=int, default=50000, help='Number of train samples to take labels from')
    parser.add_argument('--num_figure_3_4', type=int, default=-1, help='Number of examples to save')
    parser.add_argument('--save_mesh', type=bool, default=False, help='save mesh examples optional')
    parser.add_argument('--k_max_eval', type=int, default=10, help='number of ensembles to consider')
    parser.add_argument('--randomize', type=int, default=0, choices=[0, 1],
                        help='randomize the seed to check the variance of the results')
    parser.add_argument('--atlasSphere_path', type=str, default="", help="path to Atlasnet models")
    parser.add_argument('--atlasPatch_path', type=str, default="", help="path to Atlasnet models")
    parser.add_argument('--train_atlasnet', type=int, default=0, choices=[0, 1], help="train atlasnet or main models")
    parser.add_argument('--atlasnet', type=str, default="SPHERE", choices=['SPHERE', 'PATCH'], )
    parser.add_argument('--eval_source', type=str, default="", help="Path to input and ouput in forward_input_output.py" )
    parser.add_argument('--eval_target', type=str, default="", help="Path to input and ouput in forward_input_output.py" )
    parser.add_argument('--eval_get_criterions_for_shape', type=str, default="", help="Path to input and ouput in forward_input_output.py" )
    parser.add_argument('--shapenetv1_path', type=str, default="", help="Path to shapenet v1" )
    parser.add_argument('--dest_folder', type=str, default="/cvlabdata2/home/jan/projects/3rd_party/CycleConsistentDeformation/cyccon/html",
                        help="folder to store global results of all experiments")

    parser.add_argument('-f', type=str, required=False, help="Just to make jupl work.")
    parser.add_argument('-v', '--verbose', action='store_true', required=False, help="Just to make jupl work.")
    parser.add_argument(
        '--n_iters', type=int, default=500, help='Number of sampled random pairs.')
    parser.add_argument(
        '--n_pts', type=int, default=3125, help='Number of predicted points.')
    parser.add_argument(
        '--pck_steps', type=int, default=100, help='Number of steps for PCK.')
    parser.add_argument(
        '--pck_min', type=float, default=0., help='PCK lower threshold.')
    parser.add_argument(
        '--pck_max', type=float, default=0.1, help='PCK upper threshold.')

    opt = parser.parse_args()
    opt.knn = my_utils.int_2_boolean(opt.knn)
    opt.part_supervision = my_utils.int_2_boolean(opt.part_supervision)
    opt.skip_connections = my_utils.int_2_boolean(opt.skip_connections)
    opt.train_atlasnet = my_utils.int_2_boolean(opt.train_atlasnet)
    opt.resnet_layers = my_utils.int_2_boolean(opt.resnet_layers)
    opt.display = my_utils.int_2_boolean(opt.display)
    opt.anisotropic_scaling = my_utils.int_2_boolean(opt.anisotropic_scaling)
    opt.reload = my_utils.int_2_boolean(opt.reload)
    opt.randomize = my_utils.int_2_boolean(opt.randomize)

    # if opt.knn == 0:
    #     opt.nepoch = 3 * opt.nepoch
    #     opt.lr_decay_1 = 3 * opt.lr_decay_1
    #     opt.lr_decay_2 = 3 * opt.lr_decay_2

    # Get output path for the training run.
    if opt.logdir is None:
        logdir = datetime.date.today().strftime("%Y-%m-%d")
        opt.logdir = jbfs.unique_dir_name(logdir)
    if opt.ds == 'shapenet':
        opt.trrun_dir = Args2String(opt)
    elif opt.ds == 'dfaust':
        margs = ''
        if opt.dfaust_mode == 'neighbors':
            margs = f"-mf{opt.dfaust_max_frames}"
        dfm = (f"{opt.dfaust_mode}{margs}", f"knn{opt.num_neighbors}")[opt.knn]
        opt.trrun_dir = f"{opt.dfaust_subject}_{opt.dfaust_sequence}_mode-{dfm}"

    save_path = jbfs.jn(opt.logdir_base, opt.logdir, opt.trrun_dir)
    if os.path.exists(save_path):
        save_path_new = jbfs.unique_dir_name(save_path)
        print(f"[WANRING] Ouptut trrun path {save_path} already exists, saving "
              f"the trianing run into new path {save_path_new}.")
        save_path = save_path_new
    opt.save_path = save_path
    jbfs.make_dir(opt.save_path)

    # if opt.logdir is None:
    #     opt.save_path = Args2String(opt)
    # else:
    #     opt.save_path = opt.logdir

    opt.env = opt.env + opt.save_path
    # opt.save_path = os.path.join("./log", opt.save_path)

    if opt.ds == 'shapenet':
        if opt.cat is None:
            opt.categories = [
                "Airplane", "Bag", "Cap", "Car", "Chair", "Earphone", "Guitar",
                "Knife", "Lamp", "Laptop", "Motorbike", "Mug", "Pistol",
                "Rocket", "Skateboard", "Table", ]
        else :
            opt.categories = [opt.cat]

    opt.date = str(datetime.datetime.now())

    # my_utils.print_arg(opt)

    return opt
