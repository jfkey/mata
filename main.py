from src.utils import tab_printer
from src.parser import parameter_parser
# from src.our_model import OurNNTrainer as Trainer
from src.mata_aids import OurNNTrainer as Trainer
from src.mata_cancer import OurNNTrainer as TrainerPair
import torch


def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a genn model.
    """
    args = parameter_parser()
    tab_printer(args)
    if args.dataset in ['AIDS700nef', 'LINUX', 'ALKANE', 'IMDBMulti']:
        trainer = Trainer(args)  # 在GENN Trainer初始化的时候，预处理数据，并初始化GENN模型
    elif args.dataset in ['CANCER']:
        trainer =TrainerPair(args)

    if args.cuda:
        trainer.model = trainer.model.cuda()

    if args.test or args.val:
        weight_path = 'best_model_{}_{}_e{}_lr{}_loss{}_t{}_stru{}_b{}.pt'
        trainer.model.load_state_dict(torch.load(weight_path.format(args.dataset, args.gnn_operator, args.epochs, args.learning_rate, args.loss_type, args.tasks, args.nonstruc, args.batch_size)))
        trainer.model.eval()
        trainer.score(test=args.test)
        exit(0)
    else:  # training
        trainer.fit()
        weight_path = 'best_model_{}_{}_e{}_lr{}_loss{}_t{}_stru{}_b{}.pt'
        trainer.model.load_state_dict(torch.load(weight_path.format(args.dataset, args.gnn_operator, args.epochs, args.learning_rate, args.loss_type, args.tasks, args.nonstruc, args.batch_size)))
        trainer.model.eval()  # 不启用 Batch Normalization 和 Dropout。
        trainer.score()

if __name__ == "__main__":
    main()
