import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run Our model.")
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Run the code with cuda.")
    parser.add_argument("--test", dest="test", action="store_true", help="Run testing script.")
    parser.add_argument("--val", dest="val", action="store_true", help="Run validation script.")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Run validation script.")
    parser.add_argument("--dataset", nargs="?", default="AIDS700nef", help="Dataset name. Default is AIDS")#AIDS700nef, LINUX, ALKANE, IMDBMulti
    parser.add_argument("--gnn-operator", nargs = "?", default = "gcn", help = "Type of GNN-Operator. Default is gcn")
    parser.add_argument("--epochs",  type = int,  default = 10000, help = "Number of training epochs. Default is 10000.")
    parser.add_argument("--val-epochs", type=int, default=5000, help="Number of validation epochs. Default is 5000.")
    parser.add_argument("--filter-1",  type = int, default = 64, help = "Filter (neurons) in 1st convolution. Default is 64.")
    parser.add_argument("--filter-2",  type = int, default = 64, help = "Filter (neurons) in 2nd convolution. Default is 64.")
    parser.add_argument("--filter-3", type = int, default = 64, help = "Filter (neurons) in 3rd convolution. Default is 64.")
    parser.add_argument("--tensor-neurons", type = int, default = 16, help = "Neurons in tensor network layer. Default is 16.")
    parser.add_argument("--bottle-neck-neurons", type = int, default = 16, help = "Bottle neck layer neurons. Default is 16.")
    parser.add_argument("--batch-size",  type = int,  default = 128, help = "Number of graph pairs per batch. Default is 128.")
    parser.add_argument("--topk", type=int, default=5, help="The most similar k nodes of G2. Default is 5.")
    parser.add_argument("--loss-type", type=int, default=2, help="Test of loss type")
    parser.add_argument("--dropout", type = float, default = 0, help = "Dropout probability. Default is 0.")
    parser.add_argument("--learning-rate", type = float, default = 0.001, help = "Learning rate. Default is 0.001.")
    parser.add_argument("--weight-decay", type = float, default = 5*10**-4, help = "Adam weight decay. Default is 5*10^-4.")
    parser.add_argument("--random-walk-step", type=int, default=16, help="The steps of random walk. Default is 16")
    parser.add_argument("--max-degree", type=int, default=12, help="The number of max degree. Default is 12")
    parser.add_argument("--tasks", type=int, default=3, help="The auxiliary task. Default is 3")
    parser.add_argument("--sinkhorn", dest="sinkhorn", action="store_true")
    parser.add_argument("--nonstruc", dest="nonstruc", action="store_true")
    parser.add_argument("--beam", dest="beam", action="store_true")

    parser.set_defaults(sinkhorn=True)
    parser.set_defaults(beam=False)

    return parser.parse_args()
