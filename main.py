import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datasets.utils import *
from datasets.DiveFCDataset import DiveFCDataset
from models import FCModel, GATModel, GCNModel
from sklearn import metrics
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = ArgumentParser(description='graph-dive')
    parser.add_argument('--conf_id', type=str, required=True,
                        help="directory of journal/conference that save json files")
    parser.add_argument('--affil_embed_file', type=str, default="../data/affiliationembedding.csv",
                        help="path of CSV File path that represents authors' affiliation.")
    parser.add_argument('--citation_threshold', type=int, default=20,
                        help="criterion that decides whether a paper falls above or below top 10%")
    parser.add_argument('--val_interval', type=int, default=10,
                        help="run validation per arguments' epoch if exists")

    args = parser.parse_args()
    return args


def main():    
    args = parse_args()
    conf_path = "../data/json_" + args.conf_id
    edge_data_path = '../data/edge_data/' + args.conf_id + '_refs.csv'
    year_data_path = '../data/year_data/' + args.conf_id + '.csv'

    # load raw_inputs from data tables
    divefc_set = DiveFCDataset(conf_path, args.affil_embed_file, edge_data_path, args.citation_threshold)
    divefc_loader = DataLoader(divefc_set, batch_size=divefc_set.batch_size, shuffle=True)

    # instantiate models
    fcnet = FCModel(text_dim=1000, affiliation_dim=3789, year_dim=13)
    gat = GATModel(num_layers=3, input_dim=202, hidden_dim=30, output_dim=30, heads=6)
    gcn = GCNModel(num_layers=2, input_dim=30, hidden_dim=30, output_dim=1, dropout=0.3, training=True)

    fcnet = fcnet.to(device)
    gat = gat.to(device)
    gcn = gcn.to(device)

    # instantiate objective function and optimizer
    params = [*fcnet.parameters(), *gat.parameters(), *gcn.parameters()]
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    criterion = nn.BCELoss()

    # training
    epochs = 150
    loss_history = []

    for epoch in range(epochs):

        fcnet.train()
        gat.train()
        gcn.train()
        optimizer.zero_grad()
        # train FC layers with raw inputs
        for paper_ids, inputs, labels in divefc_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = fcnet(inputs) # # [batchsize, 202]
        
        graph_data = construct_graph_data(paper_ids, embeddings, labels, edge_data_path, year_data_path, epoch)
        graph_loader = torch_geometric.loader.DataLoader([graph_data], batch_size=len(labels), shuffle=False)
        
        # train GAT and GCN
        for idx, train_batch in enumerate(graph_loader):
            train_batch = train_batch.to(device)
            gat_embeddings = gat(train_batch) # [batchsize, 30]
            
            pred = gcn(gat_embeddings, train_batch)
            loss = criterion(pred[graph_data.train_idx].squeeze(1), train_batch.y[graph_data.train_idx])
            
            loss.backward()
            optimizer.step()

        scheduler.step()
        print("[Epoch {}/{}] Train Loss: {:.6f}".format(epoch, epochs, loss.item()))
        loss_history.append(loss.item())

        if args.val_interval and epoch % args.val_interval==0:
            fcnet.eval()
            gat.eval()
            gcn.eval()

            with torch.no_grad():

                for paper_ids, inputs, labels in divefc_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    embeddings = fcnet(inputs)  # # [batchsize, 202]

                graph_data = construct_graph_data(paper_ids, embeddings, labels, edge_data_path, year_data_path, epoch)
                graph_loader = torch_geometric.loader.DataLoader([graph_data], batch_size=len(labels), shuffle=False)

                # train GAT and GCN
                for idx, train_batch in enumerate(graph_loader):
                    train_batch = train_batch.to(device)
                    gat_embeddings = gat(train_batch)  # [batchsize, 30]

                    pred = gcn(gat_embeddings, train_batch)
                    pred = (pred>0.5).long()
                    f1_score = metrics.f1_score(y_true = train_batch.y[graph_data.val_idx],
                                                y_pred = pred[graph_data.val_idx].squeeze(1))
                    accuracy = metrics.accuracy_score(y_true = train_batch.y[graph_data.val_idx],
                                                      y_pred = pred[graph_data.val_idx].squeeze(1))

                print("[Epoch {}/{}] Validation F1 Score: {:.6f}".format(epoch, epochs, f1_score))
                print("[Epoch {}/{}] Validation Accuracy: {:.6f}".format(epoch, epochs, accuracy))
            



    # plot training loss curve
    plt.plot([i for i in range(epochs)], loss_history)
    plt.title('Loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./save/loss_curve_{}.jpg'.format(args.conf_id))

    # test
    # ...


if __name__ == "__main__":
    main()