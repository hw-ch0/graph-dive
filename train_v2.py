import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datasets.utils import *
from datasets.DiveFCDataset import DiveFCDataset
from datasets.construct_data import construct_data
from models import GraphDiveModel
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
    parser.add_argument('--val_interval', type=int, default=1,
                        help="run validation per arguments' epoch if exists")

    args = parser.parse_args()
    return args


def main():    
    args = parse_args()
    conf_path = "../data/json_" + args.conf_id
    edge_data_path = '../data/edge_data/' + args.conf_id + '_refs.csv'
    year_data_path = '../data/year_data/' + args.conf_id + '.csv'

    # load raw_inputs from data tables
    graph_data = construct_data(dir_path=conf_path,
                                affiliation_path=args.affil_embed_file,
                                edge_data_path=edge_data_path,
                                year_data_path=year_data_path,
                                citation_threshold=20,
                                epoch=0)
    graph_loader = torch_geometric.loader.DataLoader([graph_data], batch_size=len(graph_data.y), shuffle=False)

    model = GraphDiveModel(text_dim=1000, affiliation_dim=3789, embedding_dim=128, year_dim=13, dropout=0.3, hidden_dim=128)
    model.to(device)

    # instantiate objective function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    criterion = nn.BCELoss()

    # training
    epochs = 300
    loss_history = []
    f1_history = []
    accuracy_history = []

    for epoch in range(epochs):

        train_loss = 0
        model.train()
        optimizer.zero_grad()
        # train FC layers with raw inputs

        model.train()
        optimizer.zero_grad()
        # train FC layers with raw inputs
        for idx, train_batch in enumerate(graph_loader):
            train_batch = train_batch.to(device)
            pred = model(train_batch)
            loss = criterion(pred[train_batch.train_idx].squeeze(1), train_batch.y[train_batch.train_idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        loss_history.append(loss.item())
        print("[Epoch {}/{}] Train Loss: {:.6f}".format(epoch, epochs, loss.item()))

        if epoch % args.val_interval == 0:

            model.eval()

            with torch.no_grad():

                for idx, train_batch in enumerate(graph_loader):
                    train_batch = train_batch.to(device)
                    pred = model(train_batch)
                    # print('preds : {}'.format(pred[train_batch.val_idx][:10]))
                    pred = (pred > 0.5).long().squeeze(1)
                    # print('labels : {}'.format(train_batch.y[train_batch.val_idx][:10]))
                    # print('preds : {}'.format(pred[train_batch.val_idx][:10]))
                    f1_score = metrics.f1_score(y_true=train_batch.y[train_batch.val_idx],
                                                y_pred=pred[train_batch.val_idx])
                    accuracy = metrics.accuracy_score(y_true=train_batch.y[train_batch.val_idx],
                                                      y_pred=pred[train_batch.val_idx])
            f1_history.append(f1_score)
            accuracy_history.append(accuracy)
            print("[Epoch {}/{}] Validation F1 Score: {:.6f}".format(epoch, epochs, f1_score))
            print("[Epoch {}/{}] Validation Accuracy: {:.6f}".format(epoch, epochs, accuracy))

    # plot training loss curve
    plt.plot([i for i in range(epochs)], loss_history, label='CE loss')
    plt.title('Loss curve_{}'.format(args.conf_id))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./save/loss_curve_{}.jpg'.format(args.conf_id))

    plt.clf()
    plt.plot([i for i in range(epochs)], f1_history, label='f1 score')
    plt.plot([i for i in range(epochs)], accuracy_history, label='Accuracy')
    plt.title('Validation Score_{}'.format(args.conf_id))
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('./save/val_score_{}.jpg'.format(args.conf_id))



if __name__ == "__main__":
    main()