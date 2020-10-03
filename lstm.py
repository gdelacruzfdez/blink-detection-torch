import sys
import dataloader
import network
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from functional import seq
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from evaluation import evaluate, evaluatePartialBlinks


class LSTMModel:

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((100, 100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((100, 100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    LOG_FILE_HEADER_BLINK_DETECTION = 'EPOCH,TRAIN_LOSS,TRAIN_ACCURACY,TRAIN_PRECISION,TRAIN_RECALL,TRAIN_F1,EVAL_F1,EVAL_PRECISION,EVAL_RECALL,EVAL_TP,EVAL_FP,EVAL_TN,EVAL_FN'
    LOG_FILE_HEADER_BLINK_COMPLETENESS = 'EPOCH,TRAIN_LOSS,TRAIN_ACCURACY,TRAIN_PRECISION,TRAIN_RECALL,TRAIN_F1,PARTIAL_F1,PARTIAL_PRECISION,PARTIAL_RECALL,PARTIAL_TP,PARTIAL_FP,PARTIAL_TN,PARTIAL_FN,COMPLETE_F1,COMPLETE_PRECISION,COMPLETE_RECALL,COMPLETE_TP,COMPLETE_FP,COMPLETE_TN,COMPLETE_FN,'

    def __init__(self, params, cuda):
        self.params = params
        self.batch_size = params.get('batch_size')
        self.dims = params.get('dims')
        self.cnn_model_file = params.get('cnn_model_file')
        self.lstm_model_file = params.get('lstm_model_file')
        self.lstm_hidden_units = params.get('lstm_hidden_units')
        self.sequence_len = params.get('sequence_len')
        self.epochs = params.get('epochs')
        self.lstm_layers = params.get('lstm_layers')
        self.eval_mode = params.get('eval_mode')
        self.lr = params.get('lr')

        self.cuda = cuda
        if 'train_dataset_dirs' in params:
            self.train_dataset_dirs = seq(params.get('train_dataset_dirs'))\
                .map(lambda x: "{}/{}".format(params.get('datasets_base_path'), x))
        if 'test_dataset_dirs' in params:
            self.test_dataset_dirs = seq(params.get('test_dataset_dirs'))\
                .map(lambda x: "{}/{}".format(params.get('datasets_base_path'), x))

        self.best_f1 = -1
        self.best_epoch = -1
        self.current_f1 = -1

        self.__initialize_cnn_model()
        self.__initialize_lstm_model()
        self.__initialize_log_file()    
        self.__initialize_train_loader()
        self.__initialize_evaluation_loader()
        self.__initialize_training_parameters()

    def __initialize_log_file(self):
        separator = '-'
        self.base_file_name = "LSTM_train[{}]_test[{}]_batch_[{}]_dims[{}]_lr[{}]_hidden-units[{}]_lstm-layers[{}]_sequence-len[{}]_epochs[{}]"\
            .format(separator.join(self.params.get('train_dataset_dirs')),
                    separator.join(self.params.get('test_dataset_dirs')),
                    self.batch_size,
                    self.dims,
                    self.lr,
                    self.lstm_hidden_units,
                    self.lstm_layers,
                    self.sequence_len,
                    self.epochs)

        self.log_file = open(self.base_file_name + '.log', 'w')

        self.log_file.write(self.base_file_name + '\n\n')
        if dataloader.BLINK_DETECTION_MODE == self.eval_mode:
            self.log_file.write(self.LOG_FILE_HEADER_BLINK_DETECTION)
        elif dataloader.BLINK_COMPLETENESS_MODE == self.eval_mode:
            self.log_file.write(self.LOG_FILE_HEADER_BLINK_COMPLETENESS)


    def __initialize_train_loader(self):
        self.train_set = dataloader.LSTMDataset(
            self.train_dataset_dirs, self.TRAIN_TRANSFORM, mode=self.eval_mode)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def __initialize_evaluation_loader(self):
        self.test_set = dataloader.LSTMDataset(
            self.test_dataset_dirs, self.TEST_TRANSFORM, mode=self.eval_mode)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def __initialize_cnn_model(self):
        self.cnn_model = network.SiameseNetV2(self.dims)
        self.cnn_model.load_state_dict(torch.load(self.cnn_model_file))
        self.cnn_model.eval()
        if self.cuda:
            self.cnn_model = self.cnn_model.cuda()

    def __initialize_lstm_model(self):
        self.num_classes = 3 if self.eval_mode == dataloader.BLINK_COMPLETENESS_MODE else 2
        self.lstm_model = network.BiRNN(
            self.dims, self.lstm_hidden_units, self.lstm_layers, self.num_classes)
        if self.cuda:
            self.lstm_model = self.lstm_model.cuda()

    def __initialize_training_parameters(self):
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.lstm_model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'max', patience=8, factor=0.1, verbose=True)

    def __perf_measure(self, y_actual, y_hat):
        TP = FP = TN = FN = 0
        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    def __train_epoch(self):
        self.lstm_model.train()
        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.train_loader), total=len(
            self.train_loader), desc='Training', file=sys.stdout)
        for batch_idx, data in progress:
            samples, targets = data
            if self.cuda:
                samples = samples.cuda()
                targets = targets.cuda()

            features = self.cnn_model.get_embedding(samples)

            if features.numel() < self.dims * self.batch_size:
                continue
            features = features.reshape(-1, self.sequence_len, self.dims)

            outputs = self.lstm_model(features)

            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            targets = targets.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu()
            perf = self.__perf_measure(targets, predicted)
            TN += perf[2]
            FN += perf[3]
            TP += perf[0]
            FP += perf[1]

            acc = (TP + TN) / (FP + FN + TP + TN)
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if precision + recall > 0 else 0
            accuracies.append(acc)
            progress.set_description('Training Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        return np.mean(losses), np.mean(accuracies), precision, recall, f1

    def __test_epoch(self, evaluation=False):
        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.test_loader), total=len(
            self.test_loader), desc='Testing', file=sys.stdout)
        predictions = np.array([])
        all_targets = np.array([])

        for batch_idx, data in progress:
            samples, targets = data
            if self.cuda:
                samples = samples.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                features = self.cnn_model.get_embedding(samples)

                if features.numel() < self.dims * self.batch_size:
                    zeros_features = torch.zeros(
                        self.batch_size - features.shape[0], self.dims)
                    if self.cuda:
                        zeros_features = zeros_features.cuda()
                    features = torch.cat((features, zeros_features))
                    zeros_targets = torch.zeros(
                        self.batch_size - targets.shape[0], dtype=torch.long)
                    if self.cuda:
                        zeros_targets = zeros_targets.cuda()
                    targets = torch.cat((targets, zeros_targets))

                features = features.reshape(-1, self.sequence_len, self.dims)
                outputs = self.lstm_model(features)

                loss = self.criterion(outputs, targets)
                losses.append(loss.item())

                targets = targets.data.cpu()
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.data.cpu()
                all_targets = np.concatenate((all_targets, targets))
                predictions = np.concatenate((predictions, predicted))

                perf = self.__perf_measure(targets, predicted)
                TN += perf[2]
                FN += perf[3]
                TP += perf[0]
                FP += perf[1]

                acc = (TP + TN) / (FP + FN + TP + TN)
                accuracies.append(acc)
                precision = TP / (TP + FP) if TP + FP > 0 else 0
                recall = TP / (TP + FN) if TP + FN > 0 else 0
                f1 = 2 * (precision * recall) / (precision +
                                                 recall) if precision + recall > 0 else 0
                progress.set_description('Test Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                    loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        dataframe = self.test_loader.dataset.getDataframe().copy()
        predictions = predictions[:len(dataframe)]
        all_targets = all_targets[:len(dataframe)]
        dataframe['pred'] = predictions
        dataframe['targets'] = all_targets
        if evaluation:
            dataframe.to_csv('-'.join(self.params.get('test_dataset_dirs')) + '-' + self.eval_mode + '.csv', index=False)
        if dataloader.BLINK_COMPLETENESS_MODE == self.eval_mode:
            return evaluatePartialBlinks(dataframe)
        elif dataloader.BLINK_DETECTION_MODE == self.eval_mode:
            return evaluate(dataframe)

    def eval(self):
        self.lstm_model.load_state_dict(torch.load(self.lstm_model_file))
        self.lstm_model.eval()
        if self.cuda:
            self.lstm_model = self.lstm_model.cuda()
        results = self.__test_epoch(True)
        if dataloader.BLINK_COMPLETENESS_MODE == self.eval_mode:
            results_partial, results_complete = results
            print('Eval results partial => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                results_partial['f1'], results_partial['precision'], results_partial['recall'], results_partial['tp'], results_partial['fp'], results_partial['fn']))
            print('Eval results complete => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                results_complete['f1'], results_complete['precision'], results_complete['recall'], results_complete['tp'], results_complete['fp'], results_complete['fn']))
        elif dataloader.BLINK_DETECTION_MODE == self.eval_mode:
            print('Eval results => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                results['f1'], results['precision'], results['recall'], results['tp'], results['fp'], results['fn']))

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(self.current_f1)
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = self.__train_epoch()
            print('Epoch: {}/{}, Average train loss: {:.4f}, Average train accuracy: {:.4f}'.format(
                epoch, self.epochs, train_loss, train_accuracy))

            results = self.__test_epoch()
            if dataloader.BLINK_COMPLETENESS_MODE == self.eval_mode:
                results_partial, results_complete = results
                self.current_f1 = results_partial['f1'] + results_complete['f1']
                print('Epoch: {}/{}, Partial => F1:  {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                    epoch, self.epochs, results_partial['f1'], results_partial['precision'], results_partial['recall'], results_partial['tp'], results_partial['fp'], results_partial['fn']))
                print('Epoch: {}/{}, Complete => F1: {:.4f} | Precision: {:.4f} | {:.4f} | TP: {} | FP:  {}'.format(
                    epoch, self.epochs, results_complete['f1'], results_complete['precision'], results_complete['recall'], results_complete['tp'], results_complete['fp'], results_complete['fn']))
                self.log_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
                                    .format(epoch,
                                            train_loss,
                                            train_accuracy,
                                            train_precision,
                                            train_recall,
                                            train_f1,
                                            results_partial['f1'],
                                            results_partial['precision'],
                                            results_partial['recall'],
                                            results_partial['tp'],
                                            results_partial['fp'],
                                            results_partial['db'],
                                            results_partial['fn'],
                                            results_complete['f1'],
                                            results_complete['precision'],
                                            results_complete['recall'],
                                            results_complete['tp'],
                                            results_complete['fp'],
                                            results_complete['db'],
                                            results_complete['fn']
                                            ))
            elif dataloader.BLINK_DETECTION_MODE == self.eval_mode:
                self.current_f1 = results['f1']
                print('Epoch: {}/{}, F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                    epoch, self.epochs, results['f1'], results['precision'], results['recall'], results['tp'], results['fp'], results['fn']))
                self.log_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
                                    .format(epoch,
                                            train_loss,
                                            train_accuracy,
                                            train_precision,
                                            train_recall,
                                            train_f1,
                                            results['f1'],
                                            results['precision'],
                                            results['recall'],
                                            results['tp'],
                                            results['fp'],
                                            results['db'],
                                            results['fn']
                                            ))


            if self.current_f1 > self.best_f1:
                self.best_f1 = self.current_f1
                self.best_epoch = epoch
                torch.save(self.lstm_model.state_dict(),
                           self.base_file_name + '.pt')

        self.log_file.close()
