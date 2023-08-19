import torch
import torch.nn as nn


class WFEncoder(nn.Module):
    def __init__(self, encoding_size, classify=False, n_classes=None):
        # Input x is (batch, 2, 256)
        super(WFEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.n_classes = n_classes
        self.classify = classify
        self.classifier =None
        if self.classify:
            if self.n_classes is None:
                raise ValueError('Need to specify the number of output classes for te encoder')
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.encoding_size, self.n_classes)
                )
                nn.init.xavier_uniform_(self.classifier[1].weight)

        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            # nn.Dropout(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            # nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(79872, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Linear(2048, self.encoding_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        encoding = self.fc(x)
        if self.classify:
            c = self.classifier(encoding)
            return c
        else:
            return encoding
