from imports import *
from tab_transformer_pytorch import FTTransformer

# Initialize ML models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_jobs=-1),
    'CatBoost': cb.CatBoostClassifier(silent=True),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=10000, n_jobs=-1),
    'SVC': SVC(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'MLP': MLPClassifier(max_iter=10000, hidden_layer_sizes=(128, 128, 128), early_stopping=True),
}

# Choose ML best models
best_models = [RandomForestClassifier(), xgb.XGBClassifier(), cb.CatBoostClassifier(silent=True)]

# Create DL models
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc5(out)
        return out