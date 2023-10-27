from imports import *
from params import *
from models import *


def preprocess_labels(labels):
    labels = labels.map(class_dict)
    return labels.to_numpy()


def data_load(root_dir):
    # Load data
    df = pd.read_csv(root_dir)

    # Split the data into three datasets
    datasets = []
    for i in range(3):
        subset = df[df['class'] == 'GALAXY'].sample(frac=1 / 3)
        subset = pd.concat([subset, df[df['class'] == 'QSO'], df[df['class'] == 'STAR']])
        datasets.append(subset)

    # Preprocess features and labels for each dataset
    features = []
    labels = []
    scaler = StandardScaler()
    for dataset in datasets:
        labels.append(preprocess_labels(dataset['class']))
        subset_features = dataset[['u', 'g', 'i', 'z', 'redshift']].to_numpy()
        subset_features = scaler.fit_transform(subset_features)
        features.append(subset_features)

    # Assign variables for each dataset
    features1, features2, features3 = features
    labels1, labels2, labels3 = labels

    datasets_divided = [(features1, labels1), (features2, labels2), (features3, labels3)]
    return datasets_divided


def models_training():
    # Define your datasets and labels
    datasets = data_load(root_dir=data_path)

    # Initialize dictionaries to store mean accuracy and F1 scores for each model
    mean_train_scores = {model_name: 0 for model_name in models}
    mean_val_scores = {model_name: 0 for model_name in models}
    mean_test_scores = {model_name: 0 for model_name in models}
    mean_f1_scores = {model_name: 0 for model_name in models}

    for dataset_num, dataset in enumerate(datasets, start=1):
        for model_name, model in models.items():
            X_train, X_val, Y_train, Y_val = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=15)
            X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5, random_state=15)
            print(f"Training {model_name} on Dataset {dataset_num}...")

            # Train the model on the current dataset
            model.fit(X_train, Y_train)
            Y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(Y_train, Y_train_pred)

            # Evaluate the model on the test set of the current dataset
            acc_scores = cross_val_score(model, X_val, Y_val, cv=5, scoring='accuracy')
            f1_scores = cross_val_score(model, X_val, Y_val, cv=5, scoring='f1_weighted')

            # Get the test accuracy of the current dataset
            Y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(Y_test, Y_test_pred)

            # Calculate and update mean accuracy and F1 scores
            mean_train_scores[model_name] += train_accuracy
            mean_val_scores[model_name] += acc_scores.mean()
            mean_test_scores[model_name] += test_accuracy
            mean_f1_scores[model_name] += f1_scores.mean()
        print("-" * 20)
        print("-" * 20)

    # Calculate the overall mean accuracy and F1 scores by dividing by the number of datasets
    num_datasets = len(datasets)
    for model_name in models:
        mean_train_scores[model_name] /= num_datasets
        mean_val_scores[model_name] /= num_datasets
        mean_test_scores[model_name] /= num_datasets
        mean_f1_scores[model_name] /= num_datasets

    # Print the mean accuracy and F1 scores for each model
    print("-" * 20)
    for model_name in models:
        print(
            f"{model_name} - "
            f"Mean Train Accuracy: {mean_train_scores[model_name]:.4f}, "
            f"Mean Val Accuracy: {mean_val_scores[model_name]:.4f}, "
            f"Mean Test Accuracy: {mean_test_scores[model_name]:.4f}, "
            f"Mean Validation F1 Score: {mean_f1_scores[model_name]:.4f}")
    print("-" * 20)

    # Save the scores to a json file
    with open(f'{logs_path}/models_scores.json', 'w') as f:
        json.dump({'Train Accuracy': mean_train_scores,
                   'Validation Accuracy': mean_val_scores,
                   'Test Accuracy': mean_test_scores,
                   'Validation F1 Score': mean_f1_scores}, f, indent=4)


def best_models_training():
    # Define the hyperparameter grid
    param_grid = [rf_params, xgb_params, cb_params]

    # Define the grid search object
    grid_searches = [GridSearchCV(model,
                                  param_grid=param_grid[i],
                                  cv=2,
                                  scoring='accuracy',
                                  n_jobs=30,
                                  verbose=0)
                     for i, model in enumerate(best_models)]

    # Load the data
    datasets = data_load(root_dir=data_path)
    features1, labels1 = datasets[0]

    # Fit the grid search object to the data
    for grid_search in grid_searches:
        print(f"Running grid search for: {grid_search.estimator.__class__.__name__}")
        grid_search.fit(features1, labels1)
        # save results of the model to the disk using json
        with open(f'{grid_search.estimator.__class__.__name__}_results.json', 'w') as f:
            # Convert ndarray to list before serializing to JSON
            cv_results_serializable = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in grid_search.cv_results_.items()
            }
            json.dump(cv_results_serializable, f, indent=4)

    # Print the best hyperparameters and results for each model
    for i, grid_search in enumerate(grid_searches):
        print(f"Model: {best_models[i].__class__.__name__}")
        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Best accuracy score: {grid_search.best_score_:.4f}")
        print("-" * 20)

    # Save the best models and their accuracies, hyperparameters to a json file
    best_models_scores = {model.__class__.__name__: grid_search.best_score_ for model, grid_search in
                          zip(best_models, grid_searches)}
    best_models_params = {}
    best_models_names = [model.__class__.__name__ for model in best_models]
    for idx, name in enumerate(best_models_names):
        best_models_params[name] = grid_searches[idx].best_params_
    with open(f'{logs_path}/best_models.json', 'w') as f:
        json.dump({'Scores': best_models_scores, 'Params': best_models_params}, f, indent=4)


def nn_training():
    datasets = data_load(root_dir=data_path)
    features1, labels1 = datasets[0]

    X_train, X_test, Y_train, Y_test = train_test_split(features1, labels1, test_size=0.1, stratify=labels1,
                                                        random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.175, stratify=Y_train,
                                                      random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Transform numpy arrays to tensors
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).long()
    X_val = torch.from_numpy(X_val).float()
    Y_val = torch.from_numpy(Y_val).long()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).long()

    # Define the hyperparameters
    input_size = X_train.shape[1]
    num_classes = np.unique(Y_train).shape[0]
    learning_rate = 0.001
    num_epochs = 1000
    correct_train = 0
    total_train = 0
    train_history = {'loss': [], 'train_accuracy': [], 'val_accuracy': []}

    # Create the neural network instance
    net = Net(input_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = net(X_train)
        loss = criterion(outputs, Y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += Y_train.size(0)
        correct_train += (predicted == Y_train).sum().item()
        avg_train_accuracy = 100 * correct_train / total_train

        # Print the loss for every epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {avg_train_accuracy:.2f}%")

        # Run test set
        with torch.no_grad():
            outputs = net(X_val)
            _, predicted = torch.max(outputs.data, 1)
            total_val = Y_val.size(0)
            correct_val = (predicted == Y_val).sum().item()
            avg_val_accuracy = 100 * correct_val / total_val
            print(f"Validation Accuracy: {avg_val_accuracy:.2f}%")

        train_history['loss'].append(loss.item())
        train_history['train_accuracy'].append(avg_train_accuracy)
        train_history['val_accuracy'].append(avg_val_accuracy)

    # final test accuracy
    with torch.no_grad():
        outputs = net(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total_test = Y_test.size(0)
        correct_test = (predicted == Y_test).sum().item()
        avg_test_accuracy = 100 * correct_test / total_test
        print(f"Test Accuracy: {avg_test_accuracy:.2f}%")

    # save model
    torch.save(net.state_dict(), f'{logs_path}/nn_model.pth')

    # save history
    with open(f'{logs_path}/nn_history.json', 'w') as f:
        json.dump(train_history, f, indent=4)


def main():
    models_training()
    # best_models_training()
    # nn_training()


if __name__ == '__main__':
    main()