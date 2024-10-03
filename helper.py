from model.ffnn import NeuralNetwork, compute_loss
import numpy as np
import matplotlib.pyplot as plt


def batch_train(x, y, model, train_flag=False, epochs=1000, learning_rate=0.005):
    # Prediction without Training
    predictions_before_training = model.predict(x)
    accuracy_before_training = compute_accuracy(predictions_before_training, y)
    print(f"Accuracy before training: {accuracy_before_training}")

    if train_flag:
        # Training step
        costs = []
        num_samples = x.shape[1]

        for epoch in range(epochs):
            # We do the Forward pass here
            predictions = model.forward(x)
            # We do the Backward pass here
            gradients = model.backward(x, y)
            # Accumulating gradients
            grad_weights_hidden, grad_bias_hidden, grad_weights_output, grad_bias_output = gradients
            # Updating weights and biases using accumulated gradients
            model.weights_input_hidden -= learning_rate * grad_weights_hidden / num_samples
            model.bias_hidden -= learning_rate * grad_bias_hidden / num_samples
            model.weights_hidden_output -= learning_rate * grad_weights_output / num_samples
            model.bias_output -= learning_rate * grad_bias_output / num_samples

            # We Compute the average loss for the epoch
            average_loss = compute_loss(predictions, y)
            costs.append(average_loss)

            pred_to_acc = model.predict(x)
            acc = compute_accuracy(pred_to_acc, y)
            print(f"Accuracy in epoch {epoch}: {acc}, loss: {average_loss}")

        # Prediction after Training
        predictions_after_train = model.predict(x)
        accuracy_after_training = compute_accuracy(predictions_after_train, y)
        print(f"Accuracy after training: {accuracy_after_training}")

        # Plotting the Cost Function here
        plot_cost_function(costs, title='Cost Function During Training')


def minibatch_train(x, y, model, train_flag=True, epochs=1000, learning_rate=0.005, batch_size=64):
    predictions_before_training = model.predict(x)
    accuracy_before_training = compute_accuracy(predictions_before_training, y)
    print(f"Accuracy before training: {accuracy_before_training}")

    if train_flag:
        # Training
        costs = []
        num_samples = x.shape[1]

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            x_shuffled = x[:, indices]
            y_shuffled = y[:, indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[:, i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]

                # We do the Forward pass here
                predictions = model.forward(x_batch)
                # We do the Backward pass here
                gradients = model.backward(x_batch, y_batch)
                # Accumulating gradients
                grad_weights_hidden, grad_bias_hidden, grad_weights_output, grad_bias_output = gradients
                # Updating weights and biases using accumulated gradients
                model.weights_input_hidden -= learning_rate * grad_weights_hidden / batch_size
                model.bias_hidden -= learning_rate * grad_bias_hidden / batch_size
                model.weights_hidden_output -= learning_rate * grad_weights_output / batch_size
                model.bias_output -= learning_rate * grad_bias_output / batch_size

            # We Compute the average loss for the epoch
            preds_after_training = model.predict(x)
            accuracy_after_training = compute_accuracy(preds_after_training, y)
            average_loss = compute_loss(preds_after_training, y)
            costs.append(average_loss)
            print(f"Accuracy in epoch {epoch}: {accuracy_after_training}, loss: {average_loss}")

        print(f"Accuracy after training: {accuracy_after_training}")

        # Plotting the Cost Function here
        plot_cost_function(costs, title='Cost Function During Mini-batch Training')


def stochastic_train(X, Y, model, train_flag=True, epochs=1000, learning_rate=0.005):
    predictions_before_training = model.predict(X)
    accuracy_before_training = compute_accuracy(predictions_before_training, Y)
    print(f"Accuracy before training: {accuracy_before_training}")

    if train_flag:
        # Training
        costs = []
        num_samples = X.shape[1]

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]

            # Stochastic training
            for i in range(num_samples):
                X_sample = X_shuffled[:, i:i+1]
                Y_sample = Y_shuffled[:, i:i+1]

                # We do the Forward pass here
                predictions = model.forward(X_sample)
                # We do the Backward pass here
                gradients = model.backward(X_sample, Y_sample)
                # Accumulating gradients
                grad_weights_hidden, grad_bias_hidden, grad_weights_output, grad_bias_output = gradients
                # Updating weights and biases using accumulated gradients
                model.weights_input_hidden -= learning_rate * grad_weights_hidden
                model.bias_hidden -= learning_rate * grad_bias_hidden
                model.weights_hidden_output -= learning_rate * grad_weights_output
                model.bias_output -= learning_rate * grad_bias_output

            # We Compute the average loss for the epoch
            predictions_after_training = model.predict(X)
            accuracy_after_training = compute_accuracy(predictions_after_training, Y)
            average_loss = compute_loss(predictions_after_training, Y)
            costs.append(average_loss)
            print(f"Accuracy in epoch {epoch}: {accuracy_after_training}, loss: {average_loss}")

        print(f"Accuracy after training: {accuracy_after_training}")

        # Plotting the Cost Function
        plot_cost_function(costs, title='Cost Function During Stochastic Gradient Descent Training')


def compute_accuracy(predictions, ground_truth):
    # Your accuracy calculation logic goes here
    # Example: Calculate accuracy as the percentage of correct prediction
    correct = [1 for index in range(ground_truth.shape[1])
               if np.all(predictions[:, index] == ground_truth[:, index], axis=0)]

    number = sum(correct) / ground_truth.shape[1]
    return number


def plot_cost_function(costs, title='Cost Function During Training'):
    plt.plot(costs)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()
