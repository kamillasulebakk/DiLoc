from model_runner import run_model


def main():
    batch_sizes = [32, 64]
    weight_decay = [0.1, 0.5]
    l1_lambda = [0, 0.000001]

    parameters = {
        'N_samples': 70_000,
        'N_dipoles': 1,
        'determine_amplitude': False,
        'determine_area': False,
        'hidden_layers': [512, 256, 128, 64, 32],
        'batch_size': 32,
        'learning_rate': 0.001,
        'momentum': 0.35,
        'l1_lambda': 0.0,
        'weight_decay': 0.1,
        'N_epochs': 500,
        'noise_pct': 10
    }

    for size in batch_sizes:
        parameters['batch_size'] = size
        for weight in weight_decay:
            parameters['weight_decay'] = weight
            run_model(parameters)


if __name__ == '__main__':
    main()
