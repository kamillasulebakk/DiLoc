from model_runner import run_model


def basic_parameters():
    return {
        'N_samples': 70_000,
        'N_dipoles': 1,
        'determine_amplitude': False,
        'determine_area': False,
        'hidden_layers': [512, 256, 128, 64, 32],
        'learning_rate': 0.001,
        'momentum': 0.9,
        'N_epochs': 500,
        'noise_pct': 10
    }


def main():
    batch_sizes = [32]
    weight_decay = [0.5, 0.9]
    l1_lambdas = [0]

    parameters = basic_parameters()

    for size in batch_sizes:
        parameters['batch_size'] = size
        for weight in weight_decay:
            parameters['weight_decay'] = weight
            for l1_lambda in l1_lambdas:
                parameters['l1_lambda'] = l1_lambda
                run_model(parameters)


if __name__ == '__main__':
    main()
