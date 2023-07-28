from model_runner import run_model


def basic_parameters():
    return {
        'N_samples': 70_000,
        'N_dipoles': 1,
        'determine_amplitude': True,
        'determine_area': False,
        'hidden_layers': [512, 256, 128, 64, 32],
        'batch_size': 32,
        'learning_rate': 0.001,
        'momentum': 0.35,
        'N_epochs': 3000,
        'noise_pct': 10
    }

def main():
    batch_sizes = [32, 64]
    weight_decay = [0.1, 0.5]
    l1_lambda = [0, 0.00001]

    parameters = basic_parameters()

    for size in batch_sizes:
        parameters['batch_size'] = size
        for weight in weight_decay:
            parameters['weight_decay'] = weight
            for l1 in l1_lambda:
                parameters['l1_lambda'] = l1
                run_model(parameters)


if __name__ == '__main__':
    main()
