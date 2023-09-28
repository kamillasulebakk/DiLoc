from model_runner import run_model


def basic_parameters():
    return {
        'N_samples': 70_000,
        'N_dipoles': 1,
        'determine_amplitude': False,
        'determine_area': False,
        'hidden_layers': [512, 256, 128, 64, 32],
        'learning_rate': 0.001,
        'N_epochs': 500,
        'noise_pct': 10,
        'custom_loss': True,
        'hl_act_func': 'tanh',
        'weights': 1
    }


def main():
    batch_sizes = [32]
    weight_decay = [0.1]
    momentum = [0.35]
    parameters = basic_parameters()
    parameters['l1_lambda'] = 0

    for size in batch_sizes:
        parameters['batch_size'] = size
        for weight in weight_decay:
            parameters['weight_decay'] = weight
            for mom in momentum:
                parameters['momentum'] = mom
                run_model(parameters)


if __name__ == '__main__':
    main()
