from model_runner_cnn import run_model

def basic_parameters():
    return {
        'N_samples': 70_000,
        'N_dipoles': 1,
        'determine_amplitude': False,
        'determine_area': False,
        'learning_rate': 0.001,
        'N_epochs': 500,
        'noise_pct': 10,
        'weights': 0.5,
        'custom_loss': True,
        'interpolate': True
    }


def main():
    batch_sizes = [32]
    weight_decay = [0.1] # old = 0
    momentum = [0.35] # old = 1e-6
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
