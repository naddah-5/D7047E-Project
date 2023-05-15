from os.path import exists


def update_params(new_params):
    # Initialize params as an empty dictionary
    params = {}

    # Try to read parameters from file if it exists
    try:
        with open('best_network_params.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                params[key] = value  # Assumes all values are floats
    except FileNotFoundError:
        pass  # It's okay if the file doesn't exist

    # Update parameters with new values
    for key in new_params:
        params[key] = new_params[key]

    # Write updated parameters to file
    with open('best_network_params.txt', 'w') as f:
        for key, value in params.items():
            f.write(f'{key}={value}\n')



